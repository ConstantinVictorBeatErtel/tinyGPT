# Tiny GPT

A decoder-only transformer language model built **from scratch in PyTorch** — no
modeling library, no `transformers` import. Every piece (byte-level BPE
tokenizer, sinusoidal positional encodings, multi-head causal self-attention,
pre-norm transformer blocks, weight-tied output projection, and an
autoregressive sampler with temperature / top-k / nucleus decoding) is wired up
by hand.

The model is trained on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories),
a corpus of short, simple children's stories, which makes it possible to train a
genuinely coherent language model on a laptop (Apple Silicon / MPS) in well under
an hour.

The largest configuration (~7.1M parameters) reaches a **validation perplexity of
8.08** and generates multi-sentence stories with consistent characters and
setting.

---

## Architecture

```
token IDs  (batch, seq)
    │
    ├─ token embedding                (vocab=3000 → d_model)   ·  ×√d_model
    ├─ + sinusoidal positional encoding
    │
    ├─ ┌─────────────────────────────────────────────┐
    │  │  LayerNorm → Multi-Head Causal Self-Attn → + │
    │  │  LayerNorm → Feed-Forward (GELU)         → + │   × N layers
    │  └─────────────────────────────────────────────┘
    │
    ├─ final LayerNorm
    ├─ output projection              (weight-tied with token embedding)
    │
    └─ logits  (batch, seq, vocab=3000)
```

Design choices:

- **Byte-level BPE tokenizer** (`tokenizers` library), vocab **3,000**, byte-complete
  so any UTF-8 string is representable; `<unk>` and `<eos>` are the only special tokens.
- **Sinusoidal positional encodings** — fixed sine/cosine waves, zero extra parameters.
- **Pre-norm blocks** (LayerNorm *before* each sub-layer) for stable training.
- **Multi-head causal self-attention** with a head dimension of 32; an explicit
  additive mask built with `-1e9` (not `-inf`) to sidestep an MPS NaN edge case.
- **Weight tying**: the output projection shares the token-embedding matrix, which
  removes a `vocab × d_model` parameter block and aligns the input/output spaces.
- **AdamW** (β = 0.9, 0.95), weight decay 0.1, **linear warmup → cosine decay** LR
  schedule, and **gradient-norm clipping** at 1.0.

### Model sizes

All three sizes share `vocab=3000`, `context=256`, `dropout=0.1`, and a head
dimension of 32; they differ only in width and depth.

| Model  | d_model | heads | layers | d_ff | Parameters |
|--------|:-------:|:-----:|:------:|:----:|:----------:|
| small  |   128   |   4   |   5    | 256  |  1,046,656 |
| medium |   256   |   8   |   5    | 512  |  3,404,032 |
| large  |   384   |  12   |   5    | 768  |  7,072,128 |

---

## Results

Each model trained for 5,000 steps (batch size 64, context length 256) on the
TinyStories corpus. Validation is on the held-out split.

| Model  | Parameters | Final val loss | Perplexity |
|--------|:----------:|:--------------:|:----------:|
| small  |  1,046,656 |     2.8764     |   17.75    |
| medium |  3,404,032 |     2.3249     |   10.23    |
| **large** | **7,072,128** | **2.0889** | **8.08** |

The scaling gap is clean: the large model is more than twice as certain about the
next token as the small one, and the improvement is qualitative as well as
quantitative — it maintains a consistent setting and cast of characters across
several sentences without the repetition loops the smaller models fall into. (For
reference, the random-prediction loss is `ln(3000) ≈ 8.006`.)

### Sample generations (large model)

> **Once upon a time,** in an ancient tree. In this forest lived many friends,
> like birds and the other bugs... The squirrels were all playing in a tree and
> were very careful not the birds that were not like birds, and they all lived
> together in a small village with a kind girl named Amy and the squirrels,
> always together and sharing.

### Temperature sweep (same prompt, medium model)

- **T = 0.5** — most coherent, reads like a real children's story, but repeats phrases.
- **T = 0.8–1.0** — the sweet spot: readable stories with reasonable variety.
- **T = 1.2** — more creative (invents a frog named Fin), but grammar and logic slip.

---

## Usage

```bash
pip install -r requirements.txt

# 0. Download the TinyStories text files into ./data
#    (TinyStoriesV2-GPT4-train.txt, TinyStoriesV2-GPT4-valid.txt)

# 1. Train the BPE tokenizer (vocab 3,000)
python -m src.tokenizer --train-file data/TinyStoriesV2-GPT4-train.txt --out tokenizer.json

# 2. Train the model (small | medium | large)
python -m src.train --size large \
    --train-file data/TinyStoriesV2-GPT4-train.txt \
    --valid-file data/TinyStoriesV2-GPT4-valid.txt \
    --tokenizer tokenizer.json \
    --save-path tinylm_checkpoint.pt

# 3. Generate text
python -m src.generate --checkpoint tinylm_checkpoint.pt \
    --tokenizer tokenizer.json \
    --prompt "Once upon a time," --temperature 0.8
```

## Project layout

```
src/
  tokenizer.py   Byte-level BPE training (vocab 3,000, <unk>/<eos>)
  data.py        Story splitting, tokenization, and the token-stream Dataset
  model.py       SinusoidalPositionalEncoding, TransformerBlock, TinyLM + sizes
  train.py       AdamW + warmup/cosine schedule, grad clipping, eval, checkpoints
  generate.py    Autoregressive sampling (temperature / top-k / nucleus)
```

## Notes

Trains comfortably on a single Apple Silicon GPU via PyTorch MPS (also runs on
CUDA or CPU). The masking and positional-encoding code deliberately avoids the
`is_causal` fast path because it can produce NaNs on MPS; passing an explicit
additive mask is numerically stable and just as correct.
