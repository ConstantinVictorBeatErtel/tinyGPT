"""Autoregressive text generation from a trained Tiny GPT checkpoint.

Supports temperature scaling, top-k filtering, and nucleus (top-p) sampling.
top-k and top-p can be combined; top-k is applied first.

Usage:
    python -m src.generate --checkpoint tinylm_checkpoint.pt \
        --tokenizer tokenizer.json --prompt "Once upon a time," --temperature 0.8
"""

import argparse

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from .model import TinyLM


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    top_k: int = None,
    top_p: float = None,
    device: str = "cpu",
) -> str:
    """Autoregressively generate text from a prompt.

    Args:
        temperature : >1 = more random, <1 = more conservative, 1 = raw distribution
        top_k       : keep only the k highest-probability tokens before sampling
        top_p       : keep the smallest set of tokens whose cumulative prob >= p (nucleus)
    """
    model.eval()

    input_ids = tokenizer.encode(prompt).ids
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)  # (1, prompt_len)
    eos_id = tokenizer.token_to_id("<eos>")

    for _ in range(max_new_tokens):
        # Crop context to the model's max context length.
        ids_cond = ids[:, -model.context_length:]

        # Forward pass — only the logits at the last position are needed.
        logits = model(ids_cond)          # (1, seq_len, vocab_size)
        logits = logits[:, -1, :]         # (1, vocab_size) — last token only

        # Temperature scaling: dividing sharpens (T<1) or flattens (T>1).
        logits = logits / temperature

        # Top-k filtering: zero out everything outside the top-k.
        if top_k is not None:
            top_k = min(top_k, logits.size(-1))
            kth_val = torch.topk(logits, top_k).values[:, -1, None]
            logits = logits.masked_fill(logits < kth_val, float("-inf"))

        # Nucleus (top-p) filtering: keep the smallest set of tokens whose
        # cumulative probability >= top_p.
        if top_p is not None:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            # Shift right by one so we always keep at least the top token.
            remove_mask = cumprobs - F.softmax(sorted_logits, dim=-1) > top_p
            sorted_logits[remove_mask] = float("-inf")
            # Scatter filtered logits back to original ordering.
            logits = torch.zeros_like(logits).scatter_(1, sorted_idx, sorted_logits)

        # Sample the next token.
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)  # (1, 1)
        ids = torch.cat([ids, next_id], dim=1)

        if next_id.item() == eos_id:
            break

    # Decode the continuation only (exclude the original prompt).
    generated_ids = ids[0, len(input_ids):].tolist()
    return tokenizer.decode(generated_ids)


def load_checkpoint(checkpoint_path, tokenizer, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = TinyLM(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        d_ff=cfg["d_ff"],
        context_length=cfg["context_length"],
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", default="tokenizer.json")
    ap.add_argument("--prompt", default="Once upon a time,")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    args = ap.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    tokenizer = Tokenizer.from_file(args.tokenizer)
    model = load_checkpoint(args.checkpoint, tokenizer, device)

    text = generate(
        model, tokenizer, args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=str(device),
    )
    print(args.prompt + text)
