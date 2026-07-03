"""Train a byte-level BPE tokenizer on the TinyStories corpus.

The tokenizer is byte-complete: a ByteLevel pre-tokenizer maps every raw byte
to a visible unicode character before BPE runs, so any UTF-8 text can be
represented with `<unk>` only as a last-resort fallback. Two special tokens are
reserved:

    <unk>  index 0  — any character the tokenizer has not seen
    <eos>  index 1  — story boundary marker used during training

Usage:
    python -m src.tokenizer --train-file data/TinyStoriesV2-GPT4-train.txt
"""

import argparse

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

VOCAB_SIZE = 3000
SPECIAL_TOKENS = ["<unk>", "<eos>"]


def train_tokenizer(train_file: str, out_path: str, vocab_size: int = VOCAB_SIZE) -> Tokenizer:
    # 1. Build the tokenizer skeleton — an empty BPE model, no vocabulary yet.
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    # 2. Configure the trainer.
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,          # a merge must appear at least twice to be kept
        show_progress=True,
    )

    # 3. Train on the corpus: count adjacent-pair frequencies and iteratively
    #    merge the most common pair until vocab_size is reached.
    print(f"Training BPE tokenizer on {train_file} ...")
    tokenizer.train(files=[train_file], trainer=trainer)
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")

    # 4. Save for reuse.
    tokenizer.save(out_path)
    print(f"Tokenizer saved to {out_path}")

    # 5. Sanity check — reload and tokenize a sample.
    tokenizer = Tokenizer.from_file(out_path)
    sample = "Once upon a time, there was a little girl named Lily."
    enc = tokenizer.encode(sample)
    print("\n-- Tokenization example --")
    print(f"Text   : {sample}")
    print(f"Tokens : {enc.tokens}")
    print(f"IDs    : {enc.ids}")
    return tokenizer


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-file", required=True, help="Path to TinyStories train .txt")
    ap.add_argument("--out", default="tokenizer.json")
    ap.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    args = ap.parse_args()
    train_tokenizer(args.train_file, args.out, args.vocab_size)
