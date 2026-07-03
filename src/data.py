"""Data pipeline: turn raw TinyStories text into next-token training batches.

Each split is read, broken into individual stories (the TinyStories files use
`<|endoftext|>` as the delimiter), tokenized in batches, and concatenated into
a single long token stream with an `<eos>` token between stories. That stream is
then served as overlapping `context_length + 1` windows, where the model learns
to predict token i+1 from tokens 0..i.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer


def load_and_tokenize(filepath: str, tokenizer: Tokenizer, max_stories=None):
    """Read the raw text file, split into stories, tokenize each, and
    concatenate into one long token stream separated by <eos>."""
    eos_id = tokenizer.token_to_id("<eos>")

    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read()

    # Stories are separated by "<|endoftext|>" in the TinyStories files.
    stories = raw.split("<|endoftext|>")
    stories = [s.strip() for s in stories if s.strip()]  # drop empty strings

    if max_stories is not None:
        stories = stories[:max_stories]

    print(f"  Loaded {len(stories):,} stories from {filepath}")

    # Tokenize in batches — much faster than one story at a time.
    encodings = tokenizer.encode_batch(stories)

    # Concatenate all token-ID lists with <eos> between each story.
    token_stream = []
    for enc in encodings:
        token_stream.extend(enc.ids)
        token_stream.append(eos_id)

    return token_stream


class TokenStreamDataset(Dataset):
    """Serves overlapping (context_length + 1) windows from a flat token stream."""

    def __init__(self, token_stream, context_length: int):
        self.tokens = torch.tensor(token_stream, dtype=torch.long)
        self.context_length = context_length

    def __len__(self):
        # Each sample needs context_length + 1 tokens (input plus one shifted target).
        return len(self.tokens) - self.context_length

    def __getitem__(self, idx):
        chunk = self.tokens[idx : idx + self.context_length + 1]
        input = chunk[:-1]   # tokens 0..N-1
        target = chunk[1:]   # tokens 1..N   (shifted by one)
        return input, target


def build_dataloaders(train_tokens, valid_tokens, context_length, batch_size):
    train_ds = TokenStreamDataset(train_tokens, context_length)
    valid_ds = TokenStreamDataset(valid_tokens, context_length)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=False,   # pin_memory doesn't help on MPS
        drop_last=True,     # drop the incomplete final batch for cleaner training
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
    )
    return train_loader, valid_loader
