"""Train Tiny GPT on TinyStories.

Runs a fixed number of optimizer steps with AdamW, a linear-warmup / cosine-decay
learning-rate schedule, and gradient-norm clipping. Validation loss and
perplexity are logged periodically, and the best model plus its config are saved
to a single checkpoint.

Usage:
    python -m src.train --size medium \
        --train-file data/TinyStoriesV2-GPT4-train.txt \
        --valid-file data/TinyStoriesV2-GPT4-valid.txt \
        --tokenizer tokenizer.json
"""

import argparse
import math
import time

import torch
import torch.nn as nn
from tokenizers import Tokenizer

from .model import build_model, CONFIGS
from .data import load_and_tokenize, build_dataloaders

# --- Training config (defaults) ---------------------------------------------
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.1
MAX_STEPS = 5_000
WARMUP_STEPS = 300
GRAD_CLIP_NORM = 1.0
LOG_EVERY = 50
EVAL_EVERY = 500
EVAL_BATCHES = 64

CONTEXT_LENGTH = 256
BATCH_SIZE = 64
MAX_TRAIN_STORIES = 500_000


def get_lr(step, max_steps, warmup_steps, base_lr):
    """Linear warmup -> cosine decay to 10% of the base LR."""
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * 0.1 + (base_lr - base_lr * 0.1) * cosine


@torch.no_grad()
def evaluate(model, loader, criterion, device, max_batches=None):
    """Returns average cross-entropy loss and perplexity. NaN-safe."""
    model.eval()
    total_loss = 0.0
    total_batches = 0

    for i, (inputs, targets) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
        if torch.isnan(loss):
            continue
        total_loss += loss.item()
        total_batches += 1

    if total_batches == 0:
        return float("nan"), float("nan")

    avg_loss = total_loss / total_batches
    ppl = math.exp(min(avg_loss, 20))
    model.train()
    return avg_loss, ppl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=list(CONFIGS), default="medium")
    ap.add_argument("--train-file", required=True)
    ap.add_argument("--valid-file", required=True)
    ap.add_argument("--tokenizer", default="tokenizer.json")
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--max-train-stories", type=int, default=MAX_TRAIN_STORIES)
    ap.add_argument("--save-path", default="tinylm_checkpoint.pt")
    args = ap.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    tokenizer = Tokenizer.from_file(args.tokenizer)
    vocab_size = tokenizer.get_vocab_size()

    print("Tokenizing training set...")
    train_tokens = load_and_tokenize(args.train_file, tokenizer, args.max_train_stories)
    print("Tokenizing validation set...")
    valid_tokens = load_and_tokenize(args.valid_file, tokenizer)

    train_loader, valid_loader = build_dataloaders(
        train_tokens, valid_tokens, CONTEXT_LENGTH, args.batch_size
    )

    model = build_model(args.size, vocab_size=vocab_size, context_length=CONTEXT_LENGTH).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.size} | Total trainable parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.95),
        weight_decay=WEIGHT_DECAY,
        eps=1e-8,
    )

    print(f"Random baseline loss ln({vocab_size}) = {math.log(vocab_size):.3f}")
    print(f"Your first logged loss should be close to this.\n")

    train_iter = iter(train_loader)
    step = 0
    best_val = float("inf")
    start = time.time()

    print(f"{'Step':>6}  {'LR':>8}  {'Train Loss':>10}  {'Val Loss':>9}  {'PPL':>8}")
    print("-" * 55)

    model.train()
    while step < args.max_steps:
        try:
            inputs, targets = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            inputs, targets = next(train_iter)
        inputs, targets = inputs.to(device), targets.to(device)

        lr = get_lr(step, args.max_steps, WARMUP_STEPS, LEARNING_RATE)
        for g in optimizer.param_groups:
            g["lr"] = lr

        logits = model(inputs)
        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()

        if step % LOG_EVERY == 0:
            print(f"{step:>6}  {lr:>8.2e}  {loss.item():>10.4f}")

        if step > 0 and step % EVAL_EVERY == 0:
            val_loss, ppl = evaluate(model, valid_loader, criterion, device, EVAL_BATCHES)
            print(f"{step:>6}  {lr:>8.2e}  {'':>10}  {val_loss:>9.4f}  {ppl:>8.2f}")
            if val_loss < best_val:
                best_val = val_loss
                torch.save(
                    {
                        "config": dict(
                            size=args.size,
                            vocab_size=vocab_size,
                            context_length=CONTEXT_LENGTH,
                            **CONFIGS[args.size],
                        ),
                        "model_state_dict": model.state_dict(),
                        "step": step,
                        "val_loss": val_loss,
                    },
                    args.save_path,
                )
        step += 1

    val_loss, ppl = evaluate(model, valid_loader, criterion, device)
    elapsed = (time.time() - start) / 60
    print("-" * 55)
    print(f"Done in {elapsed:.1f} min | final val loss {val_loss:.4f} | PPL {ppl:.2f}")


if __name__ == "__main__":
    main()
