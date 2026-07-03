"""Tiny GPT — a decoder-only transformer language model built from scratch in PyTorch.

The model follows the standard GPT recipe:

    token IDs (batch, seq)
        -> token embeddings         (batch, seq, d_model)
        -> + sinusoidal positional encoding
        -> N x TransformerBlock      [pre-norm, causal self-attention + FFN]
        -> final LayerNorm
        -> output projection         (weight-tied with the token embedding)
        -> logits (batch, seq, vocab_size)

Nothing here is imported from a modeling library — the attention, masking,
positional encodings, and weight tying are all wired up by hand.
"""

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal position signals — no parameters to learn.

    Sine / cosine waves of different frequencies encode each absolute position,
    exactly as in "Attention Is All You Need".
    """

    def __init__(self, d_model: int, max_len: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block:

    - LayerNorm -> multi-head causal self-attention -> residual
    - LayerNorm -> feed-forward network -> residual
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        # Attention — pass an explicit additive mask (no is_causal flag, which
        # avoids an MPS NaN bug on Apple Silicon).
        normed = self.norm1(x)
        attn_out, _ = self.attn(
            normed, normed, normed,
            attn_mask=causal_mask,  # explicit mask is sufficient
        )
        x = x + attn_out

        # Feed-forward
        x = x + self.ff(self.norm2(x))
        return x


class TinyLM(nn.Module):
    """Decoder-only transformer language model.

    Weight tying: the output projection shares weights with the token
    embedding, which removes a parameter block and aligns the input/output
    vector spaces.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        context_length: int,
        dropout: float,
    ):
        super().__init__()

        self.context_length = context_length
        self.d_model = d_model

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, context_length, dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.output_proj.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _make_causal_mask(self, seq_len: int, device) -> torch.Tensor:
        """Upper-triangular mask: position i can only attend to 0..i.

        Uses -1e9 instead of -inf for MPS numerical stability.
        """
        return torch.triu(
            torch.full((seq_len, seq_len), -1e9, device=device),
            diagonal=1,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        batch, seq_len = token_ids.shape

        x = self.token_emb(token_ids) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        causal_mask = self._make_causal_mask(seq_len, token_ids.device)

        for block in self.blocks:
            x = block(x, causal_mask)

        x = self.final_norm(x)
        logits = self.output_proj(x)  # (batch, seq_len, vocab_size)
        return logits


# --- Named configurations (small / medium / large) --------------------------
# All three share vocab=3000, context=256, dropout=0.1 and a head dimension of
# 32, and differ only in width / depth. Parameter counts are trainable params
# with weight tying.
CONFIGS = {
    "small":  dict(d_model=128, n_heads=4,  n_layers=5, d_ff=256),  # 1,046,656 params
    "medium": dict(d_model=256, n_heads=8,  n_layers=5, d_ff=512),  # 3,404,032 params
    "large":  dict(d_model=384, n_heads=12, n_layers=5, d_ff=768),  # 7,072,128 params
}


def build_model(
    size: str = "medium",
    vocab_size: int = 3000,
    context_length: int = 256,
    dropout: float = 0.1,
) -> TinyLM:
    cfg = CONFIGS[size]
    return TinyLM(
        vocab_size=vocab_size,
        context_length=context_length,
        dropout=dropout,
        **cfg,
    )
