"""Tiny GPT — a decoder-only transformer language model built from scratch."""

from .model import TinyLM, build_model, CONFIGS

__all__ = ["TinyLM", "build_model", "CONFIGS"]
