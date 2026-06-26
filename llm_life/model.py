"""Model loading + device selection (Apple Silicon / CUDA / CPU)."""

from __future__ import annotations

import torch


def pick_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load(model_name: str, device: str):
    """Load a base (non-instruct) causal LM and its tokenizer.

    Returns (model, tokenizer, device). The model is put in eval mode; gradients
    are never needed.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    dtype = torch.float32 if device == "cpu" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    model.to(device)
    model.eval()
    return model, tok, device


def dead_token_id(model, tokenizer, bos_token: int, device: str) -> int:
    """The model's "ground state" token: the argmax prediction from BOS alone.

    This is the natural dead/quiescent state -- the token the model emits from an
    empty context -- rather than an arbitrary choice of PAD/space.
    """
    with torch.no_grad():
        inp = torch.tensor([[bos_token]], device=device)
        logits = model(inp).logits[0, -1]
        return int(logits.argmax().item())
