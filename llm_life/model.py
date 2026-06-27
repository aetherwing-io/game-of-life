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
    if device == "cpu":
        dtype = torch.float32
    elif "gemma" in model_name.lower():
        dtype = torch.bfloat16  # Gemma is bf16-native; fp16 overflows its activations to NaN
    else:
        dtype = torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    model.to(device)
    model.eval()
    return model, tok, device


def load_mlm(model_name: str, device: str):
    """Load a masked language model and its tokenizer (for the bidirectional CA)."""
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    dtype = torch.float32 if device == "cpu" else torch.float16
    model = AutoModelForMaskedLM.from_pretrained(model_name, dtype=dtype)
    model.to(device)
    model.eval()
    return model, tok, device


def dead_token_id_mlm(model, tokenizer, device: str) -> int:
    """Ground-state token for an MLM: the argmax prediction for a single masked
    position with no other content (just CLS [MASK] SEP)."""
    with torch.no_grad():
        ids = [tokenizer.cls_token_id, tokenizer.mask_token_id, tokenizer.sep_token_id]
        inp = torch.tensor([ids], device=device)
        logits = model(inp).logits[0, 1]  # the masked position
        return int(logits.argmax().item())


def dead_token_id(model, tokenizer, bos_token: int, device: str) -> int:
    """The model's "ground state" token: the argmax prediction from BOS alone.

    This is the natural dead/quiescent state -- the token the model emits from an
    empty context -- rather than an arbitrary choice of PAD/space.
    """
    with torch.no_grad():
        inp = torch.tensor([[bos_token]], device=device)
        logits = model(inp).logits[0, -1]
        return int(logits.argmax().item())


# --------------------------------------------------------------------------- #
# MLX backend (Apple-Silicon-native quantized models via mlx_lm)
# --------------------------------------------------------------------------- #
# mlx_lm models run on Metal through Apple's MLX framework, not torch. We load
# them here and bridge their logits back to torch at the automaton boundary
# (see MLXLLMAutomaton), so the rest of the torch-based harness -- sampling,
# coupled noise, metrics -- is unchanged. This lets us iterate genuinely
# low-bit (e.g. 2-bit ternary) checkpoints as cellular automata.
def load_mlx(model_name: str):
    """Load an MLX-quantized causal LM via mlx_lm. Returns (model, tokenizer)."""
    from mlx_lm import load

    return load(model_name)


def mlx_vocab_and_dead(model, bos_token: int) -> tuple[int, int]:
    """One BOS-only forward gives both the vocab size and the dead/ground-state
    token (argmax from BOS), the MLX analogue of dead_token_id()."""
    import mlx.core as mx

    out = model(mx.array([[bos_token]]))[0, -1]
    mx.eval(out)
    return int(out.shape[-1]), int(mx.argmax(out).item())


def mlx_input_embeddings(model, vocab: int):
    """(V, d) dense input embeddings from an mlx_lm model as float32 numpy, for
    the embedding-PCA colour table. Calling the (possibly quantized) embedding
    layer on every id dequantizes on the fly, so this works for 2-bit models."""
    import mlx.core as mx
    import numpy as np

    emb = model.model.embed_tokens(mx.arange(vocab))  # (V, d), dequantized
    mx.eval(emb)
    return np.array(emb, copy=False).astype(np.float32)
