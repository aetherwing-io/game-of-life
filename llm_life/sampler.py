"""Stochastic sampling primitives with *controllable* noise.

The whole experiment rests on being able to (a) sample at a given temperature
and (b) drive two replicas with the *identical* noise realization so that any
divergence between them is attributable to the initial perturbation alone
(coupled-noise damage spreading). We get both from the Gumbel-max trick:

    sample ~ softmax(logits / T)   <=>   argmax_k ( logits_k / T + G_k )

with G_k i.i.d. Gumbel(0, 1). Passing in a fixed Gumbel tensor makes the draw
deterministic given the logits, which is exactly what the Lyapunov / damage
measurement requires.
"""

from __future__ import annotations

import torch

_EPS = 1e-20


def gumbel_like(logits: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Draw a Gumbel(0,1) tensor shaped like ``logits``.

    If ``generator`` lives on a different device than ``logits`` (e.g. a CPU
    generator with MPS logits, since MPS has no device-side generator), the
    uniforms are drawn on the generator's device and moved over.
    """
    if generator is not None and generator.device.type != logits.device.type:
        u = torch.rand(logits.shape, dtype=logits.dtype, generator=generator).to(logits.device)
    else:
        u = torch.rand(logits.shape, dtype=logits.dtype, device=logits.device, generator=generator)
    return -torch.log(-torch.log(u + _EPS) + _EPS)


def sample(
    logits: torch.Tensor,
    temperature: float,
    noise: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Return argmax token ids for ``logits`` of shape (..., V).

    temperature == 0 -> deterministic argmax (noise ignored).
    temperature  > 0 -> Gumbel-max sampling. If ``noise`` is supplied it is
    reused verbatim (this is what couples two replicas); otherwise a fresh
    Gumbel tensor is drawn from ``generator``.
    """
    if temperature <= 0.0:
        return logits.argmax(dim=-1)
    if noise is None:
        noise = gumbel_like(logits, generator=generator)
    return (logits / temperature + noise).argmax(dim=-1)
