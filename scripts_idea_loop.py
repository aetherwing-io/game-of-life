"""Idea-space iteration — THE RULER (calibration first, measurement second).

Motivation (see PERSONA_BASIN.md and the transformer-circuits "global workspace"
read): every dynamical result in this repo was measured on the *token lattice* —
the model's spoken output. But most of the model's computation never reaches the
tokens; it lives in the hidden state. This script is the apparatus for iterating
and measuring the *idea* trajectory (a low-dimensional projection of the hidden
state) instead of the word trajectory.

The lesson this repo keeps re-learning (S1, S24, S25): **build the instrument and
validate it on known-answer cases BEFORE trusting it on the unknown case.** A
continuous vector loop will show "attractors" trivially — any contraction decays
toward a dominant direction whether or not the trained model means anything by it.
So the deliverable here is not a measurement. It is the RULER: a calibration gate
with true-negative and true-positive controls that the metrics must pass.

    python scripts_idea_loop.py calibrate --model EleutherAI/pythia-160m
        -> runs every control, asserts the metrics classify them correctly,
           prints a PASS/FAIL table, exits nonzero on any failure. This is the
           thing you run first and must see green.

    python scripts_idea_loop.py measure --model EleutherAI/pythia-160m
        -> the real trained-model idea-loop. Its numbers are ONLY meaningful if
           `calibrate` passed on the same model. (Skeleton — the science goes here.)

Restricted to `--arch causal` torch models (they expose output_hidden_states); the
MLX / masked / local paths are out of scope for v1.
"""

from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch

from llm_life.run import _build, _gen, _random_init


# ---------------------------------------------------------------------------
# 1. Querying the idea space: read hidden states, one keyword on the forward pass.
# ---------------------------------------------------------------------------

@torch.no_grad()
def read_ideas(model, bos_token: int, state: torch.Tensor, layer: int, device: str) -> np.ndarray:
    """Hidden state at `layer` for each of the L sites, as (L, d) float32.

    Mirrors LLMAutomaton.logits: prepend BOS, take positions for sites 0..L-1.
    `output_hidden_states=True` is the entire "query the idea space" step — the
    model already computed these; we just ask it not to discard them.
    """
    L = state.shape[0]
    inp = torch.empty(1, L + 1, dtype=torch.long, device=device)
    inp[0, 0] = bos_token
    inp[0, 1:] = state
    out = model(inp, output_hidden_states=True)
    hs = out.hidden_states[layer]          # (1, L+1, d)
    return hs[0, 1:].float().cpu().numpy()  # (L, d): representation at each site


def default_layer(model) -> int:
    """~1/3 of the way in, where the paper puts workspace onset. hidden_states is
    length num_layers+1 (index 0 = embeddings), so a layer index in [1, num]."""
    n = int(getattr(model.config, "num_hidden_layers", None)
            or model.config.text_config.num_hidden_layers)
    return max(1, round(n / 3))


# ---------------------------------------------------------------------------
# 2. The concept basis (the "board"): top-k PCA of the hidden state at `layer`.
#    v1 proxy for the paper's J-lens; ~15 dirs matches its ~10-25 capacity.
# ---------------------------------------------------------------------------

def build_basis(model, bos_token, auto, layer, device, *, k=16, n_states=64,
                length=48, vocab=50000, seed=0):
    """Collect per-site hidden states over random token states, PCA -> (mean, comps).

    NB build the basis per-model: the true-negative control needs the random-weight
    model's OWN idea-space to read structureless, not the trained model's basis.
    """
    rows = []
    for i in range(n_states):
        s = _random_init(auto, length, vocab, device, seed + i)
        rows.append(read_ideas(model, bos_token, s, layer, device))
    X = np.concatenate(rows, axis=0)               # (n_states*L, d)
    mean = X.mean(axis=0)
    Xc = X - mean
    # economy SVD; components are right-singular vectors
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:k]                                  # (k, d)
    return mean.astype(np.float32), comps.astype(np.float32)


def project(hidden: np.ndarray, mean: np.ndarray, comps: np.ndarray) -> np.ndarray:
    """(L, d) hidden -> (k,) idea-vector for the generation (mean-pooled board)."""
    v = (hidden - mean) @ comps.T   # (L, k)
    return v.mean(axis=0)           # the board contents this generation


# ---------------------------------------------------------------------------
# 3. Metrics on an idea-trajectory (T, k). Deliberately few and simple.
# ---------------------------------------------------------------------------

def _unit(a, eps=1e-8):
    return a / (np.linalg.norm(a, axis=-1, keepdims=True) + eps)


def freeze_score(traj: np.ndarray) -> float:
    """Mean cosine SIMILARITY of consecutive idea-vectors. ~1 => fixed point."""
    u = _unit(traj)
    return float((u[1:] * u[:-1]).sum(axis=1).mean())


def period2_score(traj: np.ndarray) -> float:
    """cos-sim at lag 2 MINUS lag 1. High & positive => alternation (period-2)."""
    u = _unit(traj)
    lag1 = (u[1:] * u[:-1]).sum(axis=1).mean()
    lag2 = (u[2:] * u[:-2]).sum(axis=1).mean()
    return float(lag2 - lag1)


def wander_score(traj: np.ndarray) -> float:
    """Effective spread: mean pairwise cosine DISTANCE. High => wandering/chaos."""
    u = _unit(traj)
    S = u @ u.T
    iu = np.triu_indices(len(u), k=1)
    return float(1.0 - S[iu].mean())


def idea_damage(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine distance between two idea-trajectories over time (for the S23-lifted
    damage test: do two runs differing by one token re-converge in idea-space?)."""
    ua, ub = _unit(a), _unit(b)
    return 1.0 - (ua * ub).sum(axis=1)   # (T,)


def classify(traj: np.ndarray, th) -> str:
    if freeze_score(traj) >= th["freeze"]:
        return "frozen"
    if period2_score(traj) >= th["period"]:
        return "cyclic"
    if wander_score(traj) >= th["wander"]:
        return "wandering"
    return "unstructured"


# Thresholds are calibrated by the synthetic cases below; exposed so `measure`
# uses the exact instrument the ruler validated.
THRESHOLDS = {"freeze": 0.90, "period": 0.30, "wander": 0.25}


# ---------------------------------------------------------------------------
# 4. THE RULER. Known-answer controls; every one is asserted.
# ---------------------------------------------------------------------------

def _synthetic_cases(k=16, T=60, seed=0):
    """Trajectories with KNOWN structure, to validate the metrics themselves."""
    rng = np.random.default_rng(seed)
    fixed = np.tile(_unit(rng.standard_normal(k)), (T, 1)) + 0.02 * rng.standard_normal((T, k))
    a, b = _unit(rng.standard_normal(k)), _unit(rng.standard_normal(k))
    cycle = np.stack([a if t % 2 == 0 else b for t in range(T)]) + 0.02 * rng.standard_normal((T, k))
    randw = rng.standard_normal((T, k))
    return {"fixed": fixed, "cycle": cycle, "random": randw}


def _planted_pipeline_traj(model, bos, auto, layer, mean, comps, device, *,
                           target, clamp_frac=0.5, length=48, steps=40, seed=1):
    """TRUE-POSITIVE through the full pipeline: clamp a fraction of sites to a
    fixed target token each step, so the state -- and its read ideas -- MUST
    converge. Plants a fixed-point via the input; no arch-specific hooks needed.
    (The activation-steering version is the Option-B closed loop; TODO.)"""
    vocab = int(getattr(model.config, "vocab_size", None) or model.config.text_config.vocab_size)
    g = _gen(device, seed)
    cur = _random_init(auto, length, vocab, device, seed)
    n_clamp = int(clamp_frac * length)
    traj = []
    for _ in range(steps):
        cur = cur.clone()
        cur[:n_clamp] = target                      # hold part of the board fixed
        nxt, _ = auto.step(cur, temperature=0.7, absorbing=False, generator=g)
        cur = nxt
        traj.append(project(read_ideas(model, bos, cur, layer, device), mean, comps))
    return np.stack(traj)


def _real_pipeline_traj(model, bos, auto, layer, mean, comps, device, *,
                        temp=0.7, length=48, steps=60, seed=2, absorbing=False):
    """The trained model's own idea-trajectory (also reused as the random-weight
    true-negative when `model` is the untrained twin)."""
    vocab = int(getattr(model.config, "vocab_size", None) or model.config.text_config.vocab_size)
    g = _gen(device, seed)
    init = _random_init(auto, length, vocab, device, seed)
    states, _ = auto.trajectory(init, steps, temp, absorbing, generator=g)
    traj = [project(read_ideas(model, bos, states[t], layer, device), mean, comps)
            for t in range(states.shape[0])]
    return np.stack(traj)


def _random_weight_twin(model_name: str, device: str):
    """Same architecture, UNTRAINED weights. If the idea-loop shows the same
    structure here as on the trained model, the structure is linear algebra, not
    learned knowledge -- the artifact this control exists to catch."""
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(model_name)
    twin = AutoModelForCausalLM.from_config(cfg).to(device).eval()
    return twin


def calibrate(args) -> int:
    device = None
    auto, tok, model, info = _build(SimpleNamespace(model=args.model, arch="causal", device=args.device))
    if info["arch"] != "causal":
        print("idea-loop v1 supports --arch causal torch models only.")
        return 2
    device = info["device"]
    bos = auto.bos_token
    layer = args.layer if args.layer is not None else default_layer(model)
    th = THRESHOLDS
    results = []  # (name, expectation, metric, value, pass?)

    # -- (a) metric calibration on synthetic trajectories (arch-independent) -----
    syn = _synthetic_cases(k=args.k)
    results.append(("synthetic:fixed", "frozen", "freeze",
                    freeze_score(syn["fixed"]), freeze_score(syn["fixed"]) >= th["freeze"]))
    results.append(("synthetic:cycle", "cyclic", "period",
                    period2_score(syn["cycle"]), period2_score(syn["cycle"]) >= th["period"]))
    results.append(("synthetic:random", "not-frozen", "freeze",
                    freeze_score(syn["random"]), freeze_score(syn["random"]) < th["freeze"]))
    # idea-damage metric: identical -> ~0, independent -> high
    d_same = idea_damage(syn["fixed"], syn["fixed"]).mean()
    d_diff = idea_damage(syn["random"], np.roll(syn["random"], 1, axis=0)).mean()
    results.append(("synthetic:damage-identical", "~0", "damage", d_same, d_same < 0.05))
    results.append(("synthetic:damage-independent", "high", "damage", d_diff, d_diff > 0.5))

    # -- (b) whole-pipeline controls on the real forward pass --------------------
    mean, comps = build_basis(model, bos, auto, layer, device, k=args.k, length=args.length)

    # TRUE-POSITIVE: planted attractor must be detected as frozen.
    tp = _planted_pipeline_traj(model, bos, auto, layer, mean, comps, device,
                                target=int(auto.dead_token), length=args.length)
    results.append(("pipeline:planted-attractor", "frozen", "freeze",
                    freeze_score(tp), freeze_score(tp) >= th["freeze"]))

    # TRUE-NEGATIVE: random-weight twin must NOT show learned structure.
    twin = _random_weight_twin(info["model"], device)
    tmean, tcomps = build_basis(twin, bos, auto, layer, device, k=args.k, length=args.length)
    tn = _real_pipeline_traj(twin, bos, auto, layer, tmean, tcomps, device, length=args.length)
    results.append(("pipeline:random-weights", "not-frozen", "freeze",
                    freeze_score(tn), freeze_score(tn) < th["freeze"]))

    # -- report ------------------------------------------------------------------
    print(f"\nRULER  model={info['model']}  layer={layer}/{default_layer(model)*3}"
          f"  k={args.k}  device={device}")
    print(f"{'control':32s} {'expect':12s} {'metric':8s} {'value':>8s}  gate")
    print("-" * 74)
    ok = True
    for name, exp, metric, val, passed in results:
        ok &= passed
        print(f"{name:32s} {exp:12s} {metric:8s} {val:8.3f}  {'PASS' if passed else 'FAIL'}")
    print("-" * 74)
    print("RULER: " + ("GREEN — metrics + pipeline validated; `measure` is trustworthy."
                        if ok else "RED — do NOT trust `measure` until these pass."))
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# 5. THE MEASUREMENT (skeleton — only meaningful after `calibrate` is GREEN).
# ---------------------------------------------------------------------------

def measure(args) -> int:
    auto, tok, model, info = _build(SimpleNamespace(model=args.model, arch="causal", device=args.device))
    device = info["device"]
    bos = auto.bos_token
    layer = args.layer if args.layer is not None else default_layer(model)
    mean, comps = build_basis(model, bos, auto, layer, device, k=args.k, length=args.length)

    traj = _real_pipeline_traj(model, bos, auto, layer, mean, comps, device,
                               temp=args.temp, length=args.length, steps=args.steps)
    label = classify(traj, THRESHOLDS)
    print(f"\nMEASURE  model={info['model']}  layer={layer}  T={args.temp}  steps={args.steps}")
    print(f"  freeze={freeze_score(traj):.3f}  period={period2_score(traj):.3f} "
          f"  wander={wander_score(traj):.3f}")
    print(f"  idea-trajectory class: {label}")
    print("  (trust this ONLY if `calibrate` printed GREEN for this model.)")
    # TODO next: idea-space damage spreading (two runs, one-token diff, shared
    # Gumbel noise, idea_damage over time -> S23 echo-state test in idea-space);
    # then Option B -- activation-steering closed loop; then the complexity plane
    # on the discretized idea-trajectory.
    return 0


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("calibrate", "measure"):
        sp = sub.add_parser(name)
        sp.add_argument("--model", default="EleutherAI/pythia-160m")
        sp.add_argument("--device", default="auto")
        sp.add_argument("--layer", type=int, default=None, help="hidden_states index; default ~1/3 depth")
        sp.add_argument("--k", type=int, default=16, help="idea-space dimensions (paper: ~10-25)")
        sp.add_argument("--length", type=int, default=48)
        sp.add_argument("--steps", type=int, default=60)
        sp.add_argument("--temp", type=float, default=0.7)
    args = p.parse_args()
    fn = {"calibrate": calibrate, "measure": measure}[args.cmd]
    sys.exit(fn(args))


if __name__ == "__main__":
    main()
