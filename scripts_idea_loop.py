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

from llm_life.automaton import StepNoise
from llm_life.sampler import gumbel_like
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


def n_layers(model) -> int:
    return int(getattr(model.config, "num_hidden_layers", None)
               or model.config.text_config.num_hidden_layers)


def default_layer(model) -> int:
    """~1/3 of the way in, where the paper puts workspace onset. hidden_states is
    length num_layers+1 (index 0 = embeddings), so a layer index in [1, num].
    NB on a shallow model this is arbitrary — prefer the gap-sweep to locate the
    layer where the *learned* signal (trained - random) is actually largest."""
    return max(1, round(n_layers(model) / 3))


@torch.no_grad()
def read_ideas_all(model, bos_token: int, state: torch.Tensor, device: str) -> np.ndarray:
    """All layers at once: (n_layers+1, L, d). One forward pass, every layer —
    so a layer sweep costs the same as reading a single layer."""
    L = state.shape[0]
    inp = torch.empty(1, L + 1, dtype=torch.long, device=device)
    inp[0, 0] = bos_token
    inp[0, 1:] = state
    out = model(inp, output_hidden_states=True)
    return np.stack([h[0, 1:].float().cpu().numpy() for h in out.hidden_states])


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


def project_persite(hidden: np.ndarray, mean: np.ndarray, comps: np.ndarray) -> np.ndarray:
    """(L, d) -> (L, k): keep every site separate (NO pooling). The control."""
    return (hidden - mean) @ comps.T


def persite_freeze(traj_hidden, mean, comps) -> float:
    """freeze computed PER SITE then averaged — never mixing sites. If this is
    high too, the pooled stability is real; if it collapses, pooled freeze was a
    central-limit averaging artifact."""
    V = np.stack([project_persite(h, mean, comps) for h in traj_hidden])  # (T,L,k)
    u = V / (np.linalg.norm(V, axis=-1, keepdims=True) + 1e-8)
    return float((u[1:] * u[:-1]).sum(axis=-1).mean())


def _pca(X: np.ndarray, k: int):
    mean = X.mean(axis=0)
    _, _, Vt = np.linalg.svd(X - mean, full_matrices=False)
    return mean.astype(np.float32), Vt[:k].astype(np.float32)


def build_basis_all(model, bos, auto, device, *, k=16, n_states=64, length=48,
                    vocab=50000, seed=0):
    """PCA basis for EVERY layer from one shared set of forward passes.
    Returns list indexed by hidden_states layer -> (mean, comps)."""
    per_layer = None
    for i in range(n_states):
        s = _random_init(auto, length, vocab, device, seed + i)
        allh = read_ideas_all(model, bos, s, device)         # (nL+1, L, d)
        if per_layer is None:
            per_layer = [[] for _ in range(allh.shape[0])]
        for li in range(allh.shape[0]):
            per_layer[li].append(allh[li])
    return [_pca(np.concatenate(rows, axis=0), k) for rows in per_layer]


def freeze_by_layer(model, bos, auto, bases, traj_states, device) -> np.ndarray:
    """freeze_score at every layer for a token trajectory. Reads all layers per
    state once; returns (n_layers+1,)."""
    ideas = np.stack([read_ideas_all(model, bos, s, device) for s in traj_states])  # (T,nL+1,L,d)
    out = np.empty(len(bases))
    for li, (mean, comps) in enumerate(bases):
        traj = np.stack([project(ideas[t, li], mean, comps) for t in range(ideas.shape[0])])
        out[li] = freeze_score(traj)
    return out


@torch.no_grad()
def idea_damage_run(model, bos, auto, layer, mean, comps, device, *,
                    temp=0.7, length=48, steps=60, seed=3):
    """S23 lifted to idea-space: two replicas differ by one token at g=0, driven
    by the IDENTICAL Gumbel field. Track token Hamming AND idea-space distance
    over time. Coalescence (both -> ~0) is the echo-state property in idea-space."""
    vocab = int(getattr(model.config, "vocab_size", None) or model.config.text_config.vocab_size)
    g = _gen(device, seed)
    a = _random_init(auto, length, vocab, device, seed)
    b = a.clone()
    b[seed % length] = int((int(b[seed % length]) + 1) % vocab)
    A, B = [a], [b]
    for _ in range(steps):
        gumbel = gumbel_like(torch.zeros(length, vocab, dtype=torch.float32, device=device), generator=g)
        a, _ = auto.step(a, temp, False, noise=StepNoise(gumbel=gumbel))
        b, _ = auto.step(b, temp, False, noise=StepNoise(gumbel=gumbel))
        A.append(a); B.append(b)
    hamming = np.array([float((x != y).sum().item()) for x, y in zip(A, B)])
    ta = np.stack([project(read_ideas(model, bos, s, layer, device), mean, comps) for s in A])
    tb = np.stack([project(read_ideas(model, bos, s, layer, device), mean, comps) for s in B])
    return hamming, idea_damage(ta, tb)


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
    """A model's own idea-trajectory. `model` drives BOTH the dynamics and the
    read, so the random-weight true-negative exercises the TWIN's own dynamics
    (not the trained model's) — the fix for the confounded earlier control."""
    vocab = int(getattr(model.config, "vocab_size", None) or model.config.text_config.vocab_size)
    g = _gen(device, seed)
    init = _random_init(auto, length, vocab, device, seed)
    orig = auto.model
    try:
        auto.model = model
        states, _ = auto.trajectory(init, steps, temp, absorbing, generator=g)
    finally:
        auto.model = orig
    traj = [project(read_ideas(model, bos, states[t], layer, device), mean, comps)
            for t in range(states.shape[0])]
    return np.stack(traj)


@torch.no_grad()
def logit_lens_tokens(model, tok, state, layer, device, topn=10):
    """"See" the ideas: push the layer-`layer` hidden state through the model's
    final norm + unembedding (the logit lens) and return the most common top-1
    tokens across sites. Approximate for a mid layer, but it turns the idea-space
    from numbers into words — what the representation is 'about to say'."""
    # locate final norm + unembedding generically (GPTNeoX / GPT2 / Llama-ish)
    base = getattr(model, "gpt_neox", None) or getattr(model, "transformer", None) \
        or getattr(model, "model", None)
    final_norm = (getattr(base, "final_layer_norm", None) or getattr(base, "ln_f", None)
                  or getattr(base, "norm", None)) if base is not None else None
    unembed = (getattr(model, "embed_out", None) or getattr(model, "lm_head", None))
    if final_norm is None or unembed is None:
        return None
    L = state.shape[0]
    inp = torch.empty(1, L + 1, dtype=torch.long, device=device)
    inp[0, 0] = model.config.bos_token_id or 0
    inp[0, 1:] = state
    hs = model(inp, output_hidden_states=True).hidden_states[layer][0, 1:]  # (L,d)
    logits = unembed(final_norm(hs))                                        # (L,V)
    top = logits.argmax(dim=-1).tolist()
    from collections import Counter
    common = Counter(tok.decode([t]) for t in top).most_common(topn)
    return common


def token_freeze(states) -> float:
    """Token-space analogue of freeze_score: mean fraction of sites UNCHANGED per
    step. The control that matters — if idea-freeze ~= token-freeze, idea-space
    freezing is merely inherited from the token dynamics, not a new property."""
    x = states.detach().cpu().numpy()
    return float((x[1:] == x[:-1]).mean())


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
    vocab = int(getattr(model.config, "vocab_size", None) or model.config.text_config.vocab_size)
    nL = n_layers(model)

    # --- token trajectory (the same object the token-lattice study iterates) ---
    g = _gen(device, seed=2)
    init = _random_init(auto, args.length, vocab, device, seed=2)
    traj_states, _ = auto.trajectory(init, args.steps, args.temp, False, generator=g)

    # --- LEARNED-GAP LAYER SWEEP: freeze(trained) - freeze(random twin), per layer.
    #     The headline metric. Absolute freeze is confounded by trivial contraction
    #     (see the random-weight control); the gap is the learned part.
    bases = build_basis_all(model, bos, auto, device, k=args.k, length=args.length, vocab=vocab)
    fr_trained = freeze_by_layer(model, bos, auto, bases, traj_states, device)

    if args.twin:
        twin = _random_weight_twin(info["model"], device)
        tbases = build_basis_all(twin, bos, auto, device, k=args.k, length=args.length, vocab=vocab)
        tinit = _random_init(auto, args.length, vocab, device, seed=2)
        ttraj, _ = _twin_trajectory(twin, auto, tinit, args.steps, args.temp, device, seed=2)
        fr_random = freeze_by_layer(twin, bos, auto, tbases, ttraj, device)
    else:
        fr_random = np.full(len(bases), np.nan)  # skipped for speed; see --twin

    tok_fr = token_freeze(traj_states)   # one scalar: how frozen the WORDS are
    excess = fr_trained - tok_fr         # idea settles MORE than words? (the honest signal)
    print(f"\nMEASURE  model={info['model']}  T={args.temp}  steps={args.steps}  k={args.k}")
    print(f"  token-space freeze (baseline, all layers share it): {tok_fr:.3f}")
    print(f"  layer sweep  [excess = idea_freeze - token_freeze; POOLED read]:")
    print(f"  {'layer':>5s} {'depth':>6s} {'trained':>8s} {'random':>7s} {'excess':>7s}")
    for li in range(1, len(bases)):  # skip 0 (embeddings)
        mark = "  <- 1/3" if li == round(nL/3) else ("  <- 2/3" if li == round(2*nL/3) else "")
        rnd = "  nan" if np.isnan(fr_random[li]) else f"{fr_random[li]:7.3f}"
        print(f"  {li:5d} {li/nL:6.2f} {fr_trained[li]:8.3f} {rnd} {excess[li]:7.3f}{mark}")
    best = int(1 + np.argmax(excess[1:]))
    print(f"  peak excess-over-tokens at layer {best}/{nL} (depth {best/nL:.2f}): {excess[best]:+.3f}")

    # --- POOLING CONTROL + damage + decode at the chosen layer -----------------
    layer = args.layer if args.layer is not None else best
    mean, comps = bases[layer]
    lyr_hidden = [read_ideas(model, bos, s, layer, device) for s in traj_states]  # per-gen (L,d)
    traj = np.stack([project(h, mean, comps) for h in lyr_hidden])                # pooled
    pooled_fr = freeze_score(traj)
    persite_fr = persite_freeze(lyr_hidden, mean, comps)                          # THE control
    ham, idmg = idea_damage_run(model, bos, auto, layer, mean, comps, device,
                                temp=args.temp, length=args.length, steps=args.steps)
    heal_tok = "coalesces" if ham[-1] < 1 else "persists"
    heal_idea = "coalesces" if idmg[-1] < 0.05 else "persists"
    print(f"\n  idea-space @ layer {layer}/{nL} (depth {layer/nL:.2f}):")
    real_excess = persite_fr - tok_fr        # the meaningful quantity: un-pooled vs words
    inflation = pooled_fr - persite_fr        # how much pooling exaggerated it
    print(f"    POOLING CONTROL  pooled={pooled_fr:.3f}  per-site={persite_fr:.3f}"
          f"  token={tok_fr:.3f}")
    print(f"      pooling inflation (pooled - per-site) = {inflation:+.3f}")
    print(f"      REAL dissociation (per-site - token)  = {real_excess:+.3f}")
    if real_excess > 0.2:
        print(f"      -> per-site idea freeze >> token freeze: the representation is "
              f"stabler than the words. Dissociation SURVIVES pooling (smaller than pooled).")
    else:
        print(f"      -> per-site idea freeze ~ token freeze: no real dissociation; pooled "
              f"result was an averaging artifact.")
    print(f"    trajectory: period={period2_score(traj):.3f} wander={wander_score(traj):.3f} "
          f"-> {classify(traj, THRESHOLDS)}")
    print(f"    damage token-Hamming: peak={ham.max():.0f}/{args.length} final={ham[-1]:.0f} -> {heal_tok}")
    print(f"    damage idea-space:    peak={idmg.max():.3f} final={idmg[-1]:.3f} -> {heal_idea}")

    # --- "see the ideas": logit-lens decode of the settled representation ------
    dec = logit_lens_tokens(model, tok, traj_states[-1], layer, device)
    if dec is not None:
        words = "  ".join(f"{w!r}x{c}" for w, c in dec)
        print(f"    idea decode (logit-lens @ layer {layer}, final gen): {words}")
    print("  (trust this ONLY if `calibrate` printed GREEN for this model.)")
    # TODO next: Option B -- activation-steering closed loop (feed the idea back,
    # not just observe it); then the complexity plane on the discretized idea
    # trajectory; multi-seed error bars before any of this is a finding.
    return 0


@torch.no_grad()
def _twin_trajectory(twin, auto, init, steps, temp, device, seed):
    """Iterate the random-weight twin under the same map (its own logits)."""
    orig = auto.model
    try:
        auto.model = twin
        g = _gen(device, seed)
        return auto.trajectory(init, steps, temp, False, generator=g)
    finally:
        auto.model = orig


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
        sp.add_argument("--twin", action="store_true",
                        help="also run the random-weight gap sweep (slow; gap is deprecated)")
    args = p.parse_args()
    fn = {"calibrate": calibrate, "measure": measure}[args.cmd]
    sys.exit(fn(args))


if __name__ == "__main__":
    main()
