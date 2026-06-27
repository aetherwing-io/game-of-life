# HANDOFF — LLM-as-cellular-automaton

Next-session guidance. Read [`FINDINGS.md`](FINDINGS.md) for the full results
(§1–22) and [`README.md`](README.md) for the harness. This file is the orientation
+ what to do next.

## Bottom line (what we actually know, post-review)

Iterating LLM inference as a synchronous 1-D cellular automaton:

- **No edge of chaos / no Class-4 band** for full-attention LLMs, now confirmed
  from GPT-2 (124M) up to **gemma-4-12B** (`ξ=1`, only Class-1/2 attractors). The
  negative is carried by the **visual** space-time picture (rule-110 has gliders,
  no LLM variant does) + **damage spreading** + the **§12 locality contrast** —
  **not** by the τ_int/ξ tables (those were found non-diagnostic; see Calibration).
- **The dynamics are invariant to architecture, scale, and numeric precision; the
  *attractor* is set by training.** Recurrent (RWKV/Mamba), ternary (Bonsai),
  native-2-bit (MLX), full-precision (Qwen3/pythia), bidirectional (distilroberta),
  and multimodal-12B (gemma-4) all show the same repertoire (vacuum collapse /
  fill-to-still-life / common-noise sync). Only the genre of the attractor changes
  — a **corpus/tokenizer-family** property modulated by post-training (§16–22).
- **The one genuinely-novel, pursuable result:** the **architecture-dependent
  conditional-Lyapunov sign split** (§8) — causal full-attention is *universally
  synchronizing* (heals all damage under shared noise across T∈0.4–1.4 and 160M→1.7B
  scale), bidirectional *never* synchronizes. Bill it as: *"temperature-sampling
  'chaos' in iterated LLM inference is the consistency/echo-state property, with an
  architecture-dependent breakdown."* This is the thing worth turning into a paper.
- **No free life-form / no glider** (§14, §21). Localized structures exist but are
  lattice-commensurate standing waves; the LLM's smooth local conditional lacks
  Conway's knife-edge B3/S23 balance. Seeds select *which* resonance, not *whether*
  one is free.

## Calibration — read before quoting any number

The three-reviewer panel (§20) found and we fixed real overclaims:

- **τ_int and ξ do NOT discriminate Wolfram class** (matched-init: rule-110
  τ_int=0.47 < rule-30's 1.39; ξ=1 for 110/30/90). They are descriptive, not
  edge-detectors. `cmd_reference` and `cmd_single` now compute them **post-burn**;
  every single-run τ_int/ξ in §16–22 is a *transient* (the runs freeze →
  steady-state τ_int≈0). Don't lead with them.
- **§4/§16 synchronization is a known phenomenon** (echo-state consistency;
  arXiv 2503.00831, 1901.07729) — novel only as a *diagnostic* + the §8 sign split.
- **§18 "2-bit ≡ fp16" is a backend tautology** (same trained weights, two formats).
  The real quantization claim is §19's *checkpoint-not-bit-width*.
- **Genre attractor is corpus-family, not checkpoint-unique** (pythia≈RWKV share the
  Pile attractor). State the 3-level hierarchy: degeneration (universal) →
  self-predictivity (universal) → which basin (training).
- **Scope:** mostly ≤1.7B + the single gemma-4-12B point. Not "LLMs" in general.

## Environment (current, this machine)

- `transformers==5.12.1` (upgraded from 5.5.4 this session for `gemma4_unified`;
  the existing causal/masked/local/mlx paths were smoke-tested OK after the bump).
- `torch==2.9.1` (MPS), `mlx_lm==0.31.2`, numpy/matplotlib/pillow.
- **Always run with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`** for cached models —
  online mode stalls badly when a big download is in flight.
- HF auth: a token is in place (user `ostk-ai`).

### Cached models WITH weights (verified usable this session)
`RWKV/rwkv-4-169m-pile`, `state-spaces/mamba-130m-hf`, `EleutherAI/pythia-160m`,
`Qwen/Qwen3-1.7B-Base`, `Qwen/Qwen2.5-0.5B`(+Instruct), `distilroberta-base`,
`bert-base-uncased`, `huggingface/CodeBERTa-small-v1`,
`prism-ml/Ternary-Bonsai-1.7B-unpacked`, `prism-ml/Ternary-Bonsai-1.7B-mlx-2bit`,
`prism-ml/Ternary-Bonsai-8B-mlx-2bit`, `google/gemma-4-12B`.
⚠️ Several other snapshots (`pythia-410m/1.4b`, `rwkv-4-430m`, `Qwen2.5-1.5B`) were
cached config-only in a prior session — **verify weights exist before use** (a
missing-weights load throws `does not appear to have ... model.safetensors`).

### Gotchas
- **gemma / Gemma:** loads via the normal `--arch causal` path (text head
  `Gemma4UnifiedForCausalLM`). `model.py` already uses **bf16** for gemma (fp16
  overflows to NaN), reads vocab from `text_config`, and `viz.py` subsamples the
  SVD for the 262144-row embedding. 12B is ~24 GB resident on MPS and slow
  (~seconds/forward) — keep L/steps small (L≤64, steps≤80).
- **gitignore:** `results/single_*.csv` and `results/damage_*.csv` are ignored
  (bulk). PNG/GIF/`tokens_*.txt`/`sweep_raw_*`/`sweep2d_raw_*` are kept. When a CSV
  *backs a claim*, `git add -f` it (as done for `lam_sweep_T.csv`, `seed_hunt.csv`,
  the matched-L damage CSVs).
- **Big HF downloads stall** with the default Python downloader. Use
  `HF_XET_HIGH_PERFORMANCE=1 hf download <repo>` (no multi-pattern `--include` — the
  CLI mis-parses it as filenames). It resumes into the same cache.
- **MLX path:** `--arch mlx`; native 2-bit ≡ fp16-unpacked for the same checkpoint.

## What this session added (§16–22, 10 commits)
SSM/recurrent (§16), ternary Bonsai (§17), MLX 2-bit backend + faithfulness (§18),
matched controls (§19), the panel audit + remediation (§20), the §8 sign-split
matched-L reproduction + T×scale sweep, seed-engineering life-form hunt (§21),
gemma-4-12B (§22).

## Next experiments — prioritized

1. **Directed-percolation finite-size scaling** of (a) the §8 causal→masked
   synchronization transition and (b) the absorbing `T_c≈1.3`. Vary L∈{64…512};
   extract β, ν⊥, ν∥; attempt a 1+1-D DP data collapse (β≈0.276, ν⊥≈1.097,
   ν∥≈1.734). This is what turns "DP-style crossover" and the sign split into a
   *measured* universality claim — the highest-value path to a real result.
   (`scripts_lam_sweep.py` is the starting point; add per-replica Hamming CSVs.)
2. **The non-tautological quantization test:** 2-bit-quantize a *non-ternary* model
   (e.g. `Qwen/Qwen3-1.7B-Base`) via the MLX path and compare its dynamics to its
   own fp32 self — same checkpoint, different bits. This is the real §18.
3. **±w windowed-ring interpolation** between causal-sync and masked-chaos: sweep
   the local window `w` and see where the λ_cond sign flips. Also measure the damage
   light-cone velocity v≈w (a Lieb-Robinson bound) in `--arch local`.
4. **Glider hunt, take 3:** focused refractory×window sweep on a *sparse-genre*
   (bert-like) masked model at the `w` where structure is marginal. §21 shows seeds
   nucleate bounded oscillators but they're L-pinned; the missing piece is the
   rule's birth/death balance, not the seed — so the realistic target is an
   L-robust still-life, not a spaceship.
5. **Second masked model + larger scale** for the §8 sign split (currently one
   masked model, ≤1.7B causal) — does any causal model ever reach λ_cond>0?

## Replace τ_int/ξ as the spatial/temporal diagnostics
Both are non-diagnostic here. For a real edge-of-chaos detector, add: a
**configurational two-point token correlation** function (not the change-indicator
field), and the **mutual-information peak at Langton's λ_c** (arXiv cond-mat/9409080)
— the canonical Class-4 discriminator, never computed in this repo.
