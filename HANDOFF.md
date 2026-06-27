# HANDOFF — LLM-as-cellular-automaton

Next-session guidance. Read [`FINDINGS.md`](FINDINGS.md) for the full results
(§1–23) and [`README.md`](README.md) for the harness. This file is the orientation
+ what to do next.

## Bottom line (what we actually know, post-review)

Iterating LLM inference as a synchronous 1-D cellular automaton:

- **No edge of chaos / no Class-4 band** for full-attention LLMs, now confirmed
  from GPT-2 (124M) up to **gemma-4-12B** by the observed attractors and
  space-time diagrams. The negative is carried by the **visual** space-time picture
  (rule-110 has gliders, no LLM variant does) + **damage spreading** + the **§12
  locality contrast** — **not** by the τ_int/ξ tables (those were found
  non-diagnostic; see Calibration).
- **The coarse repertoire is robust across the tested architectures/scales; the
  *attractor* is set by training.** Recurrent (RWKV/Mamba), trained-ternary
  (Bonsai), full-precision (Qwen3/pythia), bidirectional (distilroberta), and
  multimodal-12B (gemma-4) all show variants of vacuum collapse /
  fill-to-still-life / common-noise sync. The native-2-bit MLX comparison is only
  same-checkpoint backend/packing validation, not a broad precision theorem. Only
  the genre of the attractor changes — a **corpus/tokenizer-family** property
  modulated by post-training (§16–22).
- **The most useful live object is now the common-noise damage-fate map (§23).**
  Treat the old causal-vs-masked "conditional Lyapunov sign split" as one slice
  through a broader finite-horizon coalescence problem, not as an asymptotic theorem.
  At `L=48`, tested causal maps coalesce under shared Gumbel noise; global masked
  `distilroberta-base` does not; `bert-base` and CodeBERTa are mixed. In the local
  `distilroberta` rule the window is the control parameter: `w=1` coalesces, `w=2`
  is a real boundary regime, and `w>=3` usually preserves damage.
- **No free life-form / no glider** (§14, §21). Localized structures exist but are
  lattice-commensurate standing waves; the LLM's smooth local conditional lacks
  Conway's knife-edge B3/S23 balance. The §23 structure probe reinforced this:
  a web-boilerplate seed fills the ring and `w=4` gives a period-2 standing
  oscillator with zero velocity, not a traveler.
- **The negative is now on a validated instrument, not just visuals (§24).** A
  complexity-entropy plane (excess entropy `E` vs entropy rate `h_μ`), validated
  to separate Wolfram class on the reference CAs (where τ_int/ξ failed), places
  every LLM regime **outside the Class-4 corner**. The sparse engineered
  "lifeforms" land in the **Class-2** region (standing order); causal is
  chaos/structureless; masked random-init is a saturated fill. Local transfer
  entropy is the new spatially-resolved glider filter.

## Calibration — read before quoting any number

The three-reviewer panel (§20) found and we fixed real overclaims:

- **τ_int and ξ do NOT discriminate Wolfram class** (matched-init: rule-110
  τ_int=0.47 < rule-30's 1.39; ξ=1 for 110/30/90). They are descriptive, not
  edge-detectors. `cmd_reference` and `cmd_single` now compute them **post-burn**;
  every single-run τ_int/ξ in §16–22 is a *transient* (the runs freeze →
  steady-state τ_int≈0). Don't lead with them.
- **§4/§16 synchronization is a known phenomenon** (echo-state consistency;
  arXiv 2503.00831, 1901.07729) — novel only as a *diagnostic* + the §8 sign split.
- **Use finite-horizon language for §23.** `p_sync` means sustained `H=0` over the
  final 10 generations; `late_damage` is the mean final-window Hamming damage. Do
  not call these measured asymptotic Lyapunov exponents without longer L/step
  scaling.
- **§18 "2-bit ≡ fp16" is a backend tautology** (same trained weights, two
  formats). Treat it as same-checkpoint MLX validation; the non-tautological
  quantization test is still a non-ternary model quantized and compared to its
  own full-precision checkpoint.
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
- **MLX path:** `--arch mlx`; native 2-bit matched fp16-unpacked for the same
  checkpoint's fixed point, validating the backend/packing only.
- **MPS + sandbox:** inside the managed sandbox, `torch.backends.mps.is_built()`
  was true but `is_available()` was false, so runs silently fell to CPU. With
  approved unsandboxed execution, MPS was available and the same tensor smoke test
  worked. For real experiments, request approval and run with
  `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ... --device mps`.

## What this session added (§23)

- **Raw common-noise consistency sweeps:** `scripts_consistency_sweep.py` wrote
  `results/consistency_*_{sweep,pairs,hamming}.csv` plus
  `results/endpoint_summary_all_current.csv`. Core slice at `L=48`, `T=1`,
  16 pairs: pythia causal `p_sync=1.00`; global masked distilroberta `p_sync=0`;
  bert-base `p_sync=0.44`; local distilroberta `w=1/2/4` gives
  `p_sync=1.00/0.44/0.12`.
- **Null controls:** `scripts_null_probe.py` wrote
  `results/null_probe_{aggregate,pairs,hamming}.csv`. IID uniform, Bernoulli, and
  simple absorbing-local Bernoulli nulls all synchronize trivially, so persistent
  damage in trained masked/local rules is not a shared-noise tautology.
- **Fate maps:** `scripts_fate_map.py` wrote the window×temperature maps and PNGs:
  `results/fate_map_distilroberta_L48_w1-8_T06-14_p8_*` and the refined boundary
  scan `results/fate_map_distilroberta_L48_w2_T05-15_p16_*`.
- **Structure + animation:** `scripts_structure_probe.py` found no traveler for
  `" Comments Related Posts"`; `scripts_pair_damage_gif.py` produced coupled
  damage GIFs (`results/pairdamage_*.gif`) showing causal/local healing,
  masked saturation, and mixed `w=2/w=4` outcomes. The contact sheet is
  `results/pairdamage_contact.png`.

Recent prior additions were §16–22: SSM/recurrent, ternary Bonsai, MLX backend
validation, matched controls, panel audit/remediation, seed-engineering life-form
hunt, and gemma-4-12B.

## Next experiments — prioritized

1. **Scale the §23 fate map before theorizing.** Repeat the `w×T` and refined
   `w=2` scans at `L=96/192`, longer horizons, and 32+ pairs. The key questions are
   whether the `w=2→3` wall sharpens with size and whether high-temperature
   coalescence at `w=2` survives.
2. **Build stronger nulls.** Add logit-shuffle, random-initialized MLM/causal, and
   temperature-matched marginal nulls. The current state-independent nulls only
   prove that shared noise alone does not explain trained-model non-coalescence.
3. **Measure fields, not just endpoints.** Save raw states or damage bitsets for
   selected fate-map cells; estimate front velocity, damage avalanche statistics,
   configurational token correlations, and mutual information. Generate pair-damage
   GIFs for representative cells, especially mixed `w=2` outcomes.
4. **Second masked model + larger causal/entropy-matched causal runs.** Keep the
   old §8 question alive, but phrase it as: where in model/temperature/window space
   does common-noise coalescence fail?
5. **The non-tautological quantization test:** 2-bit-quantize a *non-ternary* model
   (e.g. `Qwen/Qwen3-1.7B-Base`) via the MLX path and compare its dynamics to its
   own fp32 self — same checkpoint, different bits. This is the real §18.
6. **Glider hunt stays secondary.** Focus it on sparse BERT-like regimes and
   L-robust still-lifes first. The current map says the interesting boundary is
   damage coalescence/non-coalescence; spaceships are not the nearest target.

## Replace τ_int/ξ as the spatial/temporal diagnostics — DONE (§24)
This is now built and validated: `llm_life/complexity.py` adds **excess entropy
`E`** (Crutchfield–Feldman), **entropy rate `h_μ`**, the **configurational
two-point token MI** (not the change-indicator field), and **local transfer
entropy** (Lizier's glider filter). On the known-class reference CAs the (`h_μ`,
`E`) plane separates the classes where τ_int/ξ could not — rule-110 `E=1.36` vs
rule-30 `0.004` (τ_int had 110 *below* 30) — and local-TE traces rule-110's
gliders as coherent filaments (+0.62 bits) vs rule-30's speckle (+0.00). Every
LLM regime placed on the plane (`scripts_complexity_plane.py`,
`results/complexity_plane.png`) lands **outside** the Class-4 corner: causal →
chaos/structureless, masked random-init → saturated fill (origin), sparse bert
"lifeforms" → the **Class-2** region (rule-184 neighbourhood), confirming §14/§21.
Remaining: a finite-size `E(L)` scaling, and the **MI-peak-at-Langton-λ_c** sweep
(vary the rule's activity λ and look for the E peak; cond-mat/9409080) — the one
piece of the canonical picture still not swept.

## Final thought for next

The interesting object is the system's behavior map, not a binary "LLM life?"
answer. Chase the mixed regimes: `w=2`, BERT/CodeBERTa, high-vs-low temperature,
and cells where one pair heals after a large bloom while another saturates. Those
are where the rule is exposing structure rather than just collapsing or filling.
