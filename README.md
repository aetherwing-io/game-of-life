# LLM as a cellular automaton — iterating inference as a dynamical system

> *"What if we defined a set of rules for context inference in which the
> inference engine produces the next generation based on interpretation of the
> last, then loop?"*

This is **not** Conway's Game of Life on a grid. It treats a **language model as the
transition rule of a 1-D cellular automaton** and asks an information-theory /
dynamical-systems question:

> Fix a length-`L` token sequence, feed it through an LM, **resample every position at
> once**, feed the result back, and iterate. What kind of dynamical system do you get —
> does it freeze, explode into noise, sit at the **edge of chaos** (Wolfram Class 4,
> where Conway's Life lives), or something else? And can you *compute* with it?

The LM is treated as a high-dimensional categorical map `F : V^L → V^L`, not as a
writer. We strip away semantics and measure the *dynamics*. The full, reviewed results
are in **[`FINDINGS.md`](FINDINGS.md)** (§1–§26); next-session orientation is in
**[`HANDOFF.md`](HANDOFF.md)**.

---

## What we found

**No edge of chaos — and it doesn't depend on scale.** Under every knob tried —
temperature, a global frequency penalty, a masked/bidirectional rule — the iterated LLM
has *no* Wolfram Class-4 (edge-of-chaos) band. It either freezes to a model-specific
"ground state" or sits in structureless disorder. This holds from GPT-2 (124M) up to a
12B multimodal model (**gemma-4-12B**, §22): the *dynamical class* is invariant to scale
and architecture; only *which* attractor it settles into is set by the training corpus.

**The missing ingredient is locality.** A causal LLM's full attention mixes the entire
lattice every step, so nothing localized — no glider, no still-life that isn't a
lattice-commensurate standing wave — can persist (§10, §14, §21).

**The causal map is "consistent" — which *is* the echo-state property.** Driven by a
*shared* noise realization, two copies of the causal map started from different states
**synchronize** (§23). That is exactly the **echo-state property (ESP)**: the state
becomes a function of the input history, washing out where it started — the defining
precondition for **reservoir computing**.

**So is it a *useful* reservoir? No — a valid but barely-computing one (§25).** Driven
with a scalar input stream and measured with calibrated capacity estimators, the causal
LLM-CA is a dominant **instantaneous** nonlinear kernel with only a **weak lag-1 linear
memory** (~2% of a matched linear echo-state network), **no nonlinear computation over
time**, and — across the whole architecture × temperature plane — **no usable reservoir
window** (§26). Mechanistically it doesn't *forget* its input quickly; it **scrambles**
it across the lattice within a step or two into a form no low-degree readout can recover
(a perturbation lingers for tens of steps, but is decodable for only ~1–2). The strong
contraction that grants the echo-state property is exactly what empties the state of
usable capacity: **consistency and computational capacity are in tension.**

**A note on rigor.** "Is this thing computing / at the edge of chaos?" is easy to get
wrong with un-calibrated metrics. The diagnostics here carry true-negative and
true-positive controls, by-chance floor calibration, and a determinism gate — and the
project's own conclusions were reversed several times when those checks caught an
artifact (an α-grid that *under-counted* memory; finite-sample noise that *masqueraded*
as nonlinear computation, twice). In particular, the two scalar metrics it is tempting
to reach for — integrated autocorrelation time `τ_int` and correlation length `ξ` — were
found **not** to separate Wolfram class on the reference CAs, and are reported
descriptively only. The validated class discriminator is the excess-entropy /
entropy-rate plane (`complexity.py`, §24).

---

## The map

State `s ∈ V^L` is a fixed-length sequence of token ids. One **generation** is a single
**synchronous** forward pass: teacher-force the whole current sequence, read the
next-token distribution at every position *in parallel*, resample, feed back.

```
gen N ──[ one forward pass: logits at every position ]──▶ resample ──▶ gen N+1 ──▶ loop
```

Synchronous update (every site advanced at once from the previous generation) is what
makes this cellular-automaton-like rather than ordinary left-to-right generation.

**Neighborhood, stated honestly.** With a causal LM, position `k`'s logits depend on
positions `0..k-1` (plus a prepended BOS), so a site's "neighborhood" is its entire
*left* context — a **directed, long-range** automaton, not a local symmetric CA. That
asymmetry is a real limitation of the causal variant; a masked-LM map (bidirectional,
and *local* if you restrict the attention window) is the natural counterpoint, and the
code drops either into the same interface.

**Order vs. chaos.** Temperature `T` is the order parameter: `T → 0` (greedy) collapses
to a fixed point or short cycle (Class 1/2); large `T` is white noise (Class 3). The
original question was whether a **Class-4 band** exists in between (it does not). Two
update modes (`--absorbing`):

* **soft** — plain temperature sampling at every site; the all-dead state is not truly
  absorbing (noise can resurrect a site), so expect a smooth crossover.
* **absorbing** — *no spontaneous birth from vacuum*: a site whose entire left context
  is the dead token is forced dead. This makes all-dead a true absorbing state (a
  DP-style onset; treat as DP-*style*, not established directed-percolation universality,
  until finite-size exponents are measured).

The **dead token** is the model's own ground state — the argmax prediction from BOS
alone — not an arbitrary PAD/space choice. (For gemma-4-12B that ground state is the
`<image|>` modality token — §22.)

---

## What we measure

**Dynamics** (`metrics.py`, `complexity.py`):

| diagnostic | what it detects |
|---|---|
| **activity** `ρ(t)` | fraction of sites changed since last gen — freezing |
| **live density** | fraction of non-dead sites — the absorbing order parameter |
| **token entropy** `H(t)` | collapse (→0) vs chaos (→max) |
| **damage spreading / Lyapunov** `λ` | perturbation growth/healing under **shared** (coupled) noise — the key one |
| **excess entropy** `E` + **entropy rate** `h_μ` | the *validated* Class-4 discriminator (`complexity.py`, §24); the (`h_μ`,`E`) plane separates Wolfram class where `τ_int`/`ξ` fail |
| **local transfer entropy** | spatially-resolved information transport — a glider filter (lights up rule-110's gliders; §24) |
| `τ_int`, `ξ` | reported **descriptively only** — found non-diagnostic for Wolfram class (§1, §24) |

**Damage spreading** is the central dynamics probe. Run two replicas that differ in a
single token but are driven by the *identical* noise realization (coupled noise via the
Gumbel-max trick), and track their Hamming distance: bounded → ordered; exponential
growth (`λ>0`) → chaotic; marginal/power-law → edge of chaos. Using *shared* noise is
essential — otherwise you measure the temperature noise, not the system's sensitivity.
The causal map's replicas **coalesce** under shared noise — the consistency / ESP result.

**Computation** (`reservoir.py`, `capacity.py`, §25): with the ESP established, drive the
reservoir with a scalar input stream clamped at the left edge and measure

* **Memory Capacity** (Jaeger) — how well a linear readout reconstructs `input(t−k)` at
  each lag `k`;
* **Information Processing Capacity** (Dambre) — capacity over an orthonormal basis of
  the input history, split by polynomial **degree** and by **instantaneous** (delay 0)
  vs **temporal** (delay ≥ 1).

Both are calibrated against synthetic ground truth (a shift register → `MC = N`, a
planted nonlinear reservoir → exact recovery), use a wide ridge-`α` grid with an
interiority assert, and threshold against a **degree-stratified** by-chance floor.

---

## Scientific baseline: known-class CAs

Before claiming anything about an LLM, the same metrics and the same space-time renderer
run on **elementary cellular automata of known Wolfram class** (rule 110 = Class 4,
rule 30 = Class 3, rule 250 = Class 2, …). What validates the pipeline is the **visual
space-time diagram** plus the validated instruments — *not* the scalar `τ_int`/`ξ`
tables. Under a fair protocol (matched random init, post-burn), rule-110 (Class 4) reads
`τ_int = 0.47`, *lower* than chaotic rule-30's `1.39`, and `ξ = 1` for 110/30/90 alike:
those scalars are spatially/configurationally blind and do not see rule 110's gliders
(§1). What the rule-110 space-time diagram *does* show — and no LLM variant reproduces —
is **localized travelling structures on a structured background**; that picture, plus
damage spreading, the locality contrast, and the validated `(h_μ, E)` plane, carries the
"no edge of chaos" negative.

---

## Install

```bash
pip install -r requirements.txt   # torch ships with MPS support on Apple Silicon
```

The `reference` baseline needs only `numpy`/`matplotlib`/`pillow`; the LLM runs also need
`torch`/`transformers` (and `mlx-lm` for the 2-bit MLX path). On Apple Silicon `--device`
auto-selects MPS. Use **base** models (`gpt2`, `EleutherAI/pythia-*`, a Llama/Qwen
*base*) — not instruct/chat models, whose RLHF attractors dominate every orbit.

## Usage

**CA dynamics** (`python -m llm_life.run`):

```bash
# Known-class CA baselines (no model needed) — validates the visual renderer
python -m llm_life.run --out results reference --rules 110 30 90 250 --animate

# One LLM trajectory → space-time diagram (+ optional GIF). Auto-selects MPS.
python -m llm_life.run --model gpt2 --out results \
    single --temp 0.9 --length 128 --steps 300 --absorbing --animate

# Temperature sweep → phase diagram with error bars over seeds
python -m llm_life.run --model gpt2 --out results \
    sweep --temps 0.2:1.6:0.1 --seeds 5 --steps 300 --absorbing

# Coupled-noise damage spreading (consistency / Lyapunov) at fixed T
python -m llm_life.run --model gpt2 --out results \
    damage --temp 0.9 --pairs 16 --steps 200 --absorbing

# Local (windowed masked) rule — the closest thing to gliders; smaller window +
# lower T = more structure, refractory makes it travel
python -m llm_life.run --arch local --window 2 single \
    --temp 0.0 --length 96 --steps 120 --absorbing \
    --seed-mode single --refractory 3 --dump-tokens --animate

# MLX-backed native-quantized models (Apple Silicon, via mlx_lm)
python -m llm_life.run --arch mlx --model prism-ml/Ternary-Bonsai-1.7B-mlx-2bit \
    single --temp 0.0 --length 96 --steps 120 --seed-mode single --dump-tokens
```

**Reservoir computing & complexity diagnostics** (root-level `scripts_*.py`, §24–§26):

```bash
# The validated complexity-entropy plane + the local-TE glider filter (§24)
python scripts_complexity_plane.py
python scripts_glider_filter.py

# Reservoir apparatus / ESP / Memory Capacity / IPC (§25)
python scripts_reservoir.py esp           # echo-state convergence at the drive cell
python scripts_capacity_lock.py           # locked 3-seed MC/IPC run + α-interiority
python scripts_mc_gate.py --steps 7000    # gated MC at proper samples-per-feature (E3 bar)
python scripts_degree_floor.py            # degree-stratified floor (the nonlinear-temporal gate)

# Consistency–capacity tradeoff across architecture × temperature (§26)
python scripts_tradeoff.py
```

Set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` to use cached weights without network.

---

## Models exercised

GPT-2; `EleutherAI/pythia-*` and `Qwen3-Base` (full-precision causal controls);
`distilroberta`/`bert`/CodeBERTa (masked + windowed-local); a trained-ternary model
(Bonsai) and a native-2-bit MLX checkpoint; and **gemma-4-12B**, an any-to-any
multimodal 12B base model (§22). The reservoir study (§25/§26) is headlined on
`pythia-160m` with `distilroberta` for the cross-architecture comparison — small models,
which is the main caveat on the exact magnitudes (the dynamical *class* result is
scale-robust; the precise reservoir numbers are not pinned, and the headline `MC` is a
measured *lower bound* — §25).

## Layout

```
llm_life/
  model.py         model loading (torch + MLX/mlx_lm) + device selection + dead-token detection
  automaton.py     the synchronous LLM-CA map (causal / masked / local / mlx; soft + absorbing)
  sampler.py       Gumbel-max sampling with coupled (shared) noise
  metrics.py       activity, entropy, autocorr time, corr length, Lyapunov / damage spreading
  complexity.py    excess entropy + entropy rate + local transfer entropy — the
                   VALIDATED Class-4 diagnostics (§24); τ_int/ξ are not
  reservoir.py     driven-reservoir apparatus: codebook drive, shared-noise, PCA readout
                   (§25). NB: its own MC/IPC are SUPERSEDED — see capacity.py
  capacity.py      Memory Capacity (Jaeger) + Information Processing Capacity (Dambre):
                   wide-α, encoded-symbol basis, degree-stratified floor (§25)
  reference_ca.py  elementary CA baselines (known Wolfram classes)
  viz.py           embedding-PCA space-time diagrams + animated GIFs
  run.py           CLI: single / reference / sweep / damage

scripts_*.py       complexity plane/scaling, glider filter, reservoir/capacity/ESP,
                   degree-stratified floor, capacity-vs-lag, cross-architecture, tradeoff
FINDINGS.md        full reviewed results, §1–§26
HANDOFF.md         orientation + prioritized next experiments
results/           figures, space-time GIFs, and claim-backing CSVs
```

---

## Reading guide (FINDINGS.md)

- **§1, §24** — why `τ_int`/`ξ` are non-diagnostic, and the validated `(h_μ, E)` plane
  and local-transfer-entropy glider filter that replace them.
- **§4, §8, §10** — damage spreading, causal synchronization, and the locality argument.
- **§14, §21** — the "no free glider": localized structures are lattice-commensurate
  standing waves, not travelers.
- **§16–§22** — the dynamical class is invariant across architecture/scale; only the
  attractor genre shifts with training (culminating in gemma-4-12B at 12B).
- **§23** — consistency / the common-noise damage-fate map = the echo-state property.
- **§25** — reservoir computing: weak linear memory, instantaneous-dominant, no nonlinear
  temporal, scrambles-not-stores.
- **§26** — consistency–capacity tradeoff: no edge-of-stability window anywhere in the
  (arch × T) plane.

## Status & honest caveats

This is curiosity-driven dynamical-systems characterization, not a product. The headline
*class* result (no edge of chaos, scale-invariant) is robust and visually carried; the
*reservoir* result is a clean, calibrated **negative** with a mechanism, but on small
models and over a niche construction almost nobody uses in practice. The exact reservoir
magnitudes are not pinned (the memory `MC` is still rising at 20× samples-per-feature, so
the reported value is a lower bound — the verdict is invariant to it). The most
informative open experiment is the reservoir measurement on a larger model with **local
(sliding-window) attention**, since locality is the thesis's missing ingredient.
