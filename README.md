# LLM as a cellular automaton — iterating inference as a dynamical system

> *"What if we defined a set of rules for context inference in which the
> inference engine produces the next generation based on interpretation of the
> last, then loop?"*

This is not Conway's Game of Life on a grid. It treats a **language model as the
transition rule of a cellular automaton** and asks an information-theory /
dynamical-systems question:

> If you fix a length-`L` token sequence, feed it through an LM, resample every
> position, feed the result back, and iterate — what kind of dynamical system do
> you get? Does it freeze, explode into noise, or sit at the **edge of chaos**
> (Wolfram Class 4, where Conway's Life lives)?

The LM is treated as a high-dimensional categorical map `F : V^L → V^L`, not as a
writer. We strip away semantics and measure the *dynamics*.

> **Results: see [`FINDINGS.md`](FINDINGS.md).** Short version — GPT-2 iterated
> this way has *no* edge-of-chaos regime under any knob tried: temperature
> (disorder), a global frequency penalty (balance), or a masked/bidirectional
> rule (symmetry). It collapses to a dead state or sits in disorder; the causal
> variant's "chaos" even *synchronizes* under common noise. The missing
> ingredient is **locality** — full attention mixes the whole lattice every
> step, so nothing localized can persist. The masked variant came closest
> (drifting clusters near its transition). Local-neighborhood rules are the
> next experiment.

## The map

State `s ∈ V^L` is a fixed-length sequence of token ids. One generation is a
single **synchronous** forward pass: teacher-force the whole current sequence,
read the next-token distribution at every position *in parallel*, resample, feed
back. Synchronous update (every site advanced from the previous generation at
once) is what makes this cellular-automaton-like rather than ordinary
left-to-right generation.

```
gen N  ──[ one forward pass: logits at every position ]──▶  resample ──▶  gen N+1 ──▶ loop
```

**Neighborhood, stated honestly.** With a causal LM, position `k`'s logits depend
on positions `0..k-1` (plus a prepended BOS). So a site's neighborhood is its
entire *left* context — this is a **directed, long-range** automaton, not a local
symmetric CA. That asymmetry is a real limitation of the causal variant; a
masked-LM map (bidirectional, local-ish) is the natural follow-up and the code is
structured so it can drop into the same interface.

## Controlling order vs. chaos

Temperature `T` is the order parameter. `T → 0` (greedy) collapses to a fixed
point or short cycle within a few steps — the well-known greedy-decoding
degeneracy (Class 1/2). Large `T` is white noise (Class 3). The question is
whether a **Class-4 band** exists in between.

Two update modes (`--absorbing` flag):

* **soft** — plain temperature sampling at every site. The all-dead state is not
  truly absorbing (noise can always resurrect a site), so expect a smooth
  *crossover*, not a sharp transition.
* **absorbing** — *no spontaneous birth from vacuum*: a site whose entire left
  context is the dead token is forced dead; every other site samples at `T`.
  This makes all-dead a true absorbing state, giving a genuine
  directed-percolation-style **phase transition** with a real critical point.

The **dead token** is the model's own ground state — the argmax prediction from
BOS alone — not an arbitrary PAD/space choice.

## What we measure

| diagnostic | what it detects |
|---|---|
| **activity** `ρ(t)` | fraction of sites that changed since last gen — freezing |
| **live density** | fraction of non-dead sites — the absorbing order parameter |
| **token entropy** `H(t)` | collapse (→0) vs. chaos (→max) |
| **integrated autocorr time** `τ_int` | **critical slowing down** — peaks at the edge |
| **spatial correlation length** `ξ` | spatial order — diverges at criticality |
| **Lyapunov / damage spreading** `λ` | the sharpest edge detector (see below) |

**Damage spreading** is the key one. Run two replicas that differ in a single
token but are driven by the *identical* noise realization (coupled noise via the
Gumbel-max trick), and track their Hamming distance over time:

* bounded → ordered (Class 1/2)
* exponential growth (`λ > 0`) → chaotic (Class 3)
* marginal / power-law → **edge of chaos (Class 4)**

Using shared noise is essential — otherwise you measure the temperature noise,
not the system's sensitivity to perturbation.

## Scientific baseline: known-class CA

Before claiming anything about an LLM, the same metrics and the same space-time
renderer are run on **elementary cellular automata whose Wolfram class is known**
(rule 110 = Class 4, rule 30 = Class 3, rule 250 = Class 2, …).

**What validates the pipeline is the *visual* space-time diagram, not the scalar
metrics.** An external review (see [`FINDINGS.md`](FINDINGS.md) §1 & §20) showed
that `τ_int` and `ξ`, *as implemented*, do **not** separate the Wolfram classes:
under a fair protocol (matched random init, post-burn) rule-110 (Class 4) reads
`τ_int=0.47`, *lower* than chaotic rule-30's `1.39`, and `ξ=1.0` for 110/30/90
alike. `τ_int` is an autocorrelation of *global activity* (spatially blind) and `ξ`
of the *change-indicator* field (configuration-blind), so neither sees rule 110's
gliders. The earlier "rule-110 τ_int ≈ 17–29, by far the highest" claim was an
artifact of giving rule 110 a single-cell init (a long spreading transient) while
the others got random init.

What the rule-110 space-time diagram (`results/ref_rule110.png`) *does* show — and
no LLM variant reproduces — is **localized travelling structures (gliders) on a
structured background**. That qualitative picture, plus damage-spreading and the
locality contrast, carries the "no edge of chaos" negative; read every `τ_int`/`ξ`
number below as descriptive, not as an edge-detector.

## Visualization

Space-time diagrams put **position on x, generation (time) on y, colour =
token**. To make structure visible, token ids are coloured by projecting the
model's input-embedding matrix onto its top-3 principal components → RGB, so
embedding-similar tokens get similar colours and coherent regions / travelling
fronts appear as contiguous colour structure and diagonal streaks rather than
random speckle. `--animate` additionally writes a scrolling GIF so you can watch
fronts propagate.

## Install

```bash
pip install -r requirements.txt   # torch ships with MPS support on Apple Silicon
```

The `reference` baseline needs only `numpy`/`matplotlib`/`pillow`; the LLM runs
also need `torch`/`transformers`.

## Usage

```bash
# Known-class CA baselines (no model needed) — validates the metrics + viz
python -m llm_life.run --out results reference --rules 110 30 90 250 --animate

# One LLM trajectory -> space-time diagram (+ optional GIF). Auto-selects MPS.
python -m llm_life.run --model gpt2 --out results \
    single --temp 0.9 --length 128 --steps 300 --absorbing --animate

# Temperature sweep -> phase diagram with error bars over seeds
python -m llm_life.run --model gpt2 --out results \
    sweep --temps 0.2:1.6:0.1 --seeds 5 --steps 300 --absorbing

# Coupled-noise damage spreading (Lyapunov / butterfly) at fixed T
python -m llm_life.run --model gpt2 --out results \
    damage --temp 0.9 --pairs 16 --steps 200 --absorbing
```

Use a **base** model (`gpt2`, `EleutherAI/pythia-410m`, a Llama *base*) — not an
instruct/chat model, whose RLHF attractors dominate every orbit. On an M-series
Mac, `--device` auto-selects MPS; scale to `gpt2-large` / `pythia-1.4b` if the
small model's dynamics are uninteresting.

## Running on your own machine (handoff)

`--device` auto-selects MPS on Apple Silicon. The interface is just `--model` /
`--arch` / `--window`, so swapping models is one flag. Use **base** models for
clean dynamics; the dead token and absorbing rule auto-adapt to any model.

```bash
# The locality result — a propagating, periodic CA (the closest to gliders).
# Smaller window + lower T = more structure; refractory makes it travel.
python -m llm_life.run --arch local --window 2 single \
    --temp 0.0 --length 96 --steps 120 --absorbing \
    --seed-mode single --refractory 3 --dump-tokens --animate

# See what the cells ARE: decode the token grid to a .txt
python -m llm_life.run --arch local --window 2 single \
    --temp 0.3 --length 48 --steps 30 --absorbing --seed-mode single --dump-tokens

# Control the input: does a coherent / out-of-genre seed escape the basin?
python -m llm_life.run --arch local --window 3 single \
    --temp 0.2 --length 64 --steps 80 --absorbing \
    --seed-text " def function return value" --dump-tokens   # code-like seed
python -m llm_life.run --arch local --window 3 single \
    --temp 0.2 --length 64 --steps 80 --absorbing \
    --seed-text " the quiet river at dawn" --dump-tokens      # prose seed

# MLX-backed quantized models (Apple Silicon, via mlx_lm). --arch mlx is causal;
# the forward runs on Metal and logits are bridged to torch. Lets you iterate a
# native 2-bit checkpoint as a CA. (Needs `pip install mlx-lm`.)
python -m llm_life.run --arch mlx --model prism-ml/Ternary-Bonsai-1.7B-mlx-2bit \
    single --temp 0.0 --length 96 --steps 120 --seed-mode single \
    --dump-tokens --animate
```

### Experiments worth running with more compute

- **More capable / different-genre base models.** Swap `--model` for a larger
  base model (`EleutherAI/pythia-1.4b` causal; `roberta-large`, `bert-large`, or
  a code/BERT-of-code masked model for `--arch masked/local`). Prediction: bigger
  = sharper = *deeper* attractor basins, so unfamiliar seeds snap back faster,
  but the transient structure is richer and the "lifeform" tokens shift to that
  model's training genre.
- **Unfamiliar seeds × capable models.** `--seed-text` with out-of-distribution
  content (a code model seeded with prose, or rare/foreign tokens) probes basin
  depth — how far the model can be pushed before it collapses to its default
  genre, and how long unusual structure survives under the local+refractory
  scaffold. This is the experiment the local variant was built for.
- **Window / refractory phase map.** Sweep `--window` (2–8) and `--refractory`
  (1–6) at low T; somewhere in there is the most glider-rich regime.
- **Local frequency penalty** (not yet implemented) — make the balance knob
  neighbourhood-local rather than whole-grid; pair it with `--arch local`.

## A genuinely possible outcome

GPT-2's geometry may have **no Class-4 band at all** — it might jump straight from
frozen to static as `T` rises. That is a real, reportable result, not a failure.
The harness is built to *detect whether* the edge exists (via the `τ_int`/`ξ`
peaks and the `λ` damage curve), not to assume it does. The absorbing variant is
where a true transition, if any, is most likely to appear.

## Layout

```
llm_life/
  model.py         model loading (torch + MLX/mlx_lm) + device selection + dead-token detection
  automaton.py     the synchronous LLM-CA map (causal / masked / local / mlx variants; soft + absorbing)
  sampler.py       Gumbel-max sampling with coupled (shared) noise
  metrics.py       activity, entropy, autocorr time, corr length, Lyapunov
  reference_ca.py  elementary CA baselines (known Wolfram classes)
  viz.py           embedding-PCA space-time diagrams + animated GIFs
  run.py           CLI: single / reference / sweep / damage
```
