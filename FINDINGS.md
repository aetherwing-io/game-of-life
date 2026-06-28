# Findings: GPT-2 iterated as a cellular automaton

Experimental results for the question *"if you loop LLM inference — feed a
token sequence in, take the output, feed it back — what kind of dynamical
system do you get, and does it sit at the edge of chaos?"*

**Model:** GPT-2 (124M, base) for §1–15; §16–23 add RWKV/Mamba, ternary Bonsai,
MLX 2-bit, pythia/Qwen3-Base controls, gemma-4-12B, and the common-noise fate-map
work. **Lattice:** `L=128` (`L=48–96` for the newer variants). **Seeds:** 3 in
the sweeps; **n=1 in most of §16–19**, with multi-pair damage runs in §23.
**Hardware:** CPU / Apple MPS. Raw numbers in `results/*.csv`; figures in
`results/`.

> **Scope (added after external review, §20).** Most models tested are **≤1.7B
> params**, where small models are known to degenerate more readily (e.g. arXiv
> 2509.26643 finds a minimum scale for stable token distributions) — so read the
> bulk of the headline as *"for small base models under this synchronous map."*
> The one large data point, **gemma-4-12B (§22), shows the same no-Class-4
> visual/damage behavior** (with ξ=1 only a descriptive spatial-scale readout), so
> the negative now extends to 12B; but a systematic scale sweep is still missing.
> The §16–19 single-run τ_int/ξ magnitudes are transient, not steady-state (§1,
> §20).

## TL;DR

**Full-attention LLMs iterated as cellular automata have no Class-4 /
edge-of-chaos regime — not under temperature, not under a global balance knob,
not with a bidirectional rule.** The missing ingredient is *locality*: full
attention mixes information globally every step, so no localized structure can
persist (`ξ` pinned at its floor everywhere; sections 1–10). **This negative is
now on a *validated* instrument (§24):** a complexity-entropy plane (excess
entropy `E` vs entropy rate `h_μ`) that separates Wolfram class on the reference
CAs — where `τ_int`/`ξ` provably fail — places *every* LLM regime outside the
Class-4 corner that rule 110 occupies (causal → chaos, masked → saturated fill,
the sparse engineered "lifeforms" → the Class-2 standing-order region).

**Restoring locality changes everything (section 12).** A windowed-ring rule
(each site predicted from only its ±w neighbours) gives finite signal speed and
immediately produces what full attention could not: a single live cell
**propagates at ~w sites/generation**, spatial correlation jumps to `ξ ≈ 18`
(vs 1 for full attention), and stable periodic structures (oscillators) appear.
Adding a refractory/decay rule (cells die after R generations alive — the
Brian's-Brain mechanism) gives the *trailing* death that makes structures
travel rather than fill. The "lifeforms" are made of the model's
lowest-context attractor tokens — for distilroberta, web-page boilerplate
(`Comments`, `Related`, `Next`, `»`); a fingerprint of its training data.

Under temperature control it does one of two things:

- **Absorbing rule** → collapses into the dead state (the all-`\n` ground
  state) below a sharp critical temperature `T_c ≈ 1.3`; only a feeble active
  phase survives above it.
- **Soft rule** → a single frozen point at exactly `T=0`, then for *any* `T>0`
  a high-activity, spatially-structureless regime that intensifies monotonically
  into pure noise. No critical slowing down, no localized structures, no gliders.

And a subtler result from damage spreading: much of the high-activity "chaos" is
**not deterministic chaos**. Under shared noise, causal maps often *synchronize*:
perturbations amplify briefly then heal to zero. §23 shows the fuller map: global
masked distilroberta preserves damage, causal pythia/Qwen coalesce, and the local
distilroberta rule has a sharp window-dependent boundary (`w=1` coalesces, `w=2`
is mixed, `w>=3` mostly does not).

## 1. The instruments, honestly (reference baseline) — the *visual* discriminates Wolfram class; τ_int and ξ do **not**

> *Corrected after an external review (§20). The original version of this section
> claimed τ_int separated the classes (rule-110 τ_int=28.9 = "critical slowing
> down" vs rule-30's 1.25). That was a **confound**, not a result: the reference
> table quietly gave each rule a different init (rules 110/90 a single live cell,
> 30/250 a random row; `run.py` `cmd_reference`) and computed τ_int over the
> **full** trajectory. Rule-110's 28.9 is its single-cell lattice-filling
> transient (τ decays monotonically with burn-in, never converging), not critical
> slowing down. Give rule 30 the same single-cell+full-trajectory treatment and it
> reads 26.4 ≈ 110.*

Under a **fair** protocol — matched random init for every rule, post-burn (now the
committed behavior) — the separation vanishes:

| rule | class | activity ρ | entropy (bits) | τ_int | ξ |
|---|---|---|---|---|---|
| 110 | 4 (edge of chaos) | 0.42 | 0.98 | **0.47** | 1.0 |
| 30  | 3 (chaotic) | 0.50 | 1.00 | 1.39 | 1.0 |
| 90  | 3 (Sierpinski) | 0.51 | 1.00 | 0.73 | 1.0 |
| 250 | 2 (periodic) | 0.00 | 0.00 | 0.00 | 0.0 |

The Class-4 rule (110, τ_int=0.47) is now **indistinguishable from — indeed lower
than — the chaotic rule 30** (1.39), and ξ=1.0 for 110, 30, and 90 alike.
**Neither τ_int nor ξ, as implemented, discriminates the Wolfram classes** —
τ_int is a temporal autocorrelation of *global activity* (blind to spatial
structure) and ξ is a spatial autocorrelation of the *change-indicator* field
(blind to token-configuration order, so it cannot see rule 110's gliders). Only
ρ/entropy separate the dead/periodic rule 250 from the active rules.

**What actually validates the pipeline is the qualitative space-time picture**:
`results/ref_rule110.png` is full of localized travelling gliders and no LLM
variant produces any — together with the damage-spreading analysis and the §12
locality contrast. The "no edge of chaos" negative is therefore meaningful, but it
rests on **visual + damage + locality** evidence, *not* on a τ_int/ξ edge-detector.
Consequence for everything below: **every τ_int/ξ magnitude (§3, §12, §16–19) is
descriptive of a transient, not a steady-state edge-of-chaos signature** — read it
with the space-time image, never alone. Full audit in §20.

## 2. Absorbing variant: a DP-style absorbing-state onset (`T_c ≈ 1.3`)

> *Per §20: "directed-percolation transition" is downgraded to "DP-**style**
> crossover" — an order parameter switching on is necessary but not sufficient for
> DP; no exponents (β, ν) were measured and n=3. The DP hypothesis is reasonable
> (single absorbing state, no conservation) but unproven without finite-size
> scaling.*

`results/sweep_phasediagram_absorbing.png`. With the "no spontaneous birth from
vacuum" rule, the all-`\n` state is a true absorbing state. Activity, entropy,
and spatial correlation length are **identically zero for T ≤ 1.2**, then switch
on sharply at `T = 1.3–1.4` — a clean DP-style absorbing-state onset.
But the active phase is weak (`ρ ≈ 0.01` even at `T=1.8`): the dead state
dominates the entire tested range. The dead token is `\n`, GPT-2's argmax from
BOS — its lowest-energy attractor from vacuum is structural whitespace.

*Caveat:* the committed absorbing figure's `τ_int` panel is a pre-fix artifact
(it plateaus at ~30 because it was computed over the full trajectory including
the slow absorbing transient; this is transient autocorrelation, not critical
slowing down). The metric was subsequently fixed to measure steady state.

## 3. Soft variant: order only at T=0, then straight to noise

`results/sweep_phasediagram_soft.png` and `..._soft_lowT.png`.

- **T = 0** (pure argmax, deterministic): nearly frozen — `ρ ≈ 0.027`,
  `ξ ≈ 10`, `τ_int ≈ 12`. A near-fixed-point with genuine spatial structure.
- **T = 0.025** (the smallest nonzero step tested): **discontinuous jump** to
  `ρ ≈ 0.65`, `ξ = 1`. The tiniest noise shatters the frozen state.
- **T = 0.025 → ~0.7:** a wide plateau at `ρ ≈ 0.65–0.75`, entropy ~2.3 bits,
  `ξ = 1` (no spatial correlation).
- **T ≳ 0.8 → 1.1:** ramps to full chaos (`ρ → 1.0`, entropy → ~7 bits ≈ the
  `log2(128)` ceiling).

`τ_int` is **highest at the lowest temperature and decays monotonically to ~0**;
`ξ` sits at its floor of 1 across the whole active range. These are descriptive
readouts, not edge detectors. There is no visual or damage-spreading evidence of
a critical point or Class-4 band.

Space-time textures confirm it visually: `T=0.5` (`spacetime_T0.5_*.png`) is
whitespace-dominated with horizontal *temporal* streaks but no spatial
structure and no diagonal glider tracks; `T=1.2` (`spacetime_T1.2_*.png`) is
pure static. Compare to the rule-110 baseline, which is full of diagonal
glider tracks.

## 4. Damage spreading: amplify-then-synchronize (the key subtlety)

`results/damage_soft_T*.png`. Two replicas differing in one token, driven by
the **identical** noise sequence (coupled noise via Gumbel-max). The Hamming
distance is **non-monotonic** at every temperature:

| run | short-time rate | peak separation | heals to |
|---|---|---|---|
| soft T=0.2 | +0.69 | 64/128 @ gen 6 | **0** |
| soft T=0.6 | +1.04 | 63/128 @ gen 4 | **0** |
| soft T=1.0 | +0.57 | 53/128 @ gen 7 | **0** |
| absorbing T=1.4 | +0.42 | 18/128 @ gen 7 | **0** |

A single flipped token **amplifies** explosively (short-time conditional
Lyapunov exponent > 0 — there is real local instability), then under the shared
noise the two replicas **re-converge to zero** within ~80 generations
(**common-noise-induced synchronization** over the measured horizon; do not read
this alone as a measured asymptotic exponent).

Interpretation: the deterministic skeleton of the map is locally expanding but
globally contractive under common noise. The high entropy/activity of the
"chaotic" regime is the temperature noise being faithfully copied, **not** a
strange attractor. Given the same noise, the system forgets its initial
condition. (For the absorbing run, synchronization is partly trivial — both
replicas drain into the same dead state.)

## 5. Why this answers the original thought experiment

Looping LLM inference does *not* spontaneously produce Conway-like persistent,
interacting structures. GPT-2's iterated map has only:

- a trivial ordered state (frozen at `T=0`, or the dead `\n` state under the
  absorbing rule), and
- a noise-dominated disordered state that synchronizes under common noise.

There is no intermediate complex regime. The "edge of chaos" it would need to
live at simply isn't there for this model under temperature control.

## 6. Caveats bounding the claim

- **Neighborhood is causal and long-range**, not Conway's local symmetric one.
  Each site is regenerated from its entire left context, so this is a directed,
  long-range automaton. The masked-LM variant (bidirectional, local-ish) is the
  real test of the Conway-faithful version and is the most important follow-up.
- **One small model, `L=128`, CPU.** `T_c` and transition sharpness will drift
  with model scale and lattice size; a genuine phase-transition claim needs
  finite-size scaling (vary `L`). Larger / more capable base models might behave
  differently — though the synchronization result suggests the qualitative
  picture may be robust.
- **n=3 seeds** — enough to see the phases, not for critical-exponent estimates.
- **Discrete argmax/sampling feedback** only; the continuous activation field is
  never fed back (by design — we iterate the observable token lattice).

## 8. Masked-LM (bidirectional) variant — one distilroberta pseudo-Gibbs test

`results/sweep_phasediagram_masked_soft.png`, `spacetime_masked_T*.png`,
`damage_masked_*.png`. distilroberta-base, `L=48`, symmetric neighborhood (each
site recomputed from the rest via one batched masked forward pass). Dead token:
`Advertisements` (the model's bare-`[MASK]` argmax).

*Scope/provenance caveat:* this is one `distilroberta-base` checkpoint under an
engineered synchronous pseudo-Gibbs update, not a native left-to-right generator.
Tokens like `Advertisements` and `Related` are attractor/provenance clues from
the model/tokenizer/training mixture, not audited source-document claims.

Two qualitative differences from causal:

- **Low T: collapses harder.** Even in *soft* mode (no absorbing rule) it drains
  to the dead state for `T < 1.2`, where causal-soft stayed active at `ρ ≈ 0.65`.
  Bidirectional context is a stronger consensus: every site sees the whole
  (increasingly uniform) sequence and agrees on the ground state. The dead→chaos
  transition is sharper and higher (`T_c ≈ 1.2`) than causal's.
- **High T: damage persists under common noise.** At `T = 1.0` a one-token
  perturbation fills the entire lattice (48/48) and stays separated over the
  measured horizon, vs causal's uniform healing to 0. This is real finite-horizon
  sensitivity to initial condition under the same Gumbel field, not a
  shared-noise tautology.

> **Matched-L reproduction (added per the §20 review — the sign-split claim needed
> a real control).** The original comparison
> was masked `L=48` vs causal `L=128`, figures-only. Re-run at **matched `L=48`,
> `T=1.0`, 12 coupled-noise pairs**, with committed CSVs
> (`results/damage_{causal_pythia-160m,masked_distilroberta-base}_soft_T1.0_L48.csv`):
>
> | arch (L=48, T=1.0) | short-time λ | final separation | verdict |
> |---|---|---|---|
> | causal (pythia-160m, full-attn) | +0.50 | **0.0 / 48** | synchronizes |
> | masked (distilroberta, bidirectional) | +0.17 | **47.2 / 48** | stays chaotic |
>
> The **architecture-dependent coalescence split holds at matched L**: both
> architectures have a positive *short-time* exponent (local instability), but only
> the causal map contracts back to synchrony under common noise over this horizon;
> the bidirectional map saturates.
>
> **Temperature × scale sweep (`results/lam_sweep_T.csv`, L=48, 10 pairs/cell).**
> The split is *not* a single-temperature accident; pythia covers the full sweep
> and Qwen3 checks the midrange at larger scale:
>
> | model (arch) | final separation, T = 0.4 → 1.4 |
> |---|---|
> | pythia-160m (causal, 160M) | **0.0 at every T** (always synchronizes) |
> | Qwen3-1.7B-Base (causal, 1.7B) | **0.0** at T=0.8/1.0/1.2 (always synchronizes) |
> | distilroberta (masked, bidir) | **29 → 43 / 48 at every T** (never synchronizes) |
>
> So the tested **causal full-attention maps synchronize in every measured cell**:
> pythia across `T=0.4–1.4`, and Qwen3 across `T=0.8/1.0/1.2` at 10× larger scale.
> None fails to coalesce despite positive short-time exponents (0.5–1.0). The
> consistency breakdown points to **bidirectional coupling**, but scale and
> temperature are not exhaustively ruled out. This is the sharpest form of the
> result: *full-attention causal LLM inference is a consistent (echo-state) map
> in the tested regimes; bidirectional coupling is what destroys consistency.*
> (Honest caveats: the masked side is one model with high pair-to-pair variance,
> std 14–23 — it is bimodal, most pairs saturate; the causal side is clean, std
> 0. Still open: a second masked model, larger scale, full Qwen3 temperature
> coverage, and larger-`L` window×temperature maps, plus finite-size scaling before
> any DP-universality claim.)

> **Update after §23.** The windowed-ring interpolation is now partly run. The
> binary causal-vs-masked story should be read as a slice through a broader
> finite-horizon damage-fate map: `w=1` local distilroberta coalesces, `w=2` is
> mixed, and `w>=3` usually preserves damage. The right next claim is about the
> geometry of that fate map, not an asymptotic Lyapunov sign from one horizon.

And the most structured texture in the whole study: the `T = 1.2` space-time
diagram shows **localized, persistent activity clusters drifting on the
quiescent dead background** — Class-4-*adjacent*, consistent with the masked
variant's elevated spatial correlation length (`ξ` up to 3 vs causal's 1). Still
no clean glider regime, but the closest any variant came.

## 9. Temperature × frequency-penalty phase plane

`results/sweep2d_causal_soft.png`, `sweep2d_raw_causal.csv`. GPT-2, `L=96`,
8 temperatures × 8 penalties × 2 seeds. The frequency penalty is the
homeostatic "overpopulation death" knob — a token's logit drops with its count
in the current generation — added to test whether a *balance* axis opens an edge
band the pure *disorder* axis (temperature) could not.

**It does not.** Across the entire plane:

- Activity saturates to `ρ ≈ 1.0` as soon as penalty ≥ 0.5 — the penalty kills
  the quiescent phase but only by forcing maximal turnover.
- `τ_int` is **highest at penalty = 0** and decays monotonically to ~0 as
  penalty rises — the penalty *destroys* temporal correlation. No interior ridge.
- `ξ` sits at its floor everywhere and collapses to 0 at high penalty.

So the penalty acts as a *second disorder knob*, not a balance knob. The
oscillatory-phase hypothesis (strong anti-repetition → periodic dynamics, which
would have shown as rising `τ_int`) is not borne out.

The reason is instructive: this penalty is **global** (whole-grid token counts),
whereas Conway's birth/death balance is **local**. A global balance has no
spatial degrees of freedom to organize — it can only force global turnover.

## 10. Unified conclusion: locality is the missing ingredient

Three independent control axes have now failed to produce an edge of chaos:

| axis | knob | result |
|---|---|---|
| disorder | temperature | frozen-only-at-0 → noise; no peak |
| balance | global frequency penalty | floods to max activity; `τ_int`, `ξ` → 0 |
| symmetry | masked / bidirectional | collapses lower, true chaos higher; mild structure |

The common thread: **full attention mixes the whole lattice every generation**,
so there is no finite signal speed and no way for a localized structure to
persist or propagate. `ξ` is pinned at its floor in every full-attention setting
because a perturbation reaches all sites in one step. The masked variant showed
*slightly* more spatial coherence precisely because the consensus dynamics
created transient localized agreement — but it is still globally coupled.

The prediction this licenses: an **edge of chaos, if reachable at all, requires
a local neighborhood** — a sliding-window-attention model, an explicit attention
mask restricting each site to ±w neighbors, or a state-space/recurrent model
with finite propagation speed. That is the experiment most likely to finally
produce gliders, and it is the top remaining item below.

## 12. Local windowed-ring variant — locality restores CA behavior

`results/spacetime_local_*.png`, `tokens_local_*.txt`. distilroberta with each
site predicted from only its ±w neighbours on a ring (periodic boundary), centre
masked. This is the first variant with finite signal speed, and it confirms the
section-10 prediction: **locality is what was missing.**

- **Propagation.** A single live cell on a dead background throws off fronts that
  travel at ~w sites/generation (the CA light-cone) — visible directly in the
  decoded grid: gen 0 touches one site, gen 2 spans ±2, etc. Full attention had
  no finite speed; this does.
- **Spatial structure.** `ξ` jumps to ~18 (w=2, T=0) vs 1 for every
  full-attention setting; smaller window and lower temperature give the most
  structure. `τ_int` rises to 10–14 simultaneously — spatial order *and*
  long memory at once.
- **Stable periodic motifs.** Decoded grids show columns that lock into a
  repeating pattern for many generations (oscillators / still-lifes).
- **Refractory → travelling, not filling.** Adding the Brian's-Brain decay
  (`--refractory R`: a cell dies after R generations alive) makes active regions
  migrate and leave dead space behind. The `w=2, T=0, refractory=3` run shows a
  propagating-front cone with a periodic checkerboard interior and diagonal
  travelling bands — the closest to gliders the project produced.

**What the cells are.** Decoding token ids (`--dump-tokens`) reveals the
"lifeforms" are the model's lowest-context attractor: for distilroberta, web
boilerplate — `Comments`, `Related`, `Posts`, `Tags`, `Next`, `Previous`, `»`.
With `--seed-text` you can steer the *sub*-genre (seeding `" comment reply"`
pulls it into comment-*form* chrome: `Cancel`, `Email`, `Save`, `Reply`) but not
escape the basin. These tokens are *mutually predictive* in training data, which
is exactly why they form stable self-consistent structures. The attractor is a
fingerprint of the model's training mix — a code model would converge to
brackets/`def`, an instruct model to assistant boilerplate.

*Metric caveat:* for the period-2 checkerboard textures the refractory rule
produces, `ξ` reads ~1 because adjacent cells *anti*-correlate (the
change-indicator autocorrelation drops below 1/e at lag 1). Low `ξ` here means
high-frequency structure, not absence of structure — read it together with the
space-time image, which shows strong order.

## 13. Identity probe: does an instruct model have a stable "I"?

Seed `" I am"` (one live cell on a dead background), `--arch causal`, T=0.3,
L=64, 150 steps; Qwen2.5-0.5B **base** vs **Instruct**. Reproduced
independently (this repo + a separate clone). The hypothesis: a base model has
no installed self and dissolves `" I am"`, while an RLHF instruct model locks
into a persona still-life (`" I am an AI assistant"`).

**The base half held; the instruct half did not.** Neither model has a stable
"I":

- **Base** → `" I am"` is gone by g1, replaced by a frozen still-life of
  **Chinese standardized-test boilerplate** ("以下是中国关于工程考试的单项选择题，
  请选出正确答案") with a churning A/B/C/D · 答案 option field. A training-data /
  training-objective fingerprint. No self.
- **Instruct** → also strips `" I am"`, but collapses to **code / API / doc
  boilerplate** (`/ json ### \`\`\` .md README python`) over a vast field of `0`,
  with only *transient* first-person flicker (stray `I`, `help`, `Please`) that
  never locks. No persona still-life.

**The sharpest single result is the vacuum-token flip.** The dead token (argmax
from BOS) is `'Human'` for the base and `'/API'` for the Instruct model — post-
training moved the model's ground state from a conversational turn-marker to a
code/tooling token. One token of evidence that Qwen2.5's post-training is
code/agent-weighted, not persona-weighted.

**Robustness:** the dead-token flip, the entropy drop (Instruct ~2.0 vs base
~4.4 bits), and the activity drop (~0.40 vs ~0.60) reproduced cleanly across
runs. `τ_int` did **not** — one run measured Instruct ≫ base (16.8 vs 3.9),
another a much smaller gap (6.4 vs 4.0); `τ_int` is the correlation time of a
fluctuating series and is seed-sensitive, so its *magnitude* is not a reliable
single-run statistic here (the *direction*, Instruct deeper, held).

**Chasing the persona (the obvious objection).** The assistant persona lives
behind chat scaffolding, so maybe bare `" I am"` never enters the chat manifold.
Seeding `"<|im_start|>assistant\nI am"` at T=0 was tested — and the persona
*still* did not appear: the scaffold tokens vanish by g1 and the system freezes
to a date/number still-life (`/ 2 0 2 3 … 0 0 0`, ρ=0.026, entropy 0.92 bits,
the deepest fixed point observed). The reason is structural: the CA needs a
uniform dead background, but a chat persona needs a coherent multi-token frame —
an embedded scaffold can't establish that frame against a sea of dead tokens.

**Conclusion.** The assistant persona is a **conditional** attractor — a
response to a fully-formed chat context, not a standing structure in the model's
*unconditional* token dynamics. RLHF didn't install a self that the loop can
surface; it **deepened and relocated** the unconditional basin (sharper, lower-
entropy, toward code/tooling). Strip the frame and there is no "I" lurking
underneath — there's a code-and-zeros fixed point. That is a more interesting
answer than the one predicted, and a slightly humbling one.

## 14. Local frequency penalty — a sparse regime, but a fragile structure

Next-step #2 (below), now implemented: `LocalMaskedLMAutomaton` gets a
*neighbourhood-local* frequency penalty — each site's logit for a token is cut
by `freq_penalty ×` (count of that token among its ±w spatial neighbours, centre
excluded), the local analogue of Conway's overpopulation death. (The global
penalty of §9 can only force whole-grid turnover; this one couples to local
composition.)

**What it buys: a genuinely new sparse regime.** Without it the local rule only
ever *fills* (global oscillator, ρ→1) or *dies* (vacuum). At intermediate penalty
an in-between opens: bert-base-uncased (dead token `'.'`, which dies at
essentially every zero-penalty setting) instead settles at low density — ρ ≈
0.07–0.25, a small live region on an otherwise-dead lattice. First time activity
neither fills nor drains.

**The catch: the structures in it are not robust.** The cleanest hit —
bert-base `w=2, freq_penalty=2.0`, single seed, T=0 — is a **period-8 localised
oscillator**: ~5–7 cells of conversational filler/punctuation (`and but … it oh
the right "` ↔ `" " , " i … " the`) in a ~16-cell envelope, holding 100+ gens.
But it is **not lattice-size-robust**, which a truly self-contained localised
structure would be:

| L | 48 | 56 | 64 | 72 | 80 | 88 | 96 | 112 | 128 |
|---|---|---|---|---|---|---|---|---|---|
| outcome | aperiodic | aperiodic | **period-8** | aperiodic | **period-8** | aperiodic | dead | dead | aperiodic |
| tail ρ | 0.42 | 0.29 | 0.24 | 0.08 | 0.08 | 0.07 | 0.00 | 0.00 | 0.14 |

The clean oscillator exists only at `L = 64, 80`; elsewhere it churns
aperiodically or drains to vacuum. On a ring, a self-contained localised pattern
should not care how much dead background surrounds it — so the L-sensitivity says
this is a **commensurate standing-wave resonance perched at the edge of the
vacuum basin**, not a free lifeform.

**Gliders: none.** A fine-scan (24 seeds × penalty × refractory) found no
translating localised structure. The single-cell seed washes out by g1, so the
outcome is seed-independent. A centre-of-mass "drift" detector flagged
candidates, but all were high-density (ρ≈0.55) delocalised patterns drifting
*faster than the ±w light cone* — artifacts of measuring a centroid on a
non-localised pattern, not spaceships.

**Standing:** the local penalty is the right mechanism and opens the sparse
territory the global knob could not, but the bar for a lifeform — a structure
that is *both* low-density *and* robust across L — is unmet.

**The (L × window × penalty) map closes that thread (negative).** Across
windows 1–3 × penalties 1.5–3.0 × eight lattice sizes (L = 56…112), **no
(window, penalty) holds a low-density localized periodic structure at more than
2 of 8 sizes** — the best is `w=2, pen=2.0` at 2/8 (period-8, only L=64,80), and
there is no single period that recurs across many L. So no lifeform candidate.
The phase structure is clean, though: the penalty carves a knife-edge — `w=1`
drains to vacuum almost everywhere; `w=3` fills (ρ≈0.9) at every penalty; `w=2`
sits *between* them, and only there, in a thin sliver around `pen≈2.0`, does
localized structure appear at all — tipping to fill by `pen≈2.5`. The localized
structures that live on that sliver are **standing-wave resonances pinned to
commensurate L, not free lifeforms.**

This is the cleanest statement of the project's negative: with all three of
Conway's ingredients now present — **locality** (windowed ring), **refractory
death**, and a **local birth/death balance** (this penalty) — there is still no
robust localized still-life, oscillator, or glider. The missing piece is not a
knob left unturned; it is that an LLM's learned local rule lacks Conway's
fine-tuned `B3/S23` balance. The penalty pushes the dynamics to the edge between
vacuum and fill, but for these models that edge is a **lattice-size-pinned
knife-edge**, not a stable basin a lifeform can occupy.

## 15. Seeding meaning: the iterated map is indifferent to it

The geometry experiments (§12–14) seed live *cells*; here we seed meaningful
*word-strings* into the **causal full-attention** rule (where each site sees its
whole left context, so semantics actually operate) and watch the transient
before the genre attractor wins. Qwen2.5-0.5B, soft (not absorbing), T=0.3,
L=64. Three findings, one law.

**1. Contradictions are "anti-still-lifes."** Internally-conflicting seeds —
`yes no`, `true false`, the liar paradox, `war is peace`, `I must obey / I
cannot obey`, `left right` — are erased *fastest* (gone by g1–2), *faster* than
coherent text (g4–5). No pole ever wins or oscillates; the conflict simply
evaporates. Robust across base/instruct and T=0/0.3 — only the destination
basin changes (base → Chinese exam text, dead token `Human`; instruct → a
`/API` + field-of-`0` code basin). At T=0, base lands on a *fully grammatical*
sentence; instruct collapses to nearly pure `0` (the emptier basin).

**2. It is frequency, not meaning.** Evocative, image-dense phrases (`velvet
shadows…`, `glittering crimson splendor…`) die *as fast as contradictions*
(g2–3) and faster than dull common text — because ornate words are
*low-probability*, so the rule overwrites them on contact. Per-token survival on
`" the speed of light is the only constant in the universe"` is the cleanest
demonstration: only `the` survives (to g43); every content word (`light`,
`constant`, `universe`) is gone by g0–1. *The system's actual "only constant" is
the word "the."* Genre-match doesn't help either — an engineering claim matching
the exam basin dies just as fast. Quantified across 79 seed tokens
(`results/seed_survival_vs_logprob.png`, `seed_survival_data.csv`): survival vs
the model's unconditional log-prob (its frequency prior) is **a threshold, not a
gradient** — **86% of tokens die by gen 1 regardless of frequency**, and only
the 3–4 highest-frequency function words (`the`, `a`, `is`) escape the floor
(Spearman ρ≈0.29, weak precisely because the relationship is winner-take-all).
The rule preserves the bland-and-frequent and erases the contradictory, the
beautiful, and the profound alike.

**3. A memory substrate delays but does not defeat — and reveals a negativity
bias.** The synchronous update overwrites every site at once, so a chain of
reasoning has no substrate to form in. The asynchronous variant (`update_frac` <
1.0; §code) resamples only a fraction of sites per generation, freezing the rest
as committed memory. At `update_frac=0.1` a conflict that vanishes by g1–3
synchronously survives to **g34–38** — and the *negation* token consistently
outlives its partner (`no` outlives `yes`, `cannot` outlives `must`; the last
conflict-token standing is always the negative one). But no stable
conflict-resolution structure forms: memory stretches the transient and exposes
the bias, then the dissipative genre attractor still wins the fixed point.

**Conclusion.** As a dynamical system the iterated map is a **frequency filter,
not a reasoner**. It strips any seed — contradictory, ornate, profound, or
on-topic — to its most frequent grammatical tokens within a generation, then
overwrites with the model's training-genre reflex. Meaning is not a conserved
quantity; the only thing it reliably keeps is `the`, and the only thing it
"prefers," given memory, is *no*.

## 16. Recurrent / state-space models (RWKV-4, Mamba): architecture gives finite-*effective*-range propagation, but not lifeforms

`results/spacetime_causal_{rwkv-4-169m-pile,mamba-130m-hf}_*.png`,
`tokens_causal_*`, `single_causal_*`. RWKV-4-169m-pile and Mamba-130m-hf run as
drop-in `--arch causal` models. This tests §11 next-step #1: the hypothesis that
a recurrent/SSM model's *decaying* memory gives an effectively-local, finite-speed
neighbourhood that full attention lacks — and might therefore support gliders.
Single live cell on a dead background, L=96, T ∈ {0, 0.3}.

**"Finite signal speed," stated honestly.** In the synchronous-CA setting one
generation is a single parallel forward pass, and a causal SSM still ingests its
*entire* left context in that pass (the recurrent state summarises sites 0..k-1).
So there is no per-generation finite signal speed in general — same as full
attention. What an SSM/RNN *does* add is a strong **recency bias** (RWKV's
time-mixing decays exponentially), making the effective leftward neighbourhood
short-range. The question is whether that *effective* locality changes anything.

**A. Absorbing + single cell → a drifting front, then vacuum.** All four runs
end in the all-`Q` absorbing state. But the route there is a propagating front,
not a collapse in place:

| run | behaviour | final |
|---|---|---|
| RWKV  T=0.0 | one live (whitespace) cell drifts right at **+1 site/gen**, runs off the boundary at g49 | dead g49 |
| RWKV  T=0.3 | brief spread to ~5 cells, drifts right, dies | dead g8 |
| Mamba T=0.0 | single-cell drifter at +1 site/gen, dies spontaneously | dead g7 |
| Mamba T=0.3 | no propagation | dead g2 |

The RWKV T=0 space-time diagram is a single clean diagonal at slope +1 — a
ballistic drifter — on an otherwise dead lattice. **Dead token = `'Q'`** (token
50, argmax from BOS) for *both* models (shared NeoX/Pile tokenizer) — a third
distinct ground state alongside GPT-2's `\n` and distilroberta's `Advertisements`.

*The drift is mostly the rule, not the model.* Under the absorbing causal rule
the vacuum mask forces the leftmost live cell dead every generation while leaving
sites to its right free to sample, so the live frontier can only march right at
the +1/gen "light cone." The model's only job is to decide whether to populate
the new frontier site with a live token (front advances) or the dead token (front
collapses). The recency-biased SSMs keep emitting a token at the frontier, so the
front advances. (A matched single-cell control — §19 — shows this advancing drift
is **universal** across causal models, full-attention pythia included, *not* an
SSM trait; the "GPT-2 drains" comparison was §2's *random*-init run, a different
experiment.) This is a degenerate, content-free echo of the §12 windowed-ring
propagation — reached via the absorbing rule, not an explicit mask — and it leaves
pure vacuum behind, not a structure.

**B. Soft (non-absorbing) + single cell → the absorbing rule was doing the
killing.** Remove the vacuum mask and the picture inverts. RWKV at T=0 from the
*same* single cell does **not** collapse — it nucleates and fills the lattice to
a stable fixed point with real structure:

| metric | RWKV soft T=0 | GPT-2 soft T=0 (§3) |
|---|---|---|
| activity ρ | 0.27 | 0.027 |
| entropy (bits) | 4.25 | — |
| ξ | 11 | ~10 |
| τ_int | 14.6 | ~12 |
| final live density | 0.99 | — |

And the fixed point is a **coherent, grammatical English sentence** — a Pile
code-Q&A still-life, *"How to get the value of a variable in a function? … I have
a function that takes a variable and returns the value of that variable …"* —
identical from **g60 through g120** (a frozen fixed point; the ρ=0.27 is the
nucleation transient). It is markedly more "alive" than GPT-2's soft-T=0 state,
which has the same ξ but ~10× lower activity and is whitespace-dominated. It is
tempting to credit recurrence for growing a *globally self-consistent sentence*
where GPT-2 grows whitespace — but §17 shows a *full-attention* Bonsai-1.7B fills
too, so it is not recurrence. It is **not scale either**: the matched full-precision
Qwen3-1.7B-Base (same arch and scale as Bonsai) half-freezes instead of filling
(§19). The fill is **checkpoint-specific**. As in §12/§13/§15, the stable structure
is a training-data fingerprint: it is built from the model's own most mutually-
predictive tokens.

**Conclusion: the SSM hypothesis is partly right and mostly wrong.** *Right:*
architecture alone (recency bias) does change the dynamics — it produces a
propagating front under the absorbing rule and a coherent-sentence still-life
under the soft rule, neither of which GPT-2 produces. *Wrong:* none of this is a
localized lifeform. The absorbing "drifter" is a frontier artifact carrying a
single whitespace token; the soft fixed point is a **global** still-life (the
whole lattice is one sentence), not a localized, translating, interacting
structure. As in §14 the missing ingredient is not architecture but Conway's
fine-tuned *local birth/death balance*: a causal SSM gives finite *effective*
range but no rule that lets a bounded droplet sit stably between growth and
death. The genuine locality win remains the explicit windowed-ring rule (§12).

*Caveats:* single seeds, single-cell, T ∈ {0, 0.3} only; the absorbing "drift" is
substantially a property of the absorbing rule (confirmed by the soft control,
which fills instead). A matched full-attention NeoX-tokenizer control
(pythia-160m absorbing + single-cell at T=0) is the clean architecture-isolating
experiment to add — it was not run here because the cached pythia weights were
incomplete; the GPT-2 comparison (§2/§3, different tokenizer) stands in for it.

## 17. Ternary-trained weights preserve the same causal repertoire

`results/spacetime_causal_Ternary-Bonsai-1.7B-unpacked_*`,
`tokens_causal_Ternary-Bonsai-1.7B-unpacked_*`,
`damage_causal_Ternary-Bonsai-1.7B-unpacked_*`. prism-ml's **Ternary-Bonsai-1.7B**
— a Qwen3-1.7B-architecture model trained to *ternary* weights {−1, 0, +1}, here
in the "unpacked" checkpoint (ternary values materialised to standard tensors) so
it loads as a plain `Qwen3ForCausalLM`. Question: does this trained-ternary
checkpoint change the iterated-map *dynamics*, or only the attractor's token
content?
`--arch causal`, L=96.

**The full CA repertoire is intact, cleanly:**

| run | ρ | entropy | τ_int | ξ | outcome |
|---|---|---|---|---|---|
| absorbing, single cell, T=0 | 0.014 | 0.30 | 10.6 | 2 | +1/gen frontier drift → vacuum |
| soft, single cell, T=0 | 0.21 | 4.42 | 14.5 | 10 | coherent still-life (filled) |
| soft, random, T=0.3 | 0.44 | 2.14 | 5.9 | 5 | active, partially-ordered |
| damage, soft, T=0.6 | — | — | — | — | λ_short=+0.75, heals to 0/96 |

- **Absorbing single cell → the same +1/gen frontier drifter as RWKV/Mamba (§16),
  then vacuum.** Because Bonsai is *full-attention* (Qwen3), this independently
  confirms the §16 drift is a property of the **absorbing rule**, not recurrence.
- **Soft single cell → a coherent still-life.** The single cell nucleates and
  freezes (g80→g120) into a grammatical *Chinese* factual sentence — *"1990年，
  中国在联合国大会上投票通过了《联合国公约》… 公约的签署标志着中国正式进入国际法
  体系。"* (repeating). The dynamical signature (ρ=0.21, ξ=10, τ_int=14.5, live=0.99)
  is nearly identical to RWKV-soft (§16); only the genre differs — a Qwen
  training fingerprint (Chinese encyclopedic text), echoing the §13 identity probe
  where Qwen base collapsed to Chinese exam boilerplate. Ternary did **not** blunt
  the attractor: the model still falls into a coherent, self-consistent fixed point.
- **Damage (coupled noise), T=0.6 → amplify-then-synchronize.** Positive short-time
  conditional exponent (+0.75 — real local instability) but the Hamming distance
  heals to **0/96**: the §4 common-noise-induced synchronization, reproduced. The
  high-T "chaos" is faithful transcription of the injected noise, not a strange
  attractor — as for every causal full-attention model tested.
- **Dead token = `':'`** (token 25, argmax from BOS) — a fifth distinct ground
  state (GPT-2 `\n`, distilroberta `Advertisements`, RWKV/Mamba `Q`, Bonsai `:`).

**Two cross-cutting conclusions:**

1. **This trained-ternary checkpoint preserves the causal repertoire.** It shows
   the same phases (vacuum collapse, fill-to-still-life, common-noise
   synchronization) seen in full-precision causal controls. This does **not**
   prove a generic 2-bit=f16 rule: Bonsai was trained in ternary form and has no
   same-checkpoint full-precision twin; only the *genre* of the attractor tokens
   is clearly tied to the training mix.
2. **Fill-vs-freeze is checkpoint-specific** — not attention topology, and (per
   the §19 control) **not scale either.** GPT-2 (124M, full attention) *freezes*
   to whitespace under soft T=0 (§3); RWKV-169m and Bonsai-1.7B both *fill* to a
   coherent-sentence still-life — so it is not a recurrence effect. But the
   matched full-precision twin, Qwen3-1.7B-Base (same arch, same 1.7B scale),
   does **not** fill (it half-freezes, live=0.44; §19) — so it is not a scale
   effect either. The soft-T=0 attractor is set by the specific checkpoint's
   training. (§19 has the control; this supersedes the earlier "tracks scale"
   reading.)

*Caveats:* single seeds; the "unpacked" checkpoint stores ternary values in fp16,
so this measures the *trained-ternary network's* dynamics, not behaviour under
live 2-bit packed kernels and not a broad quantization claim. The native mlx-2bit
variant only tests whether runtime packing perturbs this same checkpoint (§18).

## 18. Native 2-bit MLX packing validates the same-checkpoint backend

`results/spacetime_mlx_*`, `tokens_mlx_*`, `damage_mlx_*`. A new MLX backend
(`--arch mlx`, via `mlx_lm`) iterates genuinely low-bit checkpoints on Metal: the
forward runs in MLX and the logits are bridged back to torch at the automaton
boundary, so the map / Gumbel sampling / coupled noise / metrics are reused
unchanged. We compare prism-ml's Ternary-Bonsai in its **native 2-bit MLX
packing** against the same model **unpacked to fp16** (§17), at two sizes.

**Result 1 — same-checkpoint 2-bit packing converges to the same attractor as
fp16-unpacked.** *(Framing corrected per §20: this is a near-tautology — the
unpacked and mlx-2bit checkpoints are the **same trained ternary network** in two
storage formats, so ≈identical logits are expected, not discovered. It validates
the MLX backend; it is not evidence about quantization in general. And it is not
"exact": the
**transients differ** (decoded grids diverge at several generations); only the
**fixed point** matches.)* 1.7B-mlx-2bit vs 1.7B-unpacked, identical settings
(τ/ξ shown are full-trajectory transients, ≈0 at steady state — see §1/§20):

| run | 1.7B-mlx-2bit | 1.7B-unpacked (§17) |
|---|---|---|
| abs single T=0 | ρ=0.0143, H=0.3005, τ=10.58, ξ=2 → vacuum | ρ=0.0143, H=0.3005, τ=10.58, ξ=2 → vacuum |
| soft single T=0 | ρ=0.206, H=4.417, τ=14.7, ξ=10 | ρ=0.207, H=4.417, τ=14.5, ξ=10 |
| soft fill content | *identical Chinese sentence* | *1990年…国际法体系* |
| damage soft T=0.6 | λ_short=**+0.747**, heals to 0/96 | λ_short=**+0.745**, heals to 0/96 |

The two reach the same fixed point with the same phase (vacuum / fill / sync) in
the ordered and chaotic regimes — confirming the MLX backend and packing introduce
no dynamical artifact. **But because it is the same trained network, this is a
backend-validation, not a quantization result.** The genuinely non-tautological
quantization test — quantize a *non-ternary* model (e.g. Qwen3-1.7B-Base) to 2-bit
and compare to its own fp32 self — has not been run; it is the right next step
(§20). The substantive quantization-era claim is §19's: the attractor is set by the
**checkpoint/training**, not the bit-width.

**Result 2 — the checkpoint, not the bits, drives the attractor.** The 8B-mlx-2bit
soft single-cell run *also* fills (live 0.99) but into a **degenerate near-uniform
field of `0`** (ρ=0.04, entropy **0.34 bits**, ξ=2) — not a coherent sentence.
Same 2-bit packing as the faithful 1.7B, so this is the *checkpoint*: the 8B's
basin is the code/zeros field §13 found in instruct/agent-weighted models, where
the 1.7B base falls into a Chinese-factual sentence. (Whether the 8B Bonsai is
instruct-tuned or merely code-weighted is unverified; the basin shape matches
§13's instruct models.) Dead tokens differ too: 1.7B `':'`, 8B `' '` (space).

**Conclusion.** Across full-precision (GPT-2, Qwen), recurrent (RWKV, Mamba),
trained-ternary-in-fp16 (§17), and same-checkpoint native 2-bit (here), the
coarse iterated-map phases recur. The only direct 2-bit comparison here validates
MLX packing/backend faithfulness for the same Bonsai checkpoint; it does not show
that arbitrary 2-bit quantization matches full precision. What changes is the
attractor's *content*: the genre of its tokens and the richness of its fixed
point, which track the training mix and post-training.

*Caveats:* single seeds; the MLX bridge runs the torch side on CPU (fine — the
forward dominates); the 8B was characterised on the single-cell runs only.

## 19. Matched controls overturn two earlier claims (this is why we ran them)

`results/tokens_causal_pythia-160m_*`, `tokens_causal_Qwen3-1.7B-Base_*`,
`spacetime_causal_{pythia-160m,Qwen3-1.7B-Base}_*`. Two controls — a
tokenizer-matched full-attention baseline (**pythia-160m**, GPT-NeoX) and a
same-architecture, same-scale, full-precision baseline (**Qwen3-1.7B-Base**, the
non-ternary twin of Bonsai-1.7B) — each refute an overreach above. The harness
was built to let controls do exactly this.

**Control A — the absorbing drift is universal, not a recurrence signature.** §16
read the RWKV/Mamba +1/gen frontier drift as something the recency-biased SSMs do
that GPT-2 does not — but the GPT-2 comparison it leaned on (§2) was *random*-init,
not single-cell. The matched single-cell control settles it: every causal model
drifts at +1/gen to vacuum.

| absorbing, single cell, T=0 | arch | dead | ρ | ξ | final live |
|---|---|---|---|---|---|
| pythia-160m | full-attn (NeoX) | `Q` | 0.064 | 10 | 0.00 |
| RWKV-4-169m | recurrent | `Q` | 0.021 | 2 | 0.00 |
| Mamba-130m | SSM | `Q` | 0.020 | 2 | 0.00 |
| Bonsai-1.7B | full-attn (Qwen3, ternary) | `:` | 0.014 | 2 | 0.00 |
| Qwen3-1.7B-Base | full-attn (Qwen3, fp) | `Human` | 0.048 | 8 | 0.00 |

So the drift is the **absorbing rule's frontier** — universal across causal
models; the architecture only sets the transient's width (ξ) and which token rides
the front (pythia: `<|endoftext|>`; Qwen3-Base: a leftward phrase fragment
*"…a is what,"*). §16's core mechanism ("the absorbing rule, not the model") was
right; its hint that *advancing* vs *draining* separates SSMs from full attention
was not — they all advance, then run off the lattice into vacuum.

**Control B — fill-vs-freeze is checkpoint-specific, NOT scale.** §17 concluded the
soft-T=0 *fill* (Bonsai reaching a coherent still-life vs GPT-2 freezing) "tracks
scale, not attention topology." The full-precision twin refutes it. **Qwen3-1.7B-
Base — same Qwen3 architecture and 1.7B scale as Bonsai — does not fill:** soft
single-cell T=0 reaches a sparse, half-lattice, near-frozen state (ρ=0.035,
live=0.44, ξ=5; the right half locks into a `1 · 1 · 1 0 · 1` digit/dead
alternation, the left half stays dead), where Bonsai-1.7B fills the whole lattice
(live=0.99) with a coherent Chinese sentence. Same architecture, same scale,
opposite fill. So the soft-T=0 attractor (freeze / partial-fill / full coherent
fill) is a **per-checkpoint, training-dependent** property — it does not reduce to
scale or attention topology. Precision remains unisolated by this cross-check
because Bonsai and Qwen3-Base are different checkpoints.

**What this does and does not say about ternary.** It does *not* make Bonsai's
richer fill a ternary effect: Bonsai and Qwen3-1.7B-Base are *different checkpoints*
(different ground states — Bonsai `:` vs Qwen3-Base `Human`), so this comparison
cannot isolate quantization from training. §18 is same-checkpoint backend/packing
validation, not the clean quantization test. The clean test remains: quantize a
non-ternary model (e.g. Qwen3-Base) and compare it to its own full-precision
checkpoint. Net: Bonsai packing is transparent for the same checkpoint, but the
*attractor* is set by the checkpoint's training, and even same-arch/same-scale
checkpoints differ sharply.

**Ground-state catalogue** (argmax from BOS — a tokenizer/training fingerprint):
GPT-2 `\n`, distilroberta `Advertisements`, pythia/RWKV/Mamba `Q` (Pile/NeoX),
Bonsai-1.7B `:`, Qwen3-1.7B-Base `Human` (= Qwen2.5-base, §13), Bonsai-8B `' '`,
and — qualitatively different — **gemma-4-12B `<image|>`** (a *multimodal* token,
not a word; §22).

The method working as intended: the controls killed two tidy generalizations and
left the defensible core — the absorbing drift is rule-universal, and the soft-T=0
attractor is checkpoint-specific.

## 20. External review: audit, corrections, and the one result worth elevating

A three-reviewer panel (complexity-science, ML-literature, methodology lenses)
audited §1–19 against the code and the literature, debated to convergence, and ran
several of its own reproductions. Headline verdicts (no edge of chaos; §19
checkpoint/training controls; same-checkpoint MLX backend validation) **survive**;
several quantitative and novelty claims were **corrected**. The fixes above (§1,
§18, TL;DR scope) are already applied; this section records the rest.

**Verified bugs / overclaims (corrected):**

1. **τ_int/ξ never discriminated Wolfram class (§1).** Confirmed by re-running the
   init×burn matrix: under matched random init, rule-110 τ_int=0.47 ≈ rule-30's
   1.39, and ξ=1.0 for 110/30/90 alike. Burn-in does *not* rescue it (single-init
   post-burn: 110≈30≈250). Fixed in `cmd_reference`; §1 rewritten.
2. **Single-run τ_int/ξ are transient, not steady-state (§16–19).** `cmd_single`
   computed them over the full trajectory (no burn-in), unlike `cmd_sweep`. Every
   §16–19 run *freezes* (activity→0), so the true steady-state τ_int≈0; the
   reported 10–15 was 100% nucleation transient. Fixed in `cmd_single` (now
   post-burn). Treat all §16–19 τ_int/ξ/ρ magnitudes as **directional, single-seed,
   transient** — never steady-state signatures.
3. **§4/§16 synchronization is a known phenomenon, not novel.** It is the
   echo-state/**consistency** property (Lymburn et al., *Chaos* 2019; Mainen–
   Sejnowski 1995; common-noise sync, Pikovsky/Toral); the *identical* shared-Gumbel
   mechanism is already published on LLM sampling (*Recycled Gumbel Noise*, NAACL
   2025, arXiv 2503.00831). It is a **real** common-noise consistency effect (not
   a tautology — proof below), but bill it as a re-instance + diagnostic, not a
   discovery.
4. **"Directed-percolation transition" (§2) is unsupported** — no measured exponents
   (β, ν⊥, ν∥), n=3. Downgrade to "DP-style crossover" pending finite-size scaling.
5. **"Fingerprint of *the checkpoint*" is overstated (§16/§17/§19).** A control the
   panel ran — pythia-160m (full-attention) and RWKV-169m (recurrent), two
   architectures sharing only the NeoX tokenizer + Pile corpus — converge to the
   **same** Pile code-Q&A still-life ("How to get the value of a variable in a
   function?…"), identical across 8 seeds. The attractor is a **corpus/tokenizer-
   family** property, not checkpoint-unique. Correct reading is a three-level
   hierarchy: *collapse* itself = generic data-driven degeneration (universal, not a
   finding; Holtzman 2020, "Repetition In Repetition Out" 2310.10226); *what
   survives* (frequent > ornate) = self-predictivity (§15); *which basin* = training
   — coarse genre set by the corpus family, specific basin selected by post-training
   (§13 base-vs-instruct flip; §19 Bonsai-vs-Qwen3-Base twins reach different
   basins). Soften the single-greedy-run genre reads; keep §13/§19's controlled
   basin-selection signal.
6. **Absorbing "+1/gen drift" is rule-dominated (§16).** The frontier march and
   drain-to-vacuum are geometrically forced by the vacuum mask (model-independent);
   peak live density before draining spans 50× (Mamba 1, RWKV 5, Qwen3-Base 23,
   pythia 32, Bonsai 46/96) — only RWKV/Mamba "drift" cleanly, the rest *bloom*
   first. So absorbing single-cell is a weak model-discriminator.
7. **Ground-state catalogue (§13/§16–19): real but shallow.** argmax-from-BOS is
   BOS-convention-dependent and partly circular with the absorbing rule (which is
   defined by force-killing to it); the catalogue is high-frequency structural
   tokens — a tokenizer/frequency fingerprint, not a deep semantic probe.
8. **Masked-rule scope/provenance (§8).** The masked result is one
   `distilroberta-base` checkpoint under an engineered synchronous pseudo-Gibbs
   update. Its boilerplate tokens are attractor/provenance clues, not audited
   source-document claims.

**What the panel verified as SOLID (some strengthened):**

- **§19 Control B is seed-robust** — multi-seed test (which the doc lacked):
  pythia/RWKV fill 0.990 across 8 seeds (std 0.000); Qwen3-1.7B-Base freezes to
  0.438 across 3 — so *fill-vs-freeze is checkpoint-specific, not architecture /
  scale*. Precision still needs the non-ternary same-checkpoint quantization test.
  The earlier n=1 worry is resolved (the *attractor content* is near-deterministic;
  only the τ_int/ξ *magnitudes* are fragile — two different reliabilities).
- §19 Control A (absorbing drift rule-universal), the SSM finite-*effective*-range
  caveat (§16), §15's frequency filter (Spearman ρ=0.285 reproduced), and the
  visual + damage + §12 locality evidence for the headline negative.

**Synchronization billing after §23:** *"Temperature-sampling 'chaos' in iterated
LLM inference is often common-noise consistency — faithful transcription of
injected noise — but coalescence depends on architecture, model, local window, and
temperature."* It is **not** a tautology: under the same shared noise, causal heals
to 0 but global masked distilroberta **stays separated** (§8, §23). If shared noise
forced sync, masked/local non-coalescing cells could not exist. Honest hedge:
current `p_sync` and `late_damage` are finite-horizon endpoint metrics, not
measured asymptotic Lyapunov exponents.

**The live, pursuable result.** The contributions that survive as new (vs the
non-spatial iterated-inference work — Zhilin Wang et al., *Attractor Cycles in
LLMs*, ACL 2025, arXiv 2502.15208; and vs LifeGPT npj 2025, which is the *inverse*
problem) are: (i) the **spatially-extended synchronous-CA apparatus** with a
tunable coupling neighborhood, and (ii) the **common-noise damage-fate map** (§23).
The old architecture sign split is still useful, but it is one slice through this
map: causal pythia/Qwen coalesce in tested cells; global masked distilroberta does
not; BERT/CodeBERTa and local `w=2` are mixed. The masked rule's *iteration* is
itself known — Gibbs sampling from a BERT MRF: Wang & Cho 2019 arXiv 1902.04094;
Mask-Predict, Ghazvininejad 2019 arXiv 1904.09324 — but the CA damage-map framing
is the useful new apparatus here.

**Prioritized firm-up list (updated after §23):**
1. Scale the local window×temperature fate map in `L`, horizon, and pair count;
   especially test the `w=2` boundary and the `w=2→3` wall.
2. Add richer nulls: logit shuffle, random-initialized MLM/causal, and
   temperature-matched marginal baselines.
3. Add a second masked model, larger causal scales, and entropy-matched
   temperatures so coalescence failure is mapped across model family, not inferred
   from one masked checkpoint.
4. Finite-size scaling of the absorbing T_c (vary `L`) to earn or drop the "DP"
   label.
5. The non-tautological quantization test (Qwen3-Base 2-bit vs its fp32 self);
   ~~replace τ_int/ξ with configurational token correlation and spatial
   mutual-information diagnostics~~ **— done in §24** (`complexity.py`: excess
   entropy + entropy rate + configurational token MI + local transfer entropy,
   validated on the reference CAs).

*Process note:* findings cross-checked by three independent reviewers to
convergence; the central bug (§1) and the §19 Control B robustness were reproduced
locally. Key refs added: 2502.15208 (ACL 2025), 1902.04094, 1904.09324, 2503.00831
(NAACL 2025), 2310.10226, 2510.22954 (NeurIPS 2025), 2509.26643 (EMNLP 2025),
1901.07729 (*Chaos* 2019).

## 21. Seed engineering in the local rule: seeds nucleate bounded structure, but still no free life-form

`scripts_seed_hunt.py`, `scripts_seed_Lrobust.py`, `results/seed_hunt.csv`,
`results/seedhunt_*.txt`. Tests the hypothesis licensed by §8: because the
local/masked rule is *chaotic* (it remembers its seed) whereas the causal rule
*synchronizes* (it erases any seed — §15's frequency filter), an **engineered
seed** — a local fixed point, *self-predictive inside* and *predicts-dead at its
boundary* — might nucleate a bounded, persistent, L-robust life-form where random
or single-cell seeds (§14) only fizzle. Local windowed-ring (`w=2`, `T=0`,
absorbing, local overpopulation penalty), across CodeBERTa / bert-base /
distilroberta × penalty/refractory × ~17 seed families (brackets, attractor
tokens, periodic motifs, asymmetric, controls), with a glider detector
(period-up-to-ring-translation). Three findings, one law.

**1. Life needs a *non-spreading* local conditional — only bert-base qualifies.**
CodeBERTa (code) and distilroberta (web boilerplate) *flood* the lattice from
almost any seed (ρ_live → 1): their attractor tokens are mutually predictive in a
way that spreads — every `(` predicts more code, every `Comments` predicts more
chrome. Only **bert-base** (dead token `.`, a sparse punctuation/filler genre)
supports bounded structures, and only with the local penalty (`freq_penalty≈2.0`);
at penalty 0 its seeds die to vacuum. The local rule hosts structure only when the
model's own local conditional is not a spreader.

**2. The seed selects the structure (the hypothesis, confirmed — narrowly).**
Within bert+penalty the *seed tokens* pick the outcome: `( a )` → a period-8
oscillator, `one two three` / `{ }` → period-4, `( )` → a bounded blob. The decoded
grids show a **"breathing" parenthesis oscillator** — a localized ~20-cell band
alternating rows of `(` and rows of `)` on a dead `.` background, holding 150+
generations. The seed is *remembered* (the local rule is chaotic, §8), so different
tokens nucleate different bounded patterns — exactly the seed-dependence the causal
rule erases.

**3. But they are lattice-commensurate standing waves, not free life-forms — and
nothing translates.** The decisive test (vary L, hold the seed): a free life-form
keeps the same bounded support and period at *every* L. It does not. The clean
oscillators appear only at commensurate L (`( a )` period-8 at L=64 only; `( )`
period-4 at L=56, period-12 at L=96; `one two three` period-4 at L=64 only); at
other L the same seed gives a bounded-but-aperiodic churning blob, fills, or dies.
And **no seed — including asymmetric ones built for it, with the Brian's-Brain
refractory that makes structure travel — produced a translating (v≠0) structure**:
every detected period had velocity 0. No glider.

**Conclusion.** Seed engineering confirms the narrow hypothesis — the right tokens,
in the chaotic local rule, nucleate bounded persistent localized structure that the
causal rule would erase — but hits the same wall as §14: the LLM's *smooth,
high-entropy* local conditional lacks Conway's knife-edge `B3/S23` balance, so its
localized structures are **commensurate resonances pinned to L, not free objects**,
and a spaceship (which needs the leading edge to advance at exactly the rate the
trailing edge vacates) does not form. **The seed controls *which* resonance, not
*whether* a free life-form exists.** The missing ingredient is still a fine-tuned
local birth/death balance — not a better seed.

## 22. gemma-4-12B: a *multimodal* ground state — the thesis confirmed at 12B

`results/spacetime_causal_gemma-4-12B_*`, `tokens_causal_gemma-4-12B_*`. The
largest and most architecturally exotic model in the study: **google/gemma-4-12B**,
the `gemma4_unified` *any-to-any multimodal* base model (vision + audio + text,
12B, bf16). Loaded via its text-only causal head `Gemma4UnifiedForCausalLM` (666 of
677 weights are the `language_model.*` decoder; needs transformers ≥ the
gemma4_unified release). `--arch causal`, L=64. The §16–20 thesis predicts the
*dynamical repertoire* (freeze / active / no-edge-of-chaos) is invariant to
architecture and scale while only the attractor *content* shifts with training.
gemma-4 confirms both halves, with one genuinely new twist.

- **The ground state is a *multimodal* token.** Dead token (argmax from BOS) =
  **`<image|>`** (id 258882) — not a word. From an empty context, an any-to-any
  model's single most likely token is the image placeholder. This is qualitatively
  new in the ground-state catalogue (GPT-2 `\n`, distilroberta `Advertisements`,
  Pile-family `Q`, Qwen `Human`, Bonsai `:` — all word/structural tokens; gemma's
  is a **modality** token): the unconditional dynamics read out that the model
  "expects" multimodal input when given nothing.
- **It freezes to that ground state — the strongest freeze observed.** Soft
  single-cell T=0: the seed cell is erased by **g1** and the whole lattice is
  `<image|>` thereafter (ρ=0.0002, live=0.0). gemma's ground state is so dominant
  one forward pass wipes any seed — the cleanest instance of §8's causal
  synchronization / §15's frequency filter (the seed is forgotten immediately).
- **Full repertoire, no edge of chaos.** Random init, soft T=0.7: a normal
  active/disordered regime (ρ=0.87, entropy 3.5 bits, **ξ=1** — no spatial
  structure, exactly like every full-attention model). The active genre is a
  web/markup fingerprint — HTML tags (`<h1> <b> </b> <strong>`), edit/user chrome,
  numbers, occasional Chinese — gemma's training mix.

**Conclusion.** The biggest (12B vs the others' ≤1.7B), newest, most exotic
(multimodal vs text) model behaves exactly as §16–20 predict: identical dynamical
repertoire (freeze + active, ξ=1, no Class-4 band), with the attractor set by
training — here a *multimodal* ground state and a web/markup active genre.
Architecture, scale, and modality change the attractor's *content*, not the
dynamics. The fresh note is conceptual: an any-to-any model's unconditional CA
ground state is a *modality* token — a compact readout of what the model expects
from nothing. *Caveat:* single seeds; the text head drops the vision/audio towers,
so this is gemma-4's text backbone, not the full multimodal model.

## 23. Common-noise damage fate map: local window controls coalescence

This session reframed the §8 "causal sync vs masked chaos" result as a
finite-horizon **common-noise damage fate map**. Two replicas differ by one token
at `g=0` and are then driven by the same Gumbel field. The endpoint metrics here
are deliberately modest:

- `p_sync`: fraction of pairs with sustained `H=0` over the final 10 generations.
- `late_damage`: mean Hamming damage over the final 20% of the run. In the raw
  consistency table below it is a count out of 48; in the fate-map tables it is
  normalized to `[0, 1]`.

**Artifacts.** Main scripts and outputs:
`scripts_consistency_sweep.py` →
`results/consistency_*_{sweep,pairs,hamming}.csv` and
`results/endpoint_summary_all_current.csv`; `scripts_null_probe.py` →
`results/null_probe_{aggregate,pairs,hamming}.csv`; `scripts_fate_map.py` →
`results/fate_map_distilroberta_L48_w1-8_T06-14_p8_*` and
`results/fate_map_distilroberta_L48_w2_T05-15_p16_*`;
`scripts_structure_probe.py` →
`results/structure_probe_distilroberta-base_T0_L64_s0_*`;
`scripts_pair_damage_gif.py` → `results/pairdamage_*.gif/.png/.csv` and
`results/pairdamage_contact.png`.

**Raw consistency sweep (`L=48`, `T=1.0`).**

| rule / model | pairs | `p_sync` | late H / 48 | final H / 48 | read |
|---|---:|---:|---:|---:|---|
| causal pythia-160m | 16 | 1.00 | 0.00 | 0.00 | all pairs coalesce |
| causal Qwen3-1.7B-Base | 3 | 1.00 | 0.00 | 0.00 | scale sanity check, all coalesce |
| masked distilroberta-base | 16 | 0.00 | 47.38 | 47.44 | damage saturates |
| masked bert-base-uncased | 16 | 0.44 | 19.88 | 17.31 | mixed / bimodal |
| masked CodeBERTa-small-v1 | 8 | 0.50 | 14.53 | 12.75 | mixed / bimodal |
| local distilroberta `w=1` | 16 | 1.00 | 0.00 | 0.00 | all pairs coalesce |
| local distilroberta `w=2` | 16 | 0.44 | 11.69 | 11.88 | boundary regime |
| local distilroberta `w=4` | 16 | 0.12 | 41.68 | 41.81 | mostly non-coalescing |
| local CodeBERTa `w=2` | 8 | 0.38 | 18.28 | 17.13 | mixed / bimodal |

Temperature spot-checks reinforce the same split but weaken any universal
"masked always chaotic" wording: pythia causal coalesced at `T=0.6` and `T=1.4`;
global masked distilroberta was partial at `T=0.6` (`p_sync=0.25`) and
non-coalescing at `T=1.4` (`p_sync=0`). BERT and CodeBERTa show that "masked" is
not one behavior; pair-level fates are often bimodal.

**Window × temperature map for local distilroberta (`L=48`, 80 steps, 8 pairs).**

The local rule has a clean geometry:

| local window | result across `T=0.6,0.8,1.0,1.2,1.4` |
|---|---|
| `w=1` | mostly coalescing: `p_sync=0.75` at `T=0.6`, then `1.00`; late damage ≈0 |
| `w=2` | boundary: `p_sync=0.12,0.38,0.38,0.62,1.00`; late damage falls `0.55→0.00` |
| `w=3` | mostly non-coalescing: `p_sync=0–0.12`; late damage `0.74–0.99` |
| `w=4,6,8` | mostly non-coalescing: late damage usually `0.81–1.00`, rare healing outliers |

The refined `w=2` scan (`L=48`, 100 steps, 16 pairs) is the most interesting
slice. Increasing temperature **helps coalescence** instead of simply adding
chaos:

| `T` | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 | 1.1 | 1.2 | 1.3 | 1.4 | 1.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `p_sync` | .12 | .19 | .25 | .25 | .31 | .44 | .62 | .69 | .88 | 1.00 | 1.00 |
| `late_damage` | .49 | .52 | .46 | .46 | .36 | .25 | .13 | .10 | .03 | .00 | .00 |

So the map is not a monotone "more temperature = more chaos" story. At `w=2`,
low-to-mid temperature preserves damage; high temperature restores common-noise
coalescence. The sharpest front in the current data is the `w=2→3` wall, not a
temperature-only transition.

**Null probes.** State-independent shared-noise nulls synchronize trivially:
IID uniform fixed-logit and Bernoulli `p=0.5` sync after one step for all 16
pairs; a simple absorbing-local Bernoulli radius-1 null also ends synchronized
(`exact_sync_fraction=1.0`, mean sync step 1.375). This does not prove the trained
model effect is deep, but it does rule out the weakest objection: shared noise
alone does not force persistent damage.

**Structure probe.** `scripts_structure_probe.py` tested local distilroberta
with `T=0`, absorbing local ring, `L=64`, windows `1,2,4`, seed
`" Comments Related Posts"`. No localized traveler survived. The seed filled the
ring at generations 31, 24, and 17 for `w=1,2,4`; tail support was 62–64/64 for
`w=1` and exactly 64/64 for `w=2,4`. The `w=4` tail is an exact period-2 standing
oscillator with shift 0, velocity 0. This reinforces the glider result: the model
can make resonant standing order, but not a free-moving object.

**Animations.** The `pairdamage_*` GIFs are the best human-readable evidence:
`pairdamage_causal_pythia_T1_L48_pair0.gif` blooms to 28/48 then heals to 0;
`pairdamage_masked_distilroberta_T1_L48_pair0.gif` saturates at 48/48;
`pairdamage_local_w1_distilroberta_T1_L48_pair0.gif` heals; `w=2` and `w=4` show
mixed fates, including `w=4` pair 2 blooming to 40/48 before healing. The single
trajectory `spacetime_*_T1.0_L48_s21.gif` files are useful texture references, but
the pair-damage GIFs are the direct coalescence evidence.

**Conclusion.** The central object is now the behavior map: coalescing,
non-coalescing, and mixed finite-horizon basins as a function of architecture,
model, window, temperature, and scale. The glider hunt is secondary. The next
useful work is to sharpen the mixed regimes and determine whether the `w=2`
boundary and the `w=2→3` wall survive larger `L`, longer horizons, and stronger
nulls.

## 24. A *validated* complexity diagnostic — every LLM regime placed on the (h_μ, E) plane

`llm_life/complexity.py`, `scripts_complexity_plane.py`,
`results/complexity_plane.{csv,png}`; `scripts_glider_filter.py`,
`results/glider_filter_localTE.png`. The §1/§20 audit established that the repo's
two structural metrics, `τ_int` and `ξ`, do **not** discriminate Wolfram class
(rule-110 reads `τ_int=0.47`, *below* chaotic rule-30's 1.39; `ξ=1` for 110/30/90
alike). The entire "no edge of chaos" verdict therefore rested on the space-time
**visual** + damage spreading + the §12 locality contrast, never on a scalar that
fires on Class 4. This section closes that gap with the canonical complexity
measures the §20 HANDOFF named as missing, **validates them against the
known-class reference CAs**, and for the first time gives the headline a
*coordinate*.

**The instrument (`llm_life/complexity.py`).**

- **Excess entropy `E`** (Crutchfield–Feldman effective measure complexity): the
  predictive information stored in spatial configuration — the intercept of the
  block-entropy line `H(n) ≈ h_μ·n + E`. High for Class 4 (gliders ⇒ long-range
  order), ≈0 for Class 3 (chaos has no spatial memory) and the dead state.
  Computed on the **live/dead binarized field**, because block entropy is only
  well-sampled on a *small* alphabet — a large token alphabet saturates `H(n)` at
  the sample-count ceiling (for signal *and* null alike) and structure cancels.
  Binarizing to live/dead makes an LLM field commensurate with the binary CAs and
  is exactly the right coarse-graining for the glider question (**gliders are
  live/dead patterns**). Bias-controlled by subtracting an **IID-marginal
  surrogate floor** (`E_excess = E_raw − E_iid`), ≈0 for a structureless field of
  any alphabet size.
- **Entropy rate `h_μ`** (asymptotic block-entropy slope): randomness per site.
  The (`h_μ`, `E`) plane is Langton's complexity-vs-entropy picture.
- **Configurational two-point token MI** `I(d)` (the "token correlation, not the
  change-indicator field" the HANDOFF asked for), bias-corrected; reported as a
  secondary, alphabet-aware structure probe.
- **Local transfer entropy** (Lizier 2008): the per-cell, per-step information
  transported from a neighbour — the repo's first *spatially resolved* diagnostic,
  and a literal **glider filter**.

**Validation (matched random init, `L=120`, post-burn, 2–3 seeds).** The plane
separates the classes exactly where `τ_int`/`ξ` could not:

| rule | class | `h_μ` | `E_excess` |
|---|---|---:|---:|
| 110 | 4 (edge) | 0.65 | **1.36 ± 0.42** |
| 54  | 4 (edge) | 0.62 | **1.23 ± 0.16** |
| 30  | 3 (chaos) | 1.00 | 0.00 |
| 90  | 3 (chaos) | 0.99 | 0.02 |
| 184 | 2 (conserved) | 0.18 | 0.94 |
| 250 | 2 (fill) | 0.00 | 0.00 |
| 0   | 1 (dead) | 0.00 | 0.00 |

`τ_int` put Class-4 rule 110 *below* chaotic rule 30; **excess entropy puts it
~90–370× above** (1.36 vs 0.004–0.015), robustly over seeds. The Class-4 corner
is the only one with *both* substantial randomness (`h_μ`) and high stored
structure (`E`): Class 3 = high `h_μ`, zero `E`; Class 2 = high `E`, near-zero
`h_μ`; Class 1 = origin. And the **local-TE glider filter** traces rule 110's
gliders as coherent diagonal filaments (mean **+0.62 bits**) while rule 30 is
structureless red/blue speckle (**+0.00**) — the qualitative picture the project
always leaned on, now a number (`results/glider_filter_localTE.png`).

**The measurement — every LLM regime on the same validated plane**
(`results/complexity_plane.png`):

| variant | regime | `h_μ` | `E_excess` | live | reading |
|---|---|---:|---:|---:|---|
| causal pythia T0.5 | random-init soft | 0.45 | 0.22 | 0.87 | low structure |
| causal pythia T1.0 | random-init soft | 0.10 | 0.02 | 0.99 | fills → ~origin |
| masked distilR T1.2 | random-init soft | 0.00 | 0.00 | 1.00 | saturated fill → origin |
| local distilR cone r3 | absorbing+refr, seeded | 0.71 | 0.42 | 0.75 | dense, mid |
| **local bert osc pen2** | absorbing+penalty, seeded | 0.30 | **0.70** | 0.16 | **sparse — the closest** |
| local bert osc pen2 r3 | absorbing+penalty+refr | 0.36 | 0.48 | 0.14 | sparse |

**No LLM regime reaches the Class-4 corner** (high `E ≳ 1.2` *and* substantial
`h_μ ≳ 0.4`). The variants split across the other three:

- **Causal full-attention → the structureless / chaos region.** Lower-left:
  `E ≤ 0.22`. At high `T` the lattice fills near-all-live (`live≈0.99`), so its
  disorder lives in *token identity*, not live/dead geometry: token-space `I(1)`
  is the only nonzero signal and there is no activity structure for a glider to be
  made of — the dynamical face of §3/§4/§8's "synchronizes to noise."
- **Masked / bidirectional random-init → the saturated-fill origin.** distilR
  T=1.2 fills to `live=1.00` (`E=0`). Its texture is *entirely* in token identity
  (token `I(1)=0.71`, a dense ordered token-fill) with **zero live/dead
  geometry** — so the §8 "drifting clusters / closest to edge" reading, at matched
  `L=120` random init, is a dense token-pattern, not an activity structure a
  glider could inhabit.
- **Sparse engineered "lifeforms" (bert + local penalty, seeded) → the Class-2
  region.** The §14/§21 breathing oscillators are the only LLM fields with a real
  *dead background* (`live=0.14–0.16`), and they sit at moderate `E` (0.48–0.70)
  and low-moderate `h_μ` (0.30–0.36) — **clustered next to the rule-184 Class-2
  anchor**, at ≈half of rule 110's excess entropy and lower `h_μ`. The validated
  instrument thus *classifies* them: **Class-2 standing order, not Class-4
  gliders** — the quantitative form of §14/§21's "commensurate standing waves, not
  free life-forms." Their local-TE field confirms it: a visible localized band in
  the raw lattice but weak, incoherent transport (mean +0.04 bits) vs rule 110's
  +0.62.

**Conclusion.** The "no edge of chaos" negative — previously carried by visuals +
damage + locality — now holds on a **validated instrument**: across causal,
masked, and local families, **no iterated-LLM cell occupies the Class-4 corner**.
The plane reframes the project's whole arc as a *trajectory*: adding locality and
a local birth/death penalty moves you off the chaos/fill corners toward structure,
but overshoots into **Class-2 frozen/standing order** rather than landing on the
**Class-4 edge** — exactly the §14/§21 verdict that the missing ingredient is
Conway's fine-tuned local balance, not more locality. The masked family's "closest
to edge" structure is, on this instrument, token-identity order on a saturated
background; the sparse bert "lifeforms" are Class-2 standing waves. Rule 110 sits
alone at the edge.

**Finite-size scaling `E(L)` (firm-up).** `scripts_complexity_scaling.py`,
`results/complexity_scaling.{csv,png}`. Two questions the caveats flagged.

*(A) Is the discrimination `L`-stable, or a finite-size confound* — the exact
failure mode that sank `τ_int`/`ξ`? Across `L = 32…256` the excess entropy
separates the classes **at every size**: rule 110/54 (Class 4) stay high
(≈1.2–3.1 bits), rule 30 (Class 3) is **exactly 0.00 at every `L`**, rule 184
(Class 2) ≈0.9–1.4. The Class-4 rules' `E` neither vanishes nor diverges with `L`
— it fluctuates around a stable nonzero value, the signature of a *free* structure
(gliders are `L`-independent objects). The diagnostic is robust where `τ_int`/`ξ`
were not.

*(B) Is the sparse bert "lifeform" a free structure or a commensurate standing
wave?* This is the decisive test, and on the validated instrument it confirms
§14/§21: the bert oscillator's `E(L)` is **strongly `L`-sensitive and
non-monotonic** — `E ≈ 0.04 (L=40) → 0.82 (L=64) → 0.28 (L=96) → 0.98 (L=128)`,
with its live fraction swinging `0.06–0.69` — whereas a *free* life-form would hold
the same `E` and the same bounded support at every `L`. It only "fits" (sparse,
structured) at commensurate sizes and floods or fizzles elsewhere. The causal
active control, by contrast, is **flat and low** (`E ≈ 0.02–0.16` across `L`,
`live ≈ 0.9`) — robustly structureless at every size — and rule 110 is robustly
*high* at every size. The bert structure is neither: its `E` wanders with `L`, the
quantitative fingerprint of a **lattice-commensurate standing wave**, not a free
glider-like object — and even at its best `L` it stays below rule 110's matched-`L`
`E`. This is §14/§21's "commensurate standing waves, not free life-forms,"
now measured on a Class-validated complexity scale.

*Caveats.* (1) The live/dead binarization is the right coarse-graining for the
glider/activity question and makes LLM fields commensurate with binary CAs, but it
discards token-identity structure — dense token-ordered fills (masked T=1.2, the
§23 standing oscillators) correctly read as trivial *activity* geometry while
carrying real token structure (reported via token `I(1)`); a glider, however, is
an activity-geometry object, so this is the relevant axis. (2) `E`/`h_μ` at
`L=120`, post-burn, 2–3 seeds (seeded sparse runs are deterministic, n=1); the
finite-size `E(L)` scaling (above) shows the class discrimination is `L`-stable
(32→256) and the bert structure is `L`-commensurate, though absolute `E` for the
reference Class-4 rules fluctuates seed-to-seed at small `L`. (3) This measures the spatial
complexity of the realized orbit, not a proof of (non-)universality. Refs:
Langton 1990 (*Physica D*); Crutchfield & Feldman 2003 (*Chaos*, "Regenerating
information"); Lizier, Prokopenko & Zomaya 2008 (local information transfer,
arXiv 0809.3275); cond-mat/9409080.

## 25. Reservoir computing on the consistent causal map — echo-state property, weak linear memory, no usable reservoir window

§23 found that the causal full-attention map, iterated under a *shared*
Gumbel-noise realization, is **consistent**: two replicas started from different
states synchronize. Consistency is exactly the **echo-state property (ESP)** — the
state becomes a function of the input history with the initial condition washed
out — which is the defining precondition for *reservoir computing*. This section
asks the obvious follow-up: granted the ESP, **is the causal LLM-CA a useful
reservoir?** We drove it with a scalar input stream and measured its computational
capacity with two calibrated estimators. The answer: it is a *valid* reservoir
whose capacity is **overwhelmingly instantaneous** — a nonlinear function of the
*current* input — with only a **weak linear memory of the recent past**. That
temporal capacity is two to three orders of magnitude below a matched linear
echo-state network, but it is **not zero**: there is robust lag-1 **linear** memory
(`MC_1 ≈ 0.10` at the in-domain peak `T=0.3`, basis-independent and seed-robust,
clearing its floor by a wide margin at both in-domain temperatures). There is **no nonlinear
temporal computation**: the apparent degree-≥2 temporal capacity is finite-sample
bias (the degree-stratified floor below rejects it). So the headline is **"weak
linear lag memory, overwhelmingly instantaneous, no usable reservoir-computing
window"** — *not* "static / memoryless," but also not a temporal computer. (Two
overclaims were caught en route, in opposite directions: an earlier draft said "no
temporal capacity" — an artifact of a ridge-`α` grid railed at its maximum, whose
fix revealed the linear memory — and a later one claimed an *odd degree-3 nonlinear
tail*, which a degree-stratified floor then showed to be bias. The surviving claim
is the linear one.)

**Apparatus** (`llm_life/reservoir.py`, `scripts_reservoir.py`). The lattice is
`L=48` token sites. A scalar input `u(t) ∈ [-1,1]` is binned to a 16-token
codebook and **clamped** at the two left-edge "input" sites each step (causal
attention is leftward, so a left-edge input reaches every reservoir site); the
remaining 46 sites are the reservoir. A **fixed shared-Gumbel noise** sequence
(seeded once, replayed identically) makes the driven map a deterministic function
of `(input, init)` — the §23 coupled-noise machinery reused verbatim. The readout
projects each reservoir site's token embedding onto the top-`K=8` PCA components
of the input-embedding matrix (a `K`-dim generalization of the §1 colour table)
and concatenates → a 368-dim real vector. The clamped input sites are **excluded**
from the readout.

**Estimators and their gates** (`llm_life/capacity.py`).
- **Memory Capacity** (Jaeger): `MC_k =` test-`R²` reconstructing `u(t-k)`;
  `MC = Σ_k MC_k`.
- **Information Processing Capacity** (Dambre): capacity over an orthonormal basis
  of the input history — here the **encoded-symbol Gram-Schmidt basis**
  (orthonormal polynomials of the *binned* input symbol, matched to what the
  reservoir actually sees), summed and split by **degree** and by **instantaneous**
  (all delays 0 — a function of the current input) vs **temporal** (any delay ≥ 1
  — genuine computation over time). Each config is thresholded against a
  **degree-stratified** shuffled-input floor (a by-chance threshold computed
  *per degree*, because higher-degree targets have heavier tails and a single
  global floor under-catches structured high-degree bias).
- Ridge readout, `α` chosen on a **validation** split, capacity on **held-out
  test**, over a wide `α` grid (`1e-6 … 1e8`) with an **interiority** assert. The
  first pass had pinned `α` at the grid maximum for every lag — *under*-regularizing
  and *under*-counting capacity; widening the grid raised `MC` (e.g. `0.23 → 0.41`
  at `T=0.3`) and is now interior for every significant-capacity target.
- The estimators are calibrated on synthetic ground truth (the project's gate
  discipline). A literal `N`-tap shift register returns `MC=N` and IPC degree-1
  `=N`, degree ≥ 2 `=0` (**true negative**). A constructed encoded-symbol nonlinear
  reservoir returns the planted degree-2/3 capacities exactly, total `=` feature
  count, with the correct degree and inst/temporal split and zero off-target
  leakage (**true positive**). Determinism: identical `(init,input,noise)` gives
  byte-identical token grids on MPS, and ESP convergence to *exactly* 0
  corroborates bit-reproducible logits.

**The ESP holds (the gate).** Driven replicas from fully different random inits,
under the same input and same fixed noise, converge to **exactly 0** reservoir
Hamming across the whole swept region `T ∈ [0.3,1.1] × n_in ∈ {1,2,3}`
(`p_sync = 1.00`; initial divergence 32–41 of 46 sites). Contraction is slower at
low `T`, so `MC`/`IPC` are well-defined throughout and low `T` is the most
favourable (least-contractive) operating point.

**Licensed capacity** (causal pythia-160m, `L=48`, `n_in=2`, `K=8`, readout dim
368, 3 input+noise seeds; mean ± std).

| system | MC | IPC total | IPC **instantaneous** | IPC **temporal**‡ |
|---|---:|---:|---:|---:|
| reservoir `T=0.3` (in-domain peak) | 0.41 ± 0.01 | 1.03 | 0.81 | **0.22** |
| reservoir `T=0.7` | 0.25 ± 0.02 | 0.54 | 0.48 | **0.06** |
| reservoir `T=1.0` | 0.23 ± 0.03 | 0.36 | 0.34 | **0.02** |
| reservoir `T=0.0` (out of ESP domain†) | 0.66 ± 0.02 | 2.76 | 2.38 | 0.38 |
| linear ESN (368-dim, ρ=0.95) | 15.8 ± 0.3 | 55.3 | 1.96 | **53.3** |
| literal shift register (30-tap) | 29.9 | 9.0 | 1.0 | **8.0** |
| random-token reservoir | 0.01 | 0.0 | 0.0 | 0.0 |

†`T=0` is the **noiseless argmax** limit, a different regime: the shared-Gumbel
machinery is degenerate and the ESP was only *verified* for `T ∈ [0.3,1.1]`. It is
shown for trend only and is **excluded from the headline**, which uses `T=0.3` as
the in-domain peak.

‡ The reservoir's IPC-temporal is the *measured* total over the degree-agnostic
floor; the degree-stratified floor (below) licenses only its **linear** part
(`≈0.115` at `T=0.3`) — the degree-≥2 remainder is finite-sample bias. The headline
ratios (~460× below ESN, ~7× below instantaneous) use that licensed-linear value.

The contrast is the result. A real reservoir's capacity is **overwhelmingly
temporal** (the ESN: 53.3 of 55.3, almost all of it degree-3). The causal LLM
reservoir's capacity is **overwhelmingly instantaneous**: the *measured* temporal
capacity at the in-domain peak `T=0.3` is 0.22 (pre-degree-floor; falls to 0.02 by
`T=1`), but the degree-stratified floor below licenses only its **linear** part
(`≈0.115`) — the degree-≥2 remainder is bias. On that licensed-linear temporal the
reservoir is **~460× below the ESN** (53.3) and **~7× below its own instantaneous**
capacity (0.81). What temporal capacity there is decomposes (`T=0.3`, 3 seeds,
mean ± std) into `MC_1 = 0.099 ± 0.012` (the Jaeger lag-1 *linear* memory; the IPC
encoded-symbol degree-1 lag-1 in the table below reads `0.115` — the same quantity
in a slightly different estimator) and a small nonlinear remainder; a
**degree-stratified** shuffled-input floor (a separate by-chance threshold per
degree, since higher-degree targets have heavier tails and a single global floor
under-catches structured bias — `scripts_degree_floor.py`) then sorts the
remainder honestly:

| degree | lag-1 temporal capacity (`T=0.3`) | genuine signal? |
|---|---:|---|
| 1 (linear)    | 0.115 ± 0.002 | **yes** — clears its floor by a wide margin at both temps |
| 2 (quadratic) | 0.031 ± 0.020 | no — seed-unstable (one outlier seed) |
| 3 (cubic)     | 0.030 ± 0.004 | **no** — floor-height artifact (see below) |
| 4 (quartic)   | 0.036 ± 0.004 | no — clears at `T=0.3`, **killed at `T=0.7`** (conceded bias) |

The raw temporal profile *rises* with degree (`d2 < d3 < d4`: 0.031, 0.030, 0.036),
the classic signature of high-degree finite-sample bias — a state with `MC_1 ≈ 0.1`
cannot legitimately compute degree-4 temporal functions *better* than degree-2. The
degree-stratified floor correctly kills degree-2 (seed-unstable) and degree-4
(killed at `T=0.7`). Degree-3 appears to "survive," but that is a **floor-height
artifact, not signal**: the surviving cubic (`d3 = 0.030`) and the conceded-bias
quartic (`d4 = 0.036`) are the **same magnitude and equally seed-stable** — d3
clears only because the degree-3 floor sits *below* 0.030 while the degree-4 floor
sits *above* 0.036. The fate is set by floor height, not by the reservoir computing
odd functions. Two further nails: the cubic **fails cross-temperature stability**
(at `T=0.7` only 1 of 3 seeds clears even the agnostic floor), and the raw profile
shows **no odd preference** (`d4 > d3`), so the tanh-ESN odd-symmetry analogy does
not transfer (the LLM-CA has no such mechanism). So we license **no nonlinear
temporal computation**: the only computation over time is the weak lag-1 *linear*
memory, dwarfed (~7×) by instantaneous nonlinear capacity. The feature matrix is
**full rank** (eff-rank 368 = readout dim), so this is not rank starvation — the
input-controllable subspace is simply small, dominated by the shared-noise
variance. Lower `T` helps monotonically.

**Decodable vs dynamical memory — the headline test** (`scripts_capacity_vs_lag.py`,
`results/capacity_vs_lag_pythia160m_L48.{csv,png}`). The §23 contraction sets up a
discrepancy worth resolving head-on: initial-condition memory persists ~`t_sync`
(40–87 steps, the ESP convergence above) yet linear input correlation hits the floor
by lag ~2. Is the input at lags 2…`t_sync` in the state but encoded *nonlinearly*?
We measured single-variable capacity of `P_d(u(t-k))` at every lag `k`, per degree,
against the degree-stratified floor. The answer independently confirms the "no
nonlinear temporal" verdict above: at `T=0.3` (3 seeds) linear, quadratic and cubic
capacity **all decay together**, to their floors by lag ~2 (decodable horizons
degree-1/2/3 = 2/1/2; at `T=0.7`, 1/1/1) — the nonlinear capacity does **not** reach
deeper lags than the linear. The map does not "remember its input nonlinearly
beyond its linear horizon."

But a readout-free probe shows the input is not *gone*, only *scrambled*. Flipping a
single input symbol (same init, same fixed noise) perturbs the reservoir for **~39
steps** before the trajectories reconverge — first *amplifying* to ~34 of 46 sites
over ~4 steps, then slowly contracting (the §4 amplify-then-synchronize signature),
at both `T=0.3` and `T=0.7`. So there are **two memory timescales**: a *dynamical*
(perturbation) horizon ~39 steps ≈ `t_sync`, and a *decodable* horizon ~1–2 steps.
The gap between them is the characterization — the strongly-mixing causal map
**entangles input history into a scrambled, high-dimensional state that is present
for tens of steps but unrecoverable by a degree-≤3 embedding-PCA readout beyond
lag ~2**. It does
not forget the input quickly; it *scrambles* it quickly. (Distance check: the weak
lag-1 decodable memory is *distributed* over the first ~16 reservoir sites next to
the clamp — `first1 = 0.00`, peaking `first16 ≈ 0.12`, diluting to `first46 ≈ 0.09`
— local propagation off the clamp, not a single-site copy.) This **locates** the map
on the memory × nonlinearity plane: shallow decodable memory (depth ~1–2 at every
degree), strong instantaneous nonlinearity, long dynamical persistence, low
decodability — a consistent map that is a poor *practical* reservoir not because it
is frozen or instantly forgetful, but because it mixes inputs into an undecodable
state.

**Input-leak control.** Reading `MC` off **only** the clamped input sites (which
we exclude) gives `≈0.54` at the in-domain `T=0.3` (and falls toward `0.30` by
`T=1`) — **entirely at lag 0** (`MC_1 ≈ 0`). So the trivial "read the input off its
own site" path carries the *current* input but no memory; the weak lag-1 memory the
reservoir does have comes from genuine propagation into the reservoir sites, not
from the clamp. Excluding the input sites is the right call (reservoir `MC` 0.41 <
leak 0.54 at the in-domain `T=0.3`, and the reservoir's `MC_1` > 0 while the leak's
`MC_1 = 0`).

**Cross-architecture: consistency does not trade off against capacity**
(`scripts_crossarch.py`, `results/crossarch_capacity.{csv,png}`). Driving the §23
architecture axis as reservoirs (`T=0.7`, single-seed breadth scan):

| map | ESP (final reservoir Hamming) | temporal IPC |
|---|---|---:|
| causal pythia | **holds** (0.00, `p_sync`=1.0) | ≈0 (at noise edge) |
| local `w=1` distilR | **holds** (0.00, `p_sync`=1.0) | ≈0 |
| local `w=2` distilR | fails (0.63) | — (undefined) |
| local `w=4` distilR | fails (1.00) | — (undefined) |
| global masked distilR | fails (1.00) | — (undefined) |

The two **consistent** maps (causal, local `w=1`) are valid reservoirs with ~0
temporal capacity (this single-seed scan reads ≈0 at `T=0.7`, where even the causal
map's weak temporal signal is at the detection edge — consistent with the locked
3-seed causal value of 0.06 there); the **non-consistent** maps (`w ≥ 2`, global
masked) are not valid reservoirs at all (ESP fails, damage saturates — §23), so a
reservoir capacity is **undefined** for them. There is no "edge" in this family
where a *usable* temporal reservoir lives: the consistency that licenses the
reservoir is the strong contraction that leaves it only weak memory, and abandoning
consistency forfeits the echo-state property without buying usable capacity back.

**Conclusion.** The echo-state property is **necessary but not sufficient**. The
§23 consistency makes the causal LLM-CA a valid reservoir, but the same fast,
complete contraction leaves it dominated by **instantaneous nonlinear processing**,
with only a weak lag-1 *linear* memory of the recent past (`MC_1 ≈ 0.10` in-domain,
~7× below its instantaneous capacity and ~460× below a matched ESN); there is **no
nonlinear temporal computation** — the apparent degree-≥2 temporal capacity is
finite-sample bias that a degree-stratified floor rejects. There is **no usable
reservoir-computing window** anywhere in the architecture family. So as a reservoir
the consistent causal LLM-CA is a **dominant instantaneous nonlinear kernel with
only weak lag-1 linear memory** — consistent, but too contractive to compute over
time. This reframes §23: consistency is a genuine, measurable dynamical property,
but on its own it is not the computational power reservoir computing needs.

**Caveats.** (1) One readout family (top-8 embedding-PCA over the token lattice);
the absolute instantaneous magnitudes would shift under a richer readout, but a
readout cannot manufacture temporal memory the state does not carry, and the
*relative* dominance of instantaneous over temporal is the load-bearing claim.
(2) One small causal model (`pythia-160m`); the cross-arch scan is single-seed
breadth, not a locked magnitude. (3) IPC caps (degree ≤ 4, delay ≤ 8, ≤ 2
variables) bound the search; **no** nonlinear-temporal capacity is claimed — a
**degree-stratified** floor shows the apparent degree-≥2 temporal (including the
degree-3 cubic) to be finite-sample bias, not computation. (4) Capacity is read where the ESP holds (`T ∈
[0.3,1.1]`); the `T=0` argmax limit is out of that domain and excluded from the
headline; under ESP-failure (cross-arch `w ≥ 2`, masked) single-trajectory numbers
are not reservoir capacities and are reported only to show no hidden temporal
signal.

**Artifacts.** `llm_life/reservoir.py`, `llm_life/capacity.py`;
`scripts_reservoir.py` (`apparatus`/`esp`/`mc`/`ipc`), `scripts_capacity_lock.py`
(licensed 3-seed run + α-interiority check), `scripts_capacity_validate.py`
(G3/G4 re-license), `scripts_degree_floor.py` (degree-stratified floor),
`scripts_capacity_vs_lag.py` (capacity-vs-lag-by-degree + perturbation decay),
`scripts_ipc_control.py`, `scripts_reservoir_robust.py`, `scripts_crossarch.py`.
Data: `results/reservoir_{esp,mc,ipc}_*`,
`results/capacity_lock_pythia-160m_L48_nin2_K8.{raw,summary,configs}.csv` + `.png`,
`results/capacity_vs_lag_pythia160m_L48.{csv,png}`,
`results/capacity_{perturbation_decay,lag1_vs_distance}_pythia160m_L48.csv`,
`results/crossarch_capacity.{csv,png}`.

## 11. Next steps

1. **Scale the §23 fate map.** Repeat `w×T` and the refined `w=2` boundary at
   `L=96/192`, longer horizons, and 32+ pairs. Store raw damage fields, not just
   endpoint CSVs.
2. **Harden nulls.** Add logit-shuffle, random-initialized MLM/causal, and
   temperature-matched marginal null models. The current nulls only cover
   state-independent shared-noise baselines.
3. **Measure geometry.** Estimate damage front velocities, avalanche-size
   distributions, configurational token correlations, and mutual information for
   representative fate-map cells. Pair these with GIFs.
4. **Broaden models.** Add a second masked model, larger causal models, and
   entropy-matched temperatures so the old causal/masked split becomes one axis of
   the behavior map rather than the whole story.
5. **Run the non-tautological quantization test.** Quantize a non-ternary model
   such as `Qwen/Qwen3-1.7B-Base` and compare it to its own full-precision
   checkpoint.
6. **Keep gliders as a secondary search.** Focus on sparse BERT-like local regimes
   and L-robust still-lifes first; the current evidence says spaceships are not
   the nearest structure.
