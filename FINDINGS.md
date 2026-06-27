# Findings: GPT-2 iterated as a cellular automaton

Experimental results for the question *"if you loop LLM inference — feed a
token sequence in, take the output, feed it back — what kind of dynamical
system do you get, and does it sit at the edge of chaos?"*

**Model:** GPT-2 (124M, base). **Lattice:** `L=128` token sites. **Seeds:** 3
unless noted. **Hardware:** CPU. All raw numbers in `results/sweep_raw_*.csv`
and `results/damage_*_L128*.csv`; figures in `results/`.

## TL;DR

**Full-attention LLMs iterated as cellular automata have no Class-4 /
edge-of-chaos regime — not under temperature, not under a global balance knob,
not with a bidirectional rule.** The missing ingredient is *locality*: full
attention mixes information globally every step, so no localized structure can
persist (`ξ` pinned at its floor everywhere; sections 1–10).

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

And a subtler result from damage spreading: the high-activity "chaos" is
**not deterministic chaos**. Under shared noise the system *synchronizes* —
perturbations amplify briefly then heal to zero. The apparent disorder is
faithful transcription of the injected temperature noise, not sensitive
dependence on initial conditions.

## 1. The instruments work (reference baseline)

The same metrics + space-time renderer run on elementary CAs of known Wolfram
class correctly separate them (`results/ref_rule*.png`):

| rule | class | activity ρ | entropy (bits) | τ_int |
|---|---|---|---|---|
| 110 | 4 (edge of chaos) | 0.27 | 0.89 | **28.9** |
| 30 | 3 (chaotic) | 0.50 | 0.99 | 1.25 |
| 90 | 3 (Sierpinski) | 0.19 | 0.41 | 9.55 |
| 250 | 2 (periodic) | 0.00 | 0.00 | 3.09 |

The Class-4 rule shows the highest autocorrelation time (critical slowing
down); chaotic rule 30 decorrelates instantly at near-max entropy. So when the
same instruments report "no edge of chaos" for GPT-2, that is a meaningful
negative, not an instrument failure.

## 2. Absorbing variant: a real absorbing-state transition (`T_c ≈ 1.3`)

`results/sweep_phasediagram_absorbing.png`. With the "no spontaneous birth from
vacuum" rule, the all-`\n` state is a true absorbing state. Activity, entropy,
and spatial correlation length are **identically zero for T ≤ 1.2**, then switch
on sharply at `T = 1.3–1.4` — a clean directed-percolation-style transition.
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

`τ_int` is **highest at the lowest temperature and decays monotonically to ~0**
— the opposite of an edge-of-chaos peak. `ξ` sits at its floor of 1 across the
whole active range. There is no critical point and no Class-4 band.

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
(**common-noise-induced synchronization**; negative *asymptotic* conditional
Lyapunov exponent).

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

## 8. Masked-LM (bidirectional) variant

`results/sweep_phasediagram_masked_soft.png`, `spacetime_masked_T*.png`,
`damage_masked_*.png`. distilroberta-base, `L=48`, symmetric neighborhood (each
site recomputed from the rest via one batched masked forward pass). Dead token:
`Advertisements` (the model's bare-`[MASK]` argmax).

Two qualitative differences from causal:

- **Low T: collapses harder.** Even in *soft* mode (no absorbing rule) it drains
  to the dead state for `T < 1.2`, where causal-soft stayed active at `ρ ≈ 0.65`.
  Bidirectional context is a stronger consensus: every site sees the whole
  (increasingly uniform) sequence and agrees on the ground state. The dead→chaos
  transition is sharper and higher (`T_c ≈ 1.2`) than causal's.
- **High T: genuinely chaotic.** Damage spreading does **not** synchronize —
  at `T = 1.0` a one-token perturbation fills the entire lattice (48/48) and
  stays, vs causal's uniform healing to 0. Positive asymptotic conditional
  Lyapunov exponent: real sensitive dependence, not noise-driven pseudo-chaos.

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
collapses). The recency-biased SSMs keep emitting a whitespace token at the
frontier, so the front advances; GPT-2's absorbing runs (§2) instead drain. This
is a degenerate, content-free echo of the §12 windowed-ring propagation — reached
via architecture rather than an explicit mask — and it leaves pure vacuum behind,
not a structure.

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
to a coherent still-life too, so the fill-vs-freeze split tracks **capability/scale,
not attention topology**; a matched full-precision control is the clean test. As in
§12/§13/§15, the stable structure is a training-data fingerprint: it is built from
the model's own most mutually-predictive tokens.

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

## 17. Ternary (2-bit-trained) weights: extreme quantization is dynamically transparent

`results/spacetime_causal_Ternary-Bonsai-1.7B-unpacked_*`,
`tokens_causal_Ternary-Bonsai-1.7B-unpacked_*`,
`damage_causal_Ternary-Bonsai-1.7B-unpacked_*`. prism-ml's **Ternary-Bonsai-1.7B**
— a Qwen3-1.7B-architecture model trained to *ternary* weights {−1, 0, +1}, here
in the "unpacked" checkpoint (ternary values materialised to standard tensors) so
it loads as a plain `Qwen3ForCausalLM`. Question: does crushing the weights to
ternary change the iterated-map *dynamics*, or only the attractor's token content?
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

1. **Ternary quantization is dynamically transparent.** A ternary-weight model
   shows the same phases (vacuum collapse, fill-to-still-life, common-noise
   synchronization) with the same quantitative signatures as full-precision causal
   models. Whatever determines the iterated map's behaviour is robust to crushing
   the weights to {−1, 0, +1}; only the *genre* of the attractor tokens (a function
   of the training mix) changes, not the dynamics.
2. **Fill-vs-freeze is not about attention topology.** GPT-2 (124M, full attention)
   *freezes* to whitespace under soft T=0 (§3); RWKV-169m (recurrent) **and**
   Bonsai-1.7B (full attention) both *fill* to a coherent-sentence still-life. So
   the soft-T=0 fill is not a recurrence effect — it tracks model
   capability/scale. (§16 tentatively attributed it to recurrence; this corrects
   that.) The clean isolation is the pending full-precision Qwen3-1.7B control
   (does a same-arch, same-scale, *non*-ternary model fill identically?) plus a
   larger GPT-2.

*Caveats:* single seeds; the "unpacked" checkpoint stores ternary values in fp16,
so this measures the *trained-ternary network's* dynamics, not behaviour under
live 2-bit packed kernels — the native mlx-2bit variant is the test of whether the
runtime packing itself perturbs anything (§18 — it doesn't).

## 18. Native 2-bit packing is dynamically faithful — the checkpoint, not the precision, sets the attractor

`results/spacetime_mlx_*`, `tokens_mlx_*`, `damage_mlx_*`. A new MLX backend
(`--arch mlx`, via `mlx_lm`) iterates genuinely low-bit checkpoints on Metal: the
forward runs in MLX and the logits are bridged back to torch at the automaton
boundary, so the map / Gumbel sampling / coupled noise / metrics are reused
unchanged. We compare prism-ml's Ternary-Bonsai in its **native 2-bit MLX
packing** against the same model **unpacked to fp16** (§17), at two sizes.

**Result 1 — 2-bit packing reproduces fp16 dynamics exactly.** 1.7B-mlx-2bit vs
1.7B-unpacked, identical settings:

| run | 1.7B-mlx-2bit | 1.7B-unpacked (§17) |
|---|---|---|
| abs single T=0 | ρ=0.0143, H=0.3005, τ=10.58, ξ=2 → vacuum | ρ=0.0143, H=0.3005, τ=10.58, ξ=2 → vacuum |
| soft single T=0 | ρ=0.206, H=4.417, τ=14.7, ξ=10 | ρ=0.207, H=4.417, τ=14.5, ξ=10 |
| soft fill content | *identical Chinese sentence* | *1990年…国际法体系* |
| damage soft T=0.6 | λ_short=**+0.747**, heals to 0/96 | λ_short=**+0.745**, heals to 0/96 |

The two are the same down to fp-noise, in the ordered (still-life) *and* chaotic
(synchronization) regimes. So §17's "extreme quantization is dynamically
transparent" now holds at the level of the actual packed 2-bit kernels, not just
materialised ternary weights — the runtime quantization perturbs nothing
measurable. (It is, after all, the same trained network; this confirms the MLX
path and the packing introduce no dynamical artifact.)

**Result 2 — the checkpoint, not the bits, drives the attractor.** The 8B-mlx-2bit
soft single-cell run *also* fills (live 0.99) but into a **degenerate near-uniform
field of `0`** (ρ=0.04, entropy **0.34 bits**, ξ=2) — not a coherent sentence.
Same 2-bit packing as the faithful 1.7B, so this is the *checkpoint*: the 8B's
basin is the code/zeros field §13 found in instruct/agent-weighted models, where
the 1.7B base falls into a Chinese-factual sentence. (Whether the 8B Bonsai is
instruct-tuned or merely code-weighted is unverified; the basin shape matches
§13's instruct models.) Dead tokens differ too: 1.7B `':'`, 8B `' '` (space).

**Conclusion.** Across full-precision (GPT-2, Qwen), recurrent (RWKV, Mamba),
ternary-in-fp16 (§17), and native 2-bit (here), the iterated-map *dynamics* —
the phases, the metric signatures, the common-noise synchronization — are robust
to architecture and to numeric precision. What changes is the attractor's
*content*: the genre of its tokens and the richness of its fixed point, which
track the training mix and post-training, not the bit-width. For this whole study
quantization is a red herring; the checkpoint's training is everything.

*Caveats:* single seeds; the MLX bridge runs the torch side on CPU (fine — the
forward dominates); the 8B was characterised on the single-cell runs only.

## 11. Next steps

1. **Local-neighborhood rule (the headline follow-up).** Restrict each site to
   a window of ±w neighbors — via an explicit attention mask on a full-attention
   model, a sliding-window-attention model (e.g. Mistral SWA), or a
   state-space/recurrent model (Mamba). Finite signal speed is the precondition
   for gliders; this is the experiment most likely to finally produce one.
   *(Windowed-ring done — see §12. State-space/recurrent done — see §16: RWKV-4
   and Mamba give finite* effective *range (a propagating absorbing front; a
   coherent-sentence still-life under the soft rule) but no localized lifeform,
   because a causal SSM ingests its whole left context in one synchronous pass —
   only the explicit ±w window of §12 imposes true per-generation locality.)*
2. **Local frequency penalty** — make the balance knob *local* (penalize by
   neighborhood composition, not whole-grid counts), the natural pairing with #1.
   *(Done — see §14: opens a sparse regime, but the (L × window × penalty) map
   found no L-robust localized structure; the sparse band is a lattice-pinned
   knife-edge between vacuum and fill, not a lifeform basin.)*
3. **Finite-size scaling** of the absorbing transition (vary `L`) to test
   whether `T_c ≈ 1.3` is a true critical point or a finite-size crossover.
4. **Larger / local-attention base models** — does common-noise synchronization
   (causal) vs. true chaos (masked) track architecture or scale?
5. **Conditional-Lyapunov sweep** — map short-time rate and synchronization time
   vs. temperature to locate where asymptotic synchronization breaks down.
