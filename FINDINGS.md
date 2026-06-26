# Findings: GPT-2 iterated as a cellular automaton

Experimental results for the question *"if you loop LLM inference — feed a
token sequence in, take the output, feed it back — what kind of dynamical
system do you get, and does it sit at the edge of chaos?"*

**Model:** GPT-2 (124M, base). **Lattice:** `L=128` token sites. **Seeds:** 3
unless noted. **Hardware:** CPU. All raw numbers in `results/sweep_raw_*.csv`
and `results/damage_*_L128*.csv`; figures in `results/`.

## TL;DR

**GPT-2 iterated as a cellular automaton has no Class-4 / edge-of-chaos regime
under temperature control.** It does one of two things:

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

## 7. Next steps

1. **Masked-LM map** — true local symmetric neighborhood; the most faithful
   Conway analogue and the experiment most likely to (dis)prove an edge regime.
2. **Finite-size scaling** of the absorbing transition (vary `L`) to test
   whether `T_c ≈ 1.3` is a true critical point or a finite-size crossover.
3. **Larger base models** — does synchronization-under-common-noise persist
   with scale, or do bigger models develop a genuinely chaotic (non-syncing)
   regime?
4. **Conditional-Lyapunov sweep** — map the short-time rate and synchronization
   time vs. temperature to locate where (if ever) asymptotic synchronization
   breaks down.
