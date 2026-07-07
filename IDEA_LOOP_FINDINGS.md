# Idea-space iteration — findings so far + next steps

> **Status: WIP, single-seed, small-model, CPU. Not a finding yet.** The apparatus
> and its ruler are solid; the numbers below are one seed on `pythia-160m` (a few
> on `pythia-70m`) at `L=48`, `k=16` PCA basis. Read this as "the apparatus works
> and here's the first signal," not as a result. Reproduce with `scripts_idea_loop.py`.

## What this is

The whole repo measured dynamics on the **token lattice** (the model's spoken
output). This track asks whether the same iteration, watched one level down in the
**hidden state** ("idea space"), shows structure the tokens hid. Motivation:
PERSONA_BASIN.md and the transformer-circuits global-workspace read — most of a
model's computation never reaches the tokens.

`scripts_idea_loop.py` has two commands:
- `calibrate` — **the ruler.** Validates the metrics on known-answer controls
  before any trained-model number is trusted (synthetic fixed/cycle/random; a
  planted-attractor true-positive; a random-weight-twin true-negative). Must print
  GREEN first.
- `measure` — the trained-model idea-loop: a per-layer freeze sweep, the pooling
  control, idea-space damage spreading, and a logit-lens decode.

Query mechanism: `output_hidden_states=True` on the existing causal forward pass;
basis = top-`k` PCA of the layer's hidden states (a crude proxy for the paper's
J-lens). Scope: `--arch causal` torch models only.

## Ruler status: GREEN

All seven controls pass on pythia-70m and pythia-160m. Notably the true-negative
was **buggy in the first draft** (it read the twin's representations along the
*trained* model's trajectory → confounded 0.785); fixed to run the twin's own
dynamics → clean **0.020**. The measurement caught the instrument lying — the
repo's standard failure mode, working as intended.

## First signals (pythia-160m, T=0.7, one seed)

**1. Idea space also just settles — no gliders, echo-state holds.** The idea
trajectory has no cycling (`period≈0`) and idea-space damage **coalesces to 0 at
every layer** (peak 0.02–0.27 → 0), mirroring the §23 token-space coalescence
(Hamming 31/48 → 0). The echo-state / consistency property reproduces one level up.
The token-lattice verdict ("settles, no traveling structure") survives into idea
space.

**2. A stable-representation / churning-token dissociation — real but modest.**
At T=0.7 the *words* churn (token_freeze = 0.159, ~84% of sites change per step)
while the *representation* is far stabler. Per-site (un-pooled) freeze vs token
freeze, by depth:

| layer | depth | pooled | **per-site** | token | pooling inflation | **real dissociation** |
|---|---|---:|---:|---:|---:|---:|
| 1  | 0.08 | 0.986 | 0.771 | 0.159 | +0.215 | **+0.61** |
| 4  | 0.33 | 0.973 | 0.573 | 0.159 | +0.400 | +0.41 |
| 8  | 0.67 | 0.885 | 0.502 | 0.159 | +0.383 | +0.34 |
| 11 | 0.92 | 0.879 | 0.366 | 0.159 | +0.512 | +0.21 |

The dissociation **survives the pooling control** (per-site freeze ≫ token freeze
everywhere) — so it is not a central-limit averaging artifact. But **pooling
inflated it by +0.2…+0.5** (growing with depth); the honest number is the per-site
one, and it is smaller than the pooled headline first suggested. Per-site freeze is
"stabler than words," **not "frozen"** (0.77 → 0.37).

**3. Depth trend (answers the ⅓-vs-⅔ question).** The dissociation **decays
monotonically with depth**: shallow layers hold the representation most still; the
**⅔–¾ band is the most dynamic** part of the model (pooled classifier flips
frozen→unstructured there). Deeper = more motion, confirmed un-pooled.

**4. "Seeing" the ideas — stable ≠ meaningful.** The logit-lens decode of the
settled representation returns high-frequency / structural tokens
(`' seriousness' ' refractory' ' correspondence' '\n' '['`), **not a coherent
hidden idea** — the same frequency attractor the token study found (§15). So the
intuition "something stabler sits under the prose" is literally true and now
measured, but the hope that it is a *richer meaning* is **not** supported here:
what's underneath is more stable and *less* legible, the frequency skeleton, not a
hidden mind. (Caveat: a PCA basis gives statistical axes, not verified concepts —
the decode is only suggestive; see next steps.)

## Instrument notes

- **`excess-over-tokens` (per-site freeze − token freeze) is the metric to keep.**
  The `gap` metric (trained − random-twin) collapsed to ≈ trained-freeze once the
  twin control was clean (twin reads ~0 at every layer), so it is deprecated;
  `--twin` gates the slow sweep, off by default.
- The pooled read is confounded (averaging inflation); always report per-site.
- Damage currently uses the pooled projection — one flipped token barely moves a
  48-site mean, so the tiny idea-damage peak is partly a pooling effect too (see
  next steps).

## Caveats

Single seed; `pythia-160m`/`70m` only (12/6 layers — "⅓ depth" is nowhere near the
paper's layer-38 deep-model workspace); CPU (runs time out past ~1 layer with the
twin sweep on); `k=16` PCA basis is a proxy, not the paper's J-lens; per-site
freeze is relative stability, not a fixed point.

## Next steps (prioritized, for the laptop)

1. **Swap the PCA basis for a real J-lens.** This is the highest-value change: PCA
   axes are statistical, so the "see the ideas" decode is currently untrustworthy.
   A Jacobian-to-output basis (autograd; no need for the paper's code) gives
   *verbalizable* directions that decode to real concepts — the difference between
   "math that happens to decode to words" and "the model's actual reportable ideas."
2. **Multi-seed error bars.** Nothing here is a finding at n=1. Re-run the depth
   sweep over ≥5 input/noise seeds; put std on every number before believing the
   depth trend or the dissociation magnitude.
3. **Per-site idea-space damage.** Replace the pooled projection in
   `idea_damage_run` with a per-site read so the damage curve isn't diluted by the
   48-site average — then the echo-state claim in idea space is clean.
4. **Deeper model.** pythia-1.4B / 2.8B (24/32 layers), or a model where ⅓ depth is
   a real place, to test whether the paper's mid-depth "workspace" regime appears
   (it can't in 6–12 layers). Needs MPS/GPU — the CPU path times out.
5. **Option B: the closed idea-loop.** Currently we *observe* the idea space. Feed
   it back: read the top idea at step N, inject it (activation steering / forward
   hook) while generating step N+1. That is the only construction that could show a
   *traveling* idea-structure — the idea-space glider the token lattice forbade.
   Add a planted-traveler true-positive to the ruler before trusting any such claim.
6. **Complexity plane on the idea trajectory.** Discretize the idea-vector and run
   the validated §24 `(h_μ, E)` instrument on it, placing idea-space on the same
   Wolfram-class plane as the token lattice.

## Reproduce

```bash
python scripts_idea_loop.py calibrate --model EleutherAI/pythia-160m   # must be GREEN
python scripts_idea_loop.py measure   --model EleutherAI/pythia-160m --layer 8   # 2/3 depth
# --twin adds the (deprecated) random-weight gap sweep; slow on CPU
```
