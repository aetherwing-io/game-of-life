"""Seed-engineering life-form hunt in the local windowed-ring rule (task: seeds).

Hypothesis (from §8): the local/masked rule is chaotic (perturbations persist), so
UNLIKE the causal rule (which synchronizes/erases any seed, §15), it *remembers its
seed*. So an engineered seed -- a local fixed point: self-predictive inside,
predicts-dead at its boundary -- might nucleate a BOUNDED, PERSISTENT, L-robust
life-form (still-life / oscillator / spaceship) where random/single-cell seeds fizzle.

A life-form = bounded support (localized), persistent (doesn't die), not filled, and
ideally periodic (oscillator) or period-up-to-translation (spaceship, v != 0).

Run: python scripts_seed_hunt.py   (writes results/seed_hunt.csv + grids for hits)
"""
import csv
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch

from llm_life.model import pick_device

DEVICE = pick_device("auto")
W, T, L, STEPS = 2, 0.0, 64, 160           # window radius, temp, lattice, generations
ABSORB = True

# ---------- ring-aware structure metrics ----------
def ring_support(live, L):
    """Length of the contiguous live arc (L - largest dead gap). Small => localized."""
    idx = np.where(live)[0]
    if idx.size == 0:
        return 0
    if idx.size == 1:
        return 1
    gaps = np.diff(np.concatenate([idx, [idx[0] + L]]))
    return int(L - gaps.max())


def detect_motion(st, L, W, maxp=16):
    """Smallest period p s.t. config repeats up to a ring-shift, plus velocity v=shift/p.
    v==0, p==1 -> still-life; v==0, p>1 -> oscillator; v!=0 -> SPACESHIP (|v|<=W)."""
    for p in range(1, maxp + 1):
        a, b, c, d = st[-1], st[-1 - p], st[-2], st[-2 - p]
        for shift in range(-p * W, p * W + 1):
            if np.array_equal(a, np.roll(b, shift)) and np.array_equal(c, np.roll(d, shift)):
                return p, shift / p
    return 0, 0.0


def classify(st, dead, L, W):
    live = (st != dead)
    dens = live.mean(axis=1)
    h = len(st) // 2
    supports = np.array([ring_support(live[t], L) for t in range(len(st))])
    act = (st[1:] != st[:-1]).mean(axis=1)
    final_live = float(dens[-1])
    mean_live = float(dens[h:].mean())
    mean_sup = float(supports[h:].mean())
    sup_growth = float(supports[h:].mean() - supports[1:h].mean())  # >0 => spreading
    mean_act = float(act[h:].mean())
    p, v = detect_motion(st, L, W)
    localized = (0 < mean_sup < 0.6 * L) and final_live > 0.02 and abs(sup_growth) < 0.15 * L
    if final_live < 0.02:
        cls = "DEAD"
    elif mean_sup > 0.75 * L or mean_live > 0.85:
        cls = "FILL"
    elif localized and p == 1 and v == 0 and mean_act < 0.01:
        cls = "STILL-LIFE"
    elif localized and p > 0 and abs(v) > 0.05:
        cls = f"SPACESHIP p{p} v{v:+.2f}"
    elif localized and p > 1 and v == 0:
        cls = f"OSC p{p}"
    elif localized:
        cls = "localized-aperiodic"
    else:
        cls = "unbounded/aperiodic"
    score = (final_live > 0.02) * (1.0 - mean_sup / L) * (mean_live < 0.85)  # localized+alive
    return dict(cls=cls, final_live=round(final_live, 3), mean_live=round(mean_live, 3),
                mean_support=round(mean_sup, 1), sup_growth=round(sup_growth, 1),
                mean_act=round(mean_act, 3), period=p, velocity=round(v, 3),
                score=round(float(score), 3))


# ---------- model + seed ----------
def build_local(name):
    from llm_life.model import load_mlm, dead_token_id_mlm
    from llm_life.automaton import LocalMaskedLMAutomaton
    model, tok, dev = load_mlm(name, DEVICE)
    dead = dead_token_id_mlm(model, tok, dev)
    auto = LocalMaskedLMAutomaton(model=model, dead_token=dead, mask_token=tok.mask_token_id,
                                  cls_token=tok.cls_token_id, sep_token=tok.sep_token_id,
                                  window=W, device=dev)
    return auto, tok, dead


def attractor_seed(auto, tok, dead, k=6):
    """The model's own most-self-consistent cluster: top-k distinct non-dead tokens
    predicted from a bare [MASK] (the seed family most likely to be self-predictive)."""
    ids = [tok.cls_token_id, tok.mask_token_id, tok.sep_token_id]
    with torch.no_grad():
        logits = auto.model(torch.tensor([ids], device=DEVICE)).logits[0, 1]
    order = torch.argsort(logits, descending=True).tolist()
    toks = []
    for i in order:
        if i != dead and i not in (tok.cls_token_id, tok.sep_token_id, tok.mask_token_id,
                                   tok.pad_token_id):
            toks.append(i)
        if len(toks) >= k:
            break
    return tok.decode(toks)


def make_init(tok, dead, seed_text, L):
    state = torch.full((L,), dead, dtype=torch.long, device=DEVICE)
    ids = tok.encode(seed_text, add_special_tokens=False)[:L]
    if not ids:
        return state, ""
    live = torch.tensor(ids, dtype=torch.long)
    lo = max(0, (L - live.shape[0]) // 2)
    state[lo:lo + live.shape[0]] = live.to(DEVICE)
    return state, tok.decode(ids)


COMMON = [
    "( )", "( ) ( )", "[ ]", "{ }", "( a )",        # self-delimiting / brackets
    "a b a b", "0 1 0 1", "yes no",                  # periodic motifs (oscillator)
    "a a b", "1 2 3", "x y z",                       # asymmetric (spaceship)
    "the quick brown fox", "hello world",            # neutral control (expect fizzle)
]
EXTRA = {
    "huggingface/CodeBERTa-small-v1": ["def f ( ) :", "return x", "[ 0 , 1 ]", "( a , b )"],
    "bert-base-uncased": [". . .", "and the", "i am", "one two three"],
    "distilroberta-base": [" Comments Related Posts", " Reply Cancel Save", " Next Previous"],
}
MODELS = ["huggingface/CodeBERTa-small-v1", "bert-base-uncased", "distilroberta-base"]
CONFIGS = [(0.0, 0), (2.0, 0), (2.0, 3)]  # (freq_penalty, refractory)

g = torch.Generator(device="cpu" if DEVICE == "mps" else DEVICE)
g.manual_seed(0)

rows, hits = [], []
for name in MODELS:
    auto, tok, dead = build_local(name)
    seeds = COMMON + EXTRA.get(name, [])
    try:
        seeds = seeds + [attractor_seed(auto, tok, dead)]
    except Exception:
        pass
    print(f"\n#### {name}  dead={dead!r}->{tok.decode([dead])!r}  (w={W}, T={T}, L={L}, absorbing)")
    for pen, refr in CONFIGS:
        auto.freq_penalty = pen
        results = []
        for seed in seeds:
            init, decoded = make_init(tok, dead, seed, L)
            states, _ = auto.trajectory(init, STEPS, T, ABSORB, g, refractory=refr)
            st = states.detach().to("cpu").numpy()
            m = classify(st, dead, L, W)
            m.update(model=name.rsplit("/", 1)[-1], pen=pen, refr=refr, seed=seed)
            results.append((m, st))
            rows.append({k: m[k] for k in ("model", "pen", "refr", "seed", "cls", "final_live",
                                           "mean_live", "mean_support", "mean_act", "period",
                                           "velocity", "score")})
            if m["cls"].startswith(("STILL", "OSC", "SPACESHIP", "localized")):
                hits.append((m, st, tok, dead))
        results.sort(key=lambda r: -r[0]["score"])
        print(f"  -- pen={pen} refr={refr} --  (top 4 by localized-alive score)")
        for m, _ in results[:4]:
            print(f"     {m['cls']:<20} score={m['score']:.2f} live={m['final_live']:.2f} "
                  f"support={m['mean_support']:.0f}/{L} act={m['mean_act']:.2f}  seed={m['seed']!r}")

# ---------- record + dump grids for life-form candidates ----------
os.makedirs("results", exist_ok=True)
with open("results/seed_hunt.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"\n[wrote] results/seed_hunt.csv  ({len(rows)} runs, {len(hits)} localized candidates)")

# dump decoded grids for the strongest localized candidates (esp. any spaceship/oscillator)
hits.sort(key=lambda h: (not h[0]["cls"].startswith("SPACE"), not h[0]["cls"].startswith("OSC"),
                         -h[0]["score"]))
for m, st, tok, dead in hits[:8]:
    tag = f"{m['model']}_pen{m['pen']}_r{m['refr']}_{m['cls'].split()[0]}".replace("/", "_")
    tag = "".join(c if c.isalnum() or c in "._-" else "_" for c in tag)[:60]
    path = f"results/seedhunt_{tag}.txt"
    with open(path, "w") as f:
        f.write(f"# {m}\n# dead {dead}={tok.decode([dead])!r} shown as '.'\n")
        for t in range(st.shape[0]):
            cells = ["." if int(x) == dead else (tok.decode([int(x)]).strip() or "_") for x in st[t]]
            f.write(f"g{t:>3} | " + " ".join(c[:6] for c in cells) + "\n")
    print(f"[grid] {m['cls']:<18} {m['seed']!r} -> {path}")
print("\nDONE")
