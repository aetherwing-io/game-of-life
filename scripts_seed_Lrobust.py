"""L-robustness test for the seed-nucleated oscillators (task #25).

A true localized life-form should not care how much dead background surrounds it:
fix the seed, vary L, and a real structure keeps the SAME bounded support and period
at every L. If support scales with L or the class flips L-to-L, it is a
lattice-commensurate standing wave (the §14 negative), not a free life-form.
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np
import torch
from llm_life.model import pick_device, load_mlm, dead_token_id_mlm
from llm_life.automaton import LocalMaskedLMAutomaton

DEVICE = pick_device("auto")
W, T, STEPS, PEN = 2, 0.0, 160, 2.0
MODEL = "bert-base-uncased"


def ring_support(live, L):
    idx = np.where(live)[0]
    if idx.size == 0:
        return 0
    if idx.size == 1:
        return 1
    gaps = np.diff(np.concatenate([idx, [idx[0] + L]]))
    return int(L - gaps.max())


def detect_motion(st, L, W, maxp=16):
    for p in range(1, maxp + 1):
        a, b, c, d = st[-1], st[-1 - p], st[-2], st[-2 - p]
        for shift in range(-p * W, p * W + 1):
            if np.array_equal(a, np.roll(b, shift)) and np.array_equal(c, np.roll(d, shift)):
                return p, shift / p
    return 0, 0.0


model, tok, dev = load_mlm(MODEL, DEVICE)
dead = dead_token_id_mlm(model, tok, dev)
auto = LocalMaskedLMAutomaton(model=model, dead_token=dead, mask_token=tok.mask_token_id,
                              cls_token=tok.cls_token_id, sep_token=tok.sep_token_id,
                              window=W, device=dev)
auto.freq_penalty = PEN
g = torch.Generator(device="cpu" if DEVICE == "mps" else DEVICE)
g.manual_seed(0)

CANDS = [("( a )", 3), ("( )", 0), ("one two three", 0)]
LS = [40, 48, 56, 64, 72, 80, 96, 112, 128]

for seed, refr in CANDS:
    print(f"\n#### seed={seed!r} refr={refr}  (bert, w=2, T=0, pen=2.0)")
    print(f"{'L':>5} {'class':>14} {'support':>9} {'final_live':>11} {'period':>7} {'vel':>6}")
    for L in LS:
        state = torch.full((L,), dead, dtype=torch.long, device=DEVICE)
        ids = tok.encode(seed, add_special_tokens=False)[:L]
        live = torch.tensor(ids, dtype=torch.long)
        lo = max(0, (L - live.shape[0]) // 2)
        state[lo:lo + live.shape[0]] = live.to(DEVICE)
        states, _ = auto.trajectory(state, STEPS, T, True, g, refractory=refr)
        st = states.detach().to("cpu").numpy()
        liveb = (st != dead)
        h = len(st) // 2
        sup = float(np.mean([ring_support(liveb[t], L) for t in range(h, len(st))]))
        fl = float(liveb[-1].mean())
        p, v = detect_motion(st, L, W)
        if fl < 0.02:
            cls = "DEAD"
        elif sup > 0.75 * L:
            cls = "FILL"
        elif p == 1 and v == 0:
            cls = "still-life"
        elif p > 0 and abs(v) > 0.05:
            cls = f"SHIP v{v:+.2f}"
        elif p > 1:
            cls = f"OSC p{p}"
        else:
            cls = "loc-aperiodic"
        print(f"{L:>5} {cls:>14} {sup:>9.1f} {fl:>11.3f} {p:>7} {v:>6.2f}")
