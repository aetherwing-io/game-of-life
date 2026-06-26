"""Reference cellular automata — a scientific ground truth to compare against.

Before asking "does an *LLM* sit at the edge of chaos," it helps to render
systems whose class is *known*, with the exact same pipeline (same metrics, same
space-time visualization). Wolfram's elementary CA give us one rule per Wolfram
class:

    rule 110 -> Class 4  (complex, localized travelling structures; Turing-complete)
    rule 30  -> Class 3  (chaotic / used as a PRNG)
    rule 90  -> Class 3  (Sierpinski; linear/additive)
    rule 250 -> Class 2  (simple periodic fill)
    rule 0   -> Class 1  (dies to all-dead)

These produce the same (T+1, L) integer-state arrays the LLM automaton does, so
metrics.py and viz.py work on them unchanged. rule 110's space-time diagram is
the canonical "this is what Class 4 / edge of chaos looks like" picture.
"""

from __future__ import annotations

import numpy as np


def elementary(rule: int, length: int, steps: int, seed: int = 0,
               init: str = "random") -> np.ndarray:
    """Run an elementary (1D, 2-state, 3-neighbour) CA.

    Returns a (steps+1, length) array of {0,1}. ``init`` is "random" or
    "single" (one live cell in the centre — the classic seed for rule 110/90).
    """
    table = np.array([(rule >> k) & 1 for k in range(8)], dtype=np.int64)
    states = np.zeros((steps + 1, length), dtype=np.int64)
    if init == "single":
        states[0, length // 2] = 1
    else:
        rng = np.random.default_rng(seed)
        states[0] = rng.integers(0, 2, size=length)
    for t in range(steps):
        row = states[t]
        left = np.roll(row, 1)
        right = np.roll(row, -1)
        idx = (left << 2) | (row << 1) | right  # 3-bit neighbourhood -> 0..7
        states[t + 1] = table[idx]
    return states


def binary_rgb_table(live=(0.10, 0.12, 0.20), dead=(0.97, 0.97, 0.99)) -> np.ndarray:
    """(2,3) RGB table: dead = near-white, live = dark."""
    return np.array([dead, live], dtype=np.float64)
