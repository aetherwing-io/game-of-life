"""Local transfer-entropy as a glider filter (Lizier et al. 2008), applied to
reference CAs and the most edge-like iterated-LLM fields.

The repo never had a *spatially resolved* diagnostic -- tau_int/xi are single
scalars blind to where structure lives. Local transfer entropy

    t(i,t) = log2 p(x_{t+1} | x_t^k, y_t) / p(x_{t+1} | x_t^k)

is the per-cell, per-step information transported from neighbour y to cell i. On
rule 110 its positive values trace the gliders as coherent diagonal filaments;
on chaotic rule 30 the field is structureless. We render binary (live/dead)
space-time next to the local-TE field for: rule 110 (Class 4), rule 30 (Class 3),
and the most structured LLM cells from the complexity-plane run.

Run AFTER scripts_complexity_plane.py (consumes results/complexity_llm_fields.npz).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from llm_life import reference_ca as rc, complexity as cx

OUT = "results"
WIN = 90          # generations to display
NPZ = os.path.join(OUT, "complexity_llm_fields.npz")


def binary_ref(rule, L=120, steps=400, burn=200, seed=0):
    st = rc.elementary(rule, L, steps, seed=seed, init="random")[burn:]
    return st  # already {0,1}


def load_llm_fields():
    """Return {label: (binary_field, dead_token)} for a couple of edge-like cells."""
    out = {}
    if not os.path.exists(NPZ):
        print(f"[warn] {NPZ} not found; LLM panels skipped")
        return out
    d = np.load(NPZ)
    # the sparse dead-background structure (bert local-penalty oscillator) is the
    # one LLM field where live/dead is meaningful and a glider, if present, would
    # show; the masked fill is shown as the trivial contrast.
    want = {"local_bert_osc_pen2": "bert local oscillator (penalty) — sparse",
            "masked_distilR_T1.2": "masked distilR T1.2 — dense fill"}
    for key, label in want.items():
        if key in d.files:
            f = d[key]
            dead = int(np.bincount(f.reshape(-1)).argmax())  # modal = dead/background
            out[label] = (cx.binarize(f, dead), dead)
    return out


def panel(ax_raw, ax_te, field, title):
    """field is binary {0,1}, (T+1, L)."""
    w = min(WIN, field.shape[0])
    sub = field[:w]
    ax_raw.imshow(sub, cmap="binary", aspect="auto", interpolation="nearest")
    ax_raw.set_title(title + "  —  live/dead", fontsize=9)
    ax_raw.set_xlabel("position"); ax_raw.set_ylabel("generation")
    # local TE (right neighbour), positive = info transported
    te, mean = cx.local_transfer_entropy(field, "right", 1)
    te_sub = te[:w - 2]
    vmax = max(np.percentile(np.abs(te_sub), 99), 1e-3)
    im = ax_te.imshow(te_sub, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                      aspect="auto", interpolation="nearest")
    ax_te.set_title(f"local transfer entropy  (mean={mean:+.2f} bits)", fontsize=9)
    ax_te.set_xlabel("position")
    return im


def main():
    systems = [("rule 110  (Class 4 — gliders)", binary_ref(110)),
               ("rule 30  (Class 3 — chaos)", binary_ref(30))]
    systems += [(lbl, fld) for lbl, (fld, _) in load_llm_fields().items()]

    n = len(systems)
    fig, axes = plt.subplots(n, 2, figsize=(11, 2.7 * n))
    if n == 1:
        axes = axes[None, :]
    last_im = None
    for i, (title, field) in enumerate(systems):
        last_im = panel(axes[i, 0], axes[i, 1], field, title)
    fig.suptitle("Local transfer entropy as a glider filter — reference CAs vs iterated LLMs\n"
                 "gliders appear as coherent diagonal filaments (red); chaos/frozen = structureless",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    png = os.path.join(OUT, "glider_filter_localTE.png")
    fig.savefig(png, dpi=140)
    print(f"[wrote] {png}")


if __name__ == "__main__":
    main()
