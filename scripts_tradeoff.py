"""The CONSISTENCY–CAPACITY tradeoff, in reservoir-computing terms (lead-approved).

A good ESN maxes capacity at the EDGE of the echo-state property (spectral
radius → 1, contraction → marginal). The causal LLM-CA sits DEEP in contraction
(complete coalescence in tens of steps). The question: can the map be tuned toward
its ESP edge (weaker contraction) to gain capacity, or is capacity ≈0 across EVERY
ESP-valid cell?

We sweep contraction strength across the §23 architecture axis and, PER CELL,
measure:
  * ESP-validity + t_sync  (contraction proxy: larger t_sync = weaker contraction,
    closer to the edge). Capacity is reported ONLY where ESP holds — it is
    undefined in non-coalescing cells.
  * MC (over the encoded bin b) and the IPC split C_instantaneous (τ=0-only) vs
    C_temporal (any τ≥1).

Configs: causal pythia across T; local-w1 distilroberta (coalesces, §23) across T;
local-w2 distilroberta (the §23 boundary) across T.

Run: HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts_tradeoff.py --device mps
"""
import argparse, csv, gc, os, time, warnings
warnings.filterwarnings("ignore", message=".*encountered in matmul.*")
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np

from llm_life import capacity as cap
from llm_life import reservoir as rz
from scripts_reservoir import make_reservoir, rz_model_tag
from scripts_crossarch import build_any

# (label, arch, model, window, [temperatures])
CONFIGS = [
    ("causal",   "causal", "EleutherAI/pythia-160m", None, [0.3, 0.5, 0.7, 0.9]),
    ("local w1", "local",  "distilroberta-base",     1,    [0.3, 0.5, 0.7]),
    ("local w2", "local",  "distilroberta-base",     2,    [0.5, 0.7, 0.9, 1.1]),
    ("local w4", "local",  "distilroberta-base",     4,    [0.5, 0.9]),   # w>=3: ESP breaks -> capacity UNDEFINED (marks the edge)
]


def esp_measure(res, L, vocab, dev, n_bins, steps, pairs):
    """Mean final reservoir Hamming (fraction) + t_sync (mean-curve convergence
    step) over `pairs` init-pairs under shared input+noise."""
    nres = len(res.reservoir_sites)
    finals, curves = [], []
    for p in range(pairs):
        u = np.random.default_rng(100 + p).uniform(-1, 1, steps)
        noises = rz.make_noise(steps, L, vocab, dev, 4000 + p)
        sA = res.run(u, res.random_init(10 * p + 1), noises)
        sB = res.run(u, res.random_init(10 * p + 2), noises)
        ham = (sA[:, res.reservoir_sites] != sB[:, res.reservoir_sites]).sum(axis=1)
        finals.append(ham[-10:].mean() / nres); curves.append(ham.astype(float))
    mean_curve = np.vstack(curves).mean(0)
    below = np.where(mean_curve < 0.5)[0]
    t_sync = int(below[0]) if below.size else -1
    return float(np.mean(finals)), t_sync


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--length", type=int, default=48)
    ap.add_argument("--n-in", type=int, default=2, dest="n_in")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--n-bins", type=int, default=16, dest="n_bins")
    ap.add_argument("--esp-steps", type=int, default=160, dest="esp_steps")
    ap.add_argument("--esp-pairs", type=int, default=6, dest="esp_pairs")
    ap.add_argument("--steps", type=int, default=2600)
    ap.add_argument("--kmax", type=int, default=20)
    args = ap.parse_args()

    L = args.length
    wash = 200; rest = args.steps - wash
    tr, va = int(0.6 * rest), int(0.15 * rest); te = rest - tr - va
    sp = cap.make_split(args.steps, wash, tr, va, te)
    basis = cap.symbol_poly_basis(args.n_bins, 4)
    rows = []
    for (label, arch, model_name, window, temps) in CONFIGS:
        try:
            auto, tok, model, dev, vocab, dead, emb = build_any(arch, model_name, window, args.device)
            table = rz.pca_readout_table(emb, args.k)
            res = make_reservoir(auto, L, vocab, emb, dead, args.n_in, args.n_bins,
                                 temps[0], dev, codebook_mode="pc1")
            nres = len(res.reservoir_sites)
            for T in temps:
                t0 = time.time()
                res.temp = T
                esp_h, t_sync = esp_measure(res, L, vocab, dev, args.n_bins, args.esp_steps, args.esp_pairs)
                esp_ok = esp_h < 0.02
                row = {"label": label, "arch": arch, "window": window if window else "",
                       "temp": T, "esp_final_hamming": round(esp_h, 4), "esp_ok": int(esp_ok),
                       "t_sync": t_sync, "washout_gt_tsync": int(wash > (t_sync if t_sync > 0 else 1e9))}
                if esp_ok:
                    u = np.random.default_rng(7).uniform(-1, 1, args.steps)
                    b = rz.bin_input(u, args.n_bins).astype(float); symbols = b.astype(np.int64)
                    noises = rz.make_noise(args.steps, L, vocab, dev, 777)
                    states = res.run(u, res.random_init(123), noises)
                    X = res.features(states, table); Xz, = rz.standardize(X[sp.train], X)
                    mc = cap.memory_capacity(Xz, b, sp, kmax=args.kmax)
                    ipc = cap.information_processing_capacity(Xz, symbols, sp, basis, max_degree=4,
                                                              max_delay=8, max_vars=2, n_surrogate=32, rng_seed=7)
                    s = ipc["summary_max"]
                    row.update({"MC": round(mc["MC"], 4), "MC_kge1": round(float(mc["mc_k"][1:].sum()), 4),
                                "C_inst": round(s["inst_total"], 4), "C_temporal": round(s["temporal_total"], 4),
                                "eff_rank": ipc["eff_rank"]})
                    cap_str = (f"MC={row['MC']:.3f} MC_k≥1={row['MC_kge1']:.3f} "
                               f"C_inst={row['C_inst']:.3f} C_temporal={row['C_temporal']:.3f}")
                else:
                    row.update({"MC": "", "MC_kge1": "", "C_inst": "", "C_temporal": "", "eff_rank": ""})
                    cap_str = "capacity UNDEFINED (ESP fails)"
                rows.append(row)
                print(f"  {label:<9} T={T:<4} ESP_h={esp_h:.3f} t_sync={t_sync:<4} "
                      f"{'ESP' if esp_ok else 'no-ESP':<7} | {cap_str}  ({time.time()-t0:.0f}s)", flush=True)
            del auto, model
            gc.collect()
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception as e:
            print(f"  [skip] {label}: {type(e).__name__}: {e}", flush=True)

    os.makedirs("results", exist_ok=True)
    p = "results/capacity_tradeoff_consistency.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"[wrote] {p}")
    # verdict: is capacity ~0 across ALL ESP-valid cells?
    valid = [r for r in rows if r["esp_ok"] and r["MC"] != ""]
    if valid:
        max_temporal = max(float(r["C_temporal"]) for r in valid)
        max_mc_kge1 = max(float(r["MC_kge1"]) for r in valid)
        print(f"[verdict] across {len(valid)} ESP-valid cells (t_sync {min(r['t_sync'] for r in valid)}–"
              f"{max(r['t_sync'] for r in valid)}): max C_temporal={max_temporal:.3f}, "
              f"max MC_k≥1={max_mc_kge1:.3f} -> "
              f"{'capacity ≈0 at EVERY ESP-valid cell (consistency & capacity mutually exclusive)' if max_temporal < 0.1 else 'some cell has temporal capacity — investigate'}")
    plot(rows, "results/capacity_tradeoff_consistency.png")


def plot(rows, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"causal": "tab:purple", "local w1": "tab:green", "local w2": "tab:orange"}
    valid = [r for r in rows if r["esp_ok"] and r["MC"] != ""]
    invalid = [r for r in rows if not r["esp_ok"]]
    # panel 0: capacity vs contraction (t_sync), ESP-valid only
    for lab in colors:
        pts = [r for r in valid if r["label"] == lab]
        if pts:
            ax0.plot([r["t_sync"] for r in pts], [float(r["C_temporal"]) for r in pts], "o-",
                     color=colors[lab], label=f"{lab} C_temporal")
            ax0.plot([r["t_sync"] for r in pts], [float(r["MC_kge1"]) for r in pts], "s--",
                     color=colors[lab], alpha=0.5, label=f"{lab} MC_k≥1")
    ax0.set_xlabel("t_sync (contraction time — larger = weaker contraction, nearer ESP edge)")
    ax0.set_ylabel("temporal capacity")
    ax0.set_title("Can weaker contraction buy capacity?\n(ESP-valid cells only)")
    ax0.grid(alpha=0.3); ax0.legend(fontsize=7)
    # panel 1: the map — ESP-validity vs t_sync, capacity as size/annotation
    for r in rows:
        c = colors.get(r["label"], "0.5")
        ok = r["esp_ok"]
        x = r["t_sync"] if r["t_sync"] > 0 else 0
        y = float(r["C_temporal"]) if (ok and r["MC"] != "") else -0.02
        ax1.scatter(x, y, s=90, color=c, marker="o" if ok else "x",
                    edgecolor="k" if ok else c, zorder=3)
        ax1.annotate(f"{r['label'][:7]}\nT={r['temp']}", (x, y), fontsize=6,
                     textcoords="offset points", xytext=(5, 3))
    ax1.axhline(0, color="0.7", lw=0.8)
    ax1.set_xlabel("t_sync"); ax1.set_ylabel("C_temporal (×=ESP fails, capacity undefined)")
    ax1.set_title("consistency–capacity map")
    ax1.grid(alpha=0.3)
    fig.suptitle("Consistency–capacity tradeoff: the causal LLM-CA cannot reach its edge of stability")
    fig.tight_layout(); fig.savefig(png, dpi=140); plt.close(fig)
    print(f"[wrote] {png}")


if __name__ == "__main__":
    main()
