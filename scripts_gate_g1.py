"""GATE G1 — driven echo-state property at the MC cell, BOTH halves, >=16 pairs.

Resolves §25's "published ESP curve may be n=4" caveat by (a) raising the pair
count to >=16 and (b) showing the TWO halves that together license reservoir
computing, with a hard assert + saved artifact (gate-parity with G2/G3/G4):

  HALF A  "init forgotten" (echo-state / fading memory of initial condition):
          two replicas with DIFFERENT inits but the SAME input + SAME fixed
          noise -> reservoir Hamming -> 0.  This is the ESP itself.

  HALF B  "input perturbs" (the reservoir is genuinely DRIVEN, not a frozen
          constant map that converges regardless): two replicas with the SAME
          init + SAME fixed noise but DIFFERENT input streams -> reservoir
          Hamming stays > 0 and persists.  Without this, half A is vacuous
          (a constant map trivially has ESP and zero capacity).

Cell = the headline MC cell: causal pythia-160m, L=48, n_in=2, T=0.3, K=16.
Run: HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts_gate_g1.py --device mps
"""
import argparse, csv, os, warnings
warnings.filterwarnings("ignore", message=".*encountered in matmul.*")
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np

from llm_life import reservoir as rz
from scripts_reservoir import build_causal, make_reservoir, iid_input

OUT = "results"


def reservoir_hamming(res, sA, sB):
    return (sA[:, res.reservoir_sites] != sB[:, res.reservoir_sites]).sum(axis=1).astype(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--length", type=int, default=48)
    ap.add_argument("--n-in", type=int, default=2, dest="n_in")
    ap.add_argument("--temp", type=float, default=0.3)
    ap.add_argument("--n-bins", type=int, default=16, dest="n_bins")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--pairs", type=int, default=16)
    ap.add_argument("--tail", type=int, default=100)
    ap.add_argument("--washout", type=int, default=250)
    args = ap.parse_args()
    assert args.pairs >= 16, f"G1 requires >=16 pairs (the n=4 caveat); got {args.pairs}"

    auto, tok, model, dev, vocab, dead, emb = build_causal("EleutherAI/pythia-160m", args.device)
    L, T = args.length, args.steps
    res = make_reservoir(auto, L, vocab, emb, dead, args.n_in, args.n_bins, args.temp, dev,
                         codebook_mode="pc1")
    nres = len(res.reservoir_sites)
    print(f"[info] MC cell: causal pythia-160m L={L} n_in={args.n_in} T={args.temp} "
          f"K={args.n_bins} | pairs={args.pairs} steps={T} nres={nres}")

    curvesA, curvesB = [], []
    for p in range(args.pairs):
        # HALF A: init forgotten — different inits, SAME input + SAME noise
        uA = iid_input(T, seed=100 + p)
        noiseA = rz.make_noise(T, L, vocab, dev, seed=4000 + p)
        sA1 = res.run(uA, res.random_init(seed=10 * p + 1), noiseA)
        sA2 = res.run(uA, res.random_init(seed=10 * p + 2), noiseA)
        curvesA.append(reservoir_hamming(res, sA1, sA2))

        # HALF B: input perturbs — SAME init + SAME noise, DIFFERENT input streams
        initB = res.random_init(seed=500 + p)
        noiseB = rz.make_noise(T, L, vocab, dev, seed=6000 + p)
        uX = iid_input(T, seed=200 + p)
        uY = iid_input(T, seed=900 + p)
        sBx = res.run(uX, initB, noiseB)
        sBy = res.run(uY, initB, noiseB)
        curvesB.append(reservoir_hamming(res, sBx, sBy))

    HA = np.vstack(curvesA); HB = np.vstack(curvesB)
    mA, mB = HA.mean(axis=0), HB.mean(axis=0)
    tail = args.tail
    lateA, lateB = float(mA[-tail:].mean()), float(mB[-tail:].mean())
    fracA, fracB = lateA / nres, lateB / nres
    belowA = np.where(mA < 0.5)[0]
    t_sync = int(belowA[0]) if belowA.size else -1
    # per-pair ESP: last `tail` gens exactly 0
    sync_frac = float(np.mean([1.0 if HA[p, -tail:].max() == 0 else 0.0 for p in range(args.pairs)]))

    print(f"\n[HALF A init-forgotten] init_H={mA[0]:.0f} peak={mA.max():.0f} "
          f"late_H={lateA:.3f} (frac={fracA:.4f}) t_sync={t_sync} per-pair_sync={sync_frac:.2f}")
    print(f"[HALF B input-perturbs] init_H={mB[0]:.0f} peak={mB.max():.0f} "
          f"late_H={lateB:.3f} (frac={fracB:.4f})")

    # ---- gate asserts (parity with G2/G3/G4) ----
    checks = []
    def chk(name, ok, detail):
        checks.append({"check": name, "pass": int(bool(ok)), "detail": detail})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    chk("G1_pairs>=16", args.pairs >= 16, f"{args.pairs}")
    chk("G1a_init_forgotten_ESP", fracA < 0.02 and t_sync >= 0 and t_sync < args.washout,
        f"late_frac={fracA:.4f}<0.02, t_sync={t_sync}<{args.washout}")
    chk("G1a_all_pairs_sync", sync_frac >= 0.95, f"per-pair_sync={sync_frac:.2f}>=0.95")
    chk("G1b_input_perturbs_driven", fracB > 0.05, f"late_frac={fracB:.4f}>0.05")
    chk("G1_halves_separated", fracB > 10 * max(fracA, 1e-9),
        f"fracB={fracB:.4f} > 10x fracA={fracA:.4f}")

    os.makedirs(OUT, exist_ok=True)
    # per-generation curves (the >=16-pair published curve, both halves)
    cpath = os.path.join(OUT, "capacity_gate_g1_curves.csv")
    with open(cpath, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["generation", "halfA_init_forgotten_meanH", "halfB_input_perturbs_meanH"])
        for g in range(T):
            w.writerow([g, round(float(mA[g]), 4), round(float(mB[g]), 4)])
    vpath = os.path.join(OUT, "capacity_gate_g1.csv")
    with open(vpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["check", "pass", "detail"]); w.writeheader(); w.writerows(checks)
    print(f"\n[wrote] {cpath}\n[wrote] {vpath}")

    _plot(mA, mB, nres, args, os.path.join(OUT, "capacity_gate_g1.png"))

    failed = [c["check"] for c in checks if not c["pass"]]
    assert not failed, f"G1 FAILED: {failed}"
    print(f"\n[G1 PASS] >=16-pair ESP holds (init forgotten, frac={fracA:.4f}) AND the reservoir "
          f"is genuinely driven (input perturbs, frac={fracB:.4f}) — the n=4 caveat is resolved.")


def _plot(mA, mB, nres, args, png):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    g = np.arange(len(mA))
    ax.semilogy(g, np.clip(mA, 1e-1, None), lw=1.6, color="tab:blue",
                label="A: init forgotten (diff init, same input+noise) → ESP")
    ax.semilogy(g, np.clip(mB, 1e-1, None), lw=1.6, color="tab:red",
                label="B: input perturbs (same init+noise, diff input) → driven")
    ax.axhline(0.02 * nres, ls="--", color="gray", lw=0.8, label="ESP floor (2% of nres)")
    ax.set(xlabel="generation", ylabel="reservoir Hamming (replica A vs B)",
           title=f"Gate G1 — driven ESP at the MC cell ({args.pairs} pairs)\n"
                 f"causal pythia-160m, L={args.length}, n_in={args.n_in}, T={args.temp}")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(png, dpi=140); plt.close(fig)
    print(f"[wrote] {png}")


if __name__ == "__main__":
    main()
