"""GATED IPC at T_train/P ≥ 10 (reviewer E3 bar) — locks the degree decomposition.

The lock IPC ran at T_train/P≈4.9; only MC was re-gated at 11×. The contested
degree≥2 temporal is exactly what insufficient data inflates (the C3 breadth scan
showed readout capacity climbing with dimension at low T_train/P). This re-derives
the IPC degree × inst/temporal split at steps=7000 (T_train/P=11), 3 seeds, with
the degree-stratified shuffled-input floor — to confirm the degree≥2 temporal
SHRINKS at proper data, strengthening "weak LINEAR temporal only."

Reports, per temp, mean±std over seeds: total IPC, C_inst, C_temporal, and the
per-degree temporal (d1..d4) next to the degree-matched floor.

Run: HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts_ipc_gate.py --device mps
"""
import argparse, csv, os, time, warnings
warnings.filterwarnings("ignore", message=".*encountered in matmul.*")
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np

from llm_life import capacity as cap
from llm_life import reservoir as rz
from scripts_reservoir import build_causal, make_reservoir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--temps", default="0.3,0.7")
    ap.add_argument("--length", type=int, default=48)
    ap.add_argument("--n-in", type=int, default=2, dest="n_in")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--n-bins", type=int, default=16, dest="n_bins")
    ap.add_argument("--steps", type=int, default=7000)
    ap.add_argument("--washout", type=int, default=250)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-surrogate", type=int, default=40, dest="n_surrogate")
    args = ap.parse_args()

    temps = [float(x) for x in args.temps.split(",")]
    seed_pairs = [(7 + 4 * i, 777 + i) for i in range(args.seeds)]
    auto, tok, model, dev, vocab, dead, emb = build_causal("EleutherAI/pythia-160m", args.device)
    L = args.length
    table = rz.pca_readout_table(emb, args.k)
    basis = cap.symbol_poly_basis(args.n_bins, 4)
    rest = args.steps - args.washout
    train, val = int(0.6 * rest), int(0.15 * rest); test = rest - train - val
    sp = cap.make_split(args.steps, args.washout, train, val, test)
    res = make_reservoir(auto, L, vocab, emb, dead, args.n_in, args.n_bins, temps[0], dev,
                         codebook_mode="pc1")
    P = len(res.reservoir_sites) * args.k
    Tp = train / P
    print(f"[info] P={P} train={train} T_train/P={Tp:.1f} (E3 bar ≥10 -> {'PASS' if Tp>=10 else 'FAIL'})")
    assert Tp >= 10.0

    rows = []
    for temp in temps:
        res.temp = temp
        acc = {k: [] for k in ("total", "inst", "temporal", "d1t", "d2t", "d3t", "d4t",
                               "floor1", "floor2", "floor3", "floor4", "a_at_max")}
        for (us, ns) in seed_pairs:
            t0 = time.time()
            u = np.random.default_rng(us).uniform(-1, 1, args.steps)
            symbols = rz.bin_input(u, args.n_bins)
            noises = rz.make_noise(args.steps, L, vocab, dev, ns)
            states = res.run(u, res.random_init(123), noises)
            X = res.features(states, table); Xz, = rz.standardize(X[sp.train], X)
            ipc = cap.information_processing_capacity(Xz, symbols, sp, basis, max_degree=4,
                                                      max_delay=8, max_vars=2,
                                                      n_surrogate=args.n_surrogate, rng_seed=us)
            s = ipc["summary_max"]; pc = s["per_cell"]; thr = ipc["thresholds_by_degree"]
            acc["total"].append(s["total"]); acc["inst"].append(s["inst_total"])
            acc["temporal"].append(s["temporal_total"]); acc["a_at_max"].append(ipc["alpha_at_max_signif"])
            for d in (1, 2, 3, 4):
                acc[f"d{d}t"].append(pc.get((d, "temporal"), 0.0))
                acc[f"floor{d}"].append(thr[d]["max"])
            print(f"  [T={temp} seed={us}] total={s['total']:.3f} inst={s['inst_total']:.3f} "
                  f"temp={s['temporal_total']:.3f} d1t={pc.get((1,'temporal'),0):.3f} "
                  f"d3t={pc.get((3,'temporal'),0):.3f} a@max={ipc['alpha_at_max_signif']} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        row = {"temp": temp, "n_seeds": args.seeds, "T_train_over_P": round(Tp, 1)}
        for k, v in acc.items():
            row[f"{k}_mean"] = round(float(np.mean(v)), 4); row[f"{k}_std"] = round(float(np.std(v, ddof=1) if len(v) > 1 else 0), 4)
        rows.append(row)
        print(f"  === T={temp}: total={row['total_mean']:.3f}±{row['total_std']:.3f}  "
              f"inst={row['inst_mean']:.3f}  temporal={row['temporal_mean']:.3f}±{row['temporal_std']:.3f}")
        print(f"      d1t(LIN)={row['d1t_mean']:.3f}±{row['d1t_std']:.3f} (floor {row['floor1_mean']:.4f}) | "
              f"d2t={row['d2t_mean']:.3f}±{row['d2t_std']:.3f} | d3t={row['d3t_mean']:.3f}±{row['d3t_std']:.3f} "
              f"(floor {row['floor3_mean']:.4f}) | d4t={row['d4t_mean']:.3f}±{row['d4t_std']:.3f}", flush=True)

    os.makedirs("results", exist_ok=True)
    with open("results/capacity_ipc_gated_pythia160m_L48.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("[wrote] results/capacity_ipc_gated_pythia160m_L48.csv")
    # headline: did degree≥2 temporal shrink vs the lock (4.9x)?
    for r in rows:
        nl_t = r["d2t_mean"] + r["d3t_mean"] + r["d4t_mean"]
        print(f"[T={r['temp']}] degree≥2 temporal = {nl_t:.3f} (lock@4.9x was ~0.10 at T=0.3); "
              f"degree-1 temporal (LINEAR) = {r['d1t_mean']:.3f}")


if __name__ == "__main__":
    main()
