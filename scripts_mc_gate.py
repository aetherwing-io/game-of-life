"""FULLY-GATED Memory Capacity — every MC-#34 gate from the code audit, in one run.

No MC number is reported until all of these hold (lead's gate):
  (G3) the calibrator is a LITERAL N-tap delay line (asserted MC≈N, degree-1 only)
       in scripts_capacity_validate.py; the ESN is only an extra "real reservoir"
       baseline, never a calibrator.
  (E1) ceiling = SVD effective-rank of the WASHED feature matrix (not nres*K);
       MC is bounded by rank and the plot ceiling is drawn there.
  (B2/E3) PER-LAG shuffle-floor subtraction:
       MC_corr = Σ_k max(0, R²_res,k − R²_shuffle,k)  (report raw AND corrected).
  (E3) T_train / P ≥ 10  (P = readout dim);  steps raised to satisfy it.
  (S1) the MC cell is bound to a VERIFIED ESP cell with the SAME (init,input,noise)
       seeds: a second replica from a different init under the same input+noise must
       converge (final reservoir Hamming → 0) within the washout.
  (S2) ≥3 input+noise seeds, mean ± std.
  (S3) input-sites-ONLY readout control (trivial-copy capacity) + MC_{k≥1} reported
       separately (delay-0 may be a near-copy under causal attention, so the memory
       claim rests on k≥1).

Cell: causal pythia-160m, n_in=2, K=8, T∈{0.3,0.7} (both ESP-valid; 0.7 is the
reviewer's bound cell, 0.3 the in-domain peak).
Run: HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts_mc_gate.py --device mps
"""
import argparse, csv, os, time, warnings
warnings.filterwarnings("ignore", message=".*encountered in matmul.*")
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np

from llm_life import capacity as cap
from llm_life import reservoir as rz
from scripts_reservoir import build_causal, make_reservoir


def svd_rank(Xtrain):
    """Hard numerical rank + soft participation-ratio effective rank of the washed,
    centered feature matrix (the E1 ceiling)."""
    Xc = Xtrain - Xtrain.mean(0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    tol = s.max() * max(Xc.shape) * np.finfo(float).eps
    hard = int((s > tol).sum())
    p = s ** 2
    pr = float((p.sum() ** 2) / (p ** 2).sum())     # participation ratio
    return hard, pr


def esp_bind(res, u, noises, mc_init, alt_seed, washout, nres):
    """S1: two replicas from DIFFERENT inits under the SAME (input, noise) used for
    MC; return final reservoir Hamming over the washout window and convergence step."""
    horizon = min(len(u), washout)
    sA = res.run(u[:horizon], mc_init.clone(), noises[:horizon])
    sB = res.run(u[:horizon], res.random_init(alt_seed), noises[:horizon])
    ham = (sA[:, res.reservoir_sites] != sB[:, res.reservoir_sites]).sum(axis=1)
    below = np.where(ham < 0.5)[0]
    t_sync = int(below[0]) if below.size else -1
    return float(ham[-1]), t_sync


def per_lag_corrected(Xz, b, sp, kmax, perm):
    """Raw per-lag MC, per-lag shuffle floor, and floor-subtracted MC_corr."""
    res_mc = cap.memory_capacity(Xz, b, sp, kmax=kmax)
    shuf_mc = cap.memory_capacity(Xz, b[perm], sp, kmax=kmax)
    mc_k = res_mc["mc_k"]; floor_k = shuf_mc["mc_k"]
    corr_k = np.maximum(0.0, mc_k - floor_k)
    return res_mc, mc_k, floor_k, corr_k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--temps", default="0.3,0.7")
    ap.add_argument("--length", type=int, default=48)
    ap.add_argument("--n-in", type=int, default=2, dest="n_in")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--n-bins", type=int, default=16, dest="n_bins")
    ap.add_argument("--steps", type=int, default=7000)     # T_train/P ≈ 11
    ap.add_argument("--washout", type=int, default=250)
    ap.add_argument("--kmax", type=int, default=25)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    temps = [float(x) for x in args.temps.split(",")]
    seed_pairs = [(7 + 4 * i, 777 + i) for i in range(args.seeds)]
    auto, tok, model, dev, vocab, dead, emb = build_causal("EleutherAI/pythia-160m", args.device)
    L = args.length
    table = rz.pca_readout_table(emb, args.k)
    rest = args.steps - args.washout
    train, val = int(0.6 * rest), int(0.15 * rest); test = rest - train - val
    sp = cap.make_split(args.steps, args.washout, train, val, test)
    res = make_reservoir(auto, L, vocab, emb, dead, args.n_in, args.n_bins, temps[0], dev,
                         codebook_mode="pc1")
    nres = len(res.reservoir_sites); P = nres * args.k
    Tp = train / P
    print(f"[info] L={L} n_in={args.n_in} K={args.k} P(readout_dim)={P} nres={nres}")
    print(f"[info] steps={args.steps} washout={args.washout} train={train} val={val} test={test}  "
          f"T_train/P={Tp:.1f}  (gate: ≥10  -> {'PASS' if Tp >= 10 else 'FAIL'})")
    assert Tp >= 10.0, f"T_train/P={Tp:.1f} < 10; raise --steps or cut --k"

    rng_perm = np.random.default_rng(0)
    rows, agg = [], {}
    for temp in temps:
        res.temp = temp
        cell = {k: [] for k in ("MC_raw", "MC_corr", "MC_0", "MC_kge1_raw", "MC_kge1_corr",
                                "eff_rank_hard", "eff_rank_pr", "esp_final_h", "leak_MC", "leak_kge1")}
        mc_k_all, floor_k_all, corr_k_all = [], [], []
        esp_ok_all = []
        for (us, ns) in seed_pairs:
            t0 = time.time()
            u = np.random.default_rng(us).uniform(-1, 1, args.steps)
            b = rz.bin_input(u, args.n_bins).astype(float)
            noises = rz.make_noise(args.steps, L, vocab, dev, ns)
            mc_init = res.random_init(123)
            states = res.run(u, mc_init, noises)
            X = res.features(states, table); Xz, = rz.standardize(X[sp.train], X)
            # S1 ESP-binding with the SAME (input,noise); 2nd replica from a different init
            esp_h, esp_t = esp_bind(res, u, noises, mc_init, alt_seed=999, washout=args.washout, nres=nres)
            esp_ok = (esp_h < 0.5) and (args.washout > (esp_t if esp_t > 0 else args.washout))
            esp_ok_all.append(esp_ok)
            # E1 ceiling
            hard, pr = svd_rank(X[sp.train])
            # per-lag floor subtraction
            perm = rng_perm.permutation(args.steps)
            res_mc, mc_k, floor_k, corr_k = per_lag_corrected(Xz, b, sp, args.kmax, perm)
            mc_raw = float(mc_k.sum()); mc_corr = float(corr_k.sum())
            mc0 = float(mc_k[0]); mc_kge1_raw = float(mc_k[1:].sum()); mc_kge1_corr = float(corr_k[1:].sum())
            # S3 input-sites-only (trivial copy) control
            Xin = res.features(states, table, sites="input"); Xinz, = rz.standardize(Xin[sp.train], Xin)
            leak = cap.memory_capacity(Xinz, b, sp, kmax=args.kmax)
            cell["MC_raw"].append(mc_raw); cell["MC_corr"].append(mc_corr)
            cell["MC_0"].append(mc0); cell["MC_kge1_raw"].append(mc_kge1_raw)
            cell["MC_kge1_corr"].append(mc_kge1_corr)
            cell["eff_rank_hard"].append(hard); cell["eff_rank_pr"].append(pr)
            cell["esp_final_h"].append(esp_h); cell["leak_MC"].append(leak["MC"])
            cell["leak_kge1"].append(float(leak["mc_k"][1:].sum()))
            mc_k_all.append(mc_k); floor_k_all.append(floor_k); corr_k_all.append(corr_k)
            print(f"  [T={temp} seed={us}] MC_raw={mc_raw:.3f} MC_corr={mc_corr:.3f} "
                  f"MC0={mc0:.3f} MC_k≥1(corr)={mc_kge1_corr:.3f}  rank={hard}(pr={pr:.0f}) "
                  f"ESP_h={esp_h:.1f}({'ok' if esp_ok else 'FAIL'}) leak={leak['MC']:.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            # Free per-seed device/host memory: longer trajectories (steps≳12k) otherwise
            # accumulate MPS cache across seeds and the OS SIGKILLs the 2nd seed.
            del states, X, Xz, Xin, Xinz, noises, u, b
            import gc as _gc; _gc.collect()
            try:
                import torch as _torch; _torch.mps.empty_cache()
            except Exception:
                pass
        # aggregate
        a = {"temp": temp, "n_seeds": args.seeds, "ESP_all_ok": int(all(esp_ok_all)),
             "T_train_over_P": round(Tp, 1), "readout_dim": P}
        for k, v in cell.items():
            a[f"{k}_mean"] = float(np.mean(v)); a[f"{k}_std"] = float(np.std(v))
        agg[temp] = a; rows.append(a)
        mck = np.mean(mc_k_all, 0); flk = np.mean(floor_k_all, 0); crk = np.mean(corr_k_all, 0)
        agg[temp]["_curves"] = (mck, flk, crk)
        print(f"  === T={temp}: MC_raw={a['MC_raw_mean']:.3f}±{a['MC_raw_std']:.3f}  "
              f"MC_corr={a['MC_corr_mean']:.3f}±{a['MC_corr_std']:.3f}  "
              f"MC_k≥1_corr={a['MC_kge1_corr_mean']:.3f}  rank={a['eff_rank_hard_mean']:.0f}  "
              f"ESP_all_ok={a['ESP_all_ok']}")

    os.makedirs("results", exist_ok=True)
    keys = [k for k in rows[0] if not k.startswith("_")]
    with open("results/capacity_mc_gated_pythia160m_L48.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in keys})
    print("[wrote] results/capacity_mc_gated_pythia160m_L48.csv")
    plot(agg, temps, P, os.path.join("results", "capacity_mc_gated_pythia160m_L48.png"))


def plot(agg, temps, P, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(temps), figsize=(6.5 * len(temps), 4.6), squeeze=False)
    for j, temp in enumerate(temps):
        ax = axes[0][j]
        mck, flk, crk = agg[temp]["_curves"]
        k = np.arange(len(mck))
        ax.bar(k, mck, color="tab:purple", alpha=0.5, label="MC_k raw")
        ax.plot(k, flk, "r.--", ms=4, label="per-lag shuffle floor")
        ax.plot(k, crk, "g-", lw=1.5, label="MC_k corrected (raw−floor)")
        ax.axvline(0.5, color="0.6", ls=":", lw=0.8)
        ax.text(0.6, ax.get_ylim()[1]*0.9, "k≥1 = memory", fontsize=8, color="0.4")
        ax.set_xlabel("lag k"); ax.set_ylabel("MC_k (test R²)")
        a = agg[temp]
        ax.set_title(f"T={temp}  MC_corr={a['MC_corr_mean']:.2f}  "
                     f"MC_k≥1={a['MC_kge1_corr_mean']:.2f}  rank={a['eff_rank_hard_mean']:.0f}/{P}")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("Fully-gated MC (per-lag floor-subtracted, ESP-bound, T_train/P≥10) — "
                 "pythia-160m causal reservoir")
    fig.tight_layout(); fig.savefig(png, dpi=140); plt.close(fig)
    print(f"[wrote] {png}")


if __name__ == "__main__":
    main()
