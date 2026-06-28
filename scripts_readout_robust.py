"""Readout & encoding robustness — confounds C3 and C4 from the methodology review.

C4 (input-encoding quantization): the drive is a discrete K-token codebook, so the
headline MC/IPC must not be an artifact of K. We treat the injected bin index b(t)
as THE input and orthonormalize the IPC basis on monomials of b(t-k) under its
empirical distribution (already in llm_life.capacity), then sweep the codebook size
K∈{16,24,32} and report MC + C_inst/C_temporal vs K. A real result is K-stable.

C3 (embedding-PCA readout): a low-K PCA readout could either lose signal (MC too
low) or fit noise (MC climbs without bound). We sweep the number of PCA components
K_pca and report MC vs K_pca — it should SATURATE, not climb. We also compute a
one-hot readout (over the top-M most frequent reservoir tokens) as the maximally
expressive control: one-hot MC should be ≥ embedding-PCA MC (else the embedding
readout is leaking/overfitting).

Headline cell: causal pythia-160m, T=0.3, n_in=2, 2 seeds (breadth, not a re-lock).
Run: HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts_readout_robust.py --device mps
"""
import argparse, csv, os, time, warnings
warnings.filterwarnings("ignore", message=".*encountered in matmul.*")
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np

from llm_life import capacity as cap
from llm_life import reservoir as rz
from scripts_reservoir import build_causal, make_reservoir


def onehot_features(states, reservoir_sites, top_tokens):
    """One-hot each reservoir site over the top-M tokens (rest -> 'other' bin).
    Maximally expressive token-identity readout (no embedding compression)."""
    tok2idx = {t: i for i, t in enumerate(top_tokens)}
    M = len(top_tokens)
    R = states[:, reservoir_sites]                        # (T, nres)
    T, nres = R.shape
    X = np.zeros((T, nres * (M + 1)), dtype=np.float32)
    for j in range(nres):
        col = np.array([tok2idx.get(int(t), M) for t in R[:, j]])
        X[np.arange(T), j * (M + 1) + col] = 1.0
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--temp", type=float, default=0.3)
    ap.add_argument("--length", type=int, default=48)
    ap.add_argument("--n-in", type=int, default=2, dest="n_in")
    ap.add_argument("--steps", type=int, default=2600)
    ap.add_argument("--kmax", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()

    auto, tok, model, dev, vocab, dead, emb = build_causal("EleutherAI/pythia-160m", args.device)
    L = args.length
    wash = 200; rest = args.steps - wash
    tr, va = int(0.6 * rest), int(0.15 * rest); te = rest - tr - va
    sp = cap.make_split(args.steps, wash, tr, va, te)
    seed_pairs = [(7 + 4 * i, 777 + i) for i in range(args.seeds)]
    print(f"[info] L={L} n_in={args.n_in} T={args.temp} steps={args.steps} seeds={args.seeds}")

    # ---------- C4: codebook-size sweep ----------
    print("\n[C4] MC / IPC vs codebook size K (a real result is K-stable):")
    c4_rows = []
    for K in (16, 24, 32):
        table = rz.pca_readout_table(emb, 8)
        basis = cap.symbol_poly_basis(K, 4)
        res = make_reservoir(auto, L, vocab, emb, dead, args.n_in, K, args.temp, dev,
                             codebook_mode="pc1")
        mc_s, inst_s, temp_s = [], [], []
        for (us, ns) in seed_pairs:
            u = np.random.default_rng(us).uniform(-1, 1, args.steps)
            b = rz.bin_input(u, K).astype(float); symbols = b.astype(np.int64)
            noises = rz.make_noise(args.steps, L, vocab, dev, ns)
            states = res.run(u, res.random_init(123), noises)
            X = res.features(states, table); Xz, = rz.standardize(X[sp.train], X)
            mc_s.append(cap.memory_capacity(Xz, b, sp, kmax=args.kmax)["MC"])
            ipc = cap.information_processing_capacity(Xz, symbols, sp, basis, max_degree=4,
                                                      max_delay=8, max_vars=2, n_surrogate=32, rng_seed=us)
            inst_s.append(ipc["summary_max"]["inst_total"]); temp_s.append(ipc["summary_max"]["temporal_total"])
        row = {"K_codebook": K, "MC_mean": float(np.mean(mc_s)), "MC_std": float(np.std(mc_s)),
               "C_inst_mean": float(np.mean(inst_s)), "C_temporal_mean": float(np.mean(temp_s)),
               "log2K_bits": round(np.log2(K), 2)}
        c4_rows.append(row)
        print(f"  K={K:<3} (log2K={row['log2K_bits']} bits)  MC={row['MC_mean']:.3f}±{row['MC_std']:.3f}  "
              f"C_inst={row['C_inst_mean']:.3f}  C_temporal={row['C_temporal_mean']:.3f}", flush=True)

    # ---------- C3: PCA-component sweep + one-hot control (K=16 codebook) ----------
    print("\n[C3] MC vs #PCA readout components (should SATURATE, not climb):")
    c3_rows = []
    Kbins = 16
    res = make_reservoir(auto, L, vocab, emb, dead, args.n_in, Kbins, args.temp, dev, codebook_mode="pc1")
    # cache one trajectory per seed (features differ only by the readout table)
    traj = []
    for (us, ns) in seed_pairs:
        u = np.random.default_rng(us).uniform(-1, 1, args.steps)
        b = rz.bin_input(u, Kbins).astype(float)
        noises = rz.make_noise(args.steps, L, vocab, dev, ns)
        states = res.run(u, res.random_init(123), noises)
        traj.append((states, b))
    for Kpca in (2, 4, 8, 16, 32):
        table = rz.pca_readout_table(emb, Kpca)
        mc_s = []
        for (states, b) in traj:
            X = res.features(states, table); Xz, = rz.standardize(X[sp.train], X)
            mc_s.append(cap.memory_capacity(Xz, b, sp, kmax=args.kmax)["MC"])
        row = {"readout": f"pca{Kpca}", "dim": len(res.reservoir_sites) * Kpca,
               "MC_mean": float(np.mean(mc_s)), "MC_std": float(np.std(mc_s))}
        c3_rows.append(row)
        print(f"  PCA-K={Kpca:<3} (dim={row['dim']})  MC={row['MC_mean']:.3f}±{row['MC_std']:.3f}", flush=True)

    # one-hot control over the top-M reservoir tokens (maximally expressive)
    M = 48
    mc_oh = []
    for (states, b) in traj:
        R = states[:, res.reservoir_sites]
        vals, counts = np.unique(R, return_counts=True)
        top = vals[np.argsort(counts)[::-1][:M]]
        X = onehot_features(states, res.reservoir_sites, top)
        Xz, = rz.standardize(X[sp.train], X)
        mc_oh.append(cap.memory_capacity(Xz, b, sp, kmax=args.kmax)["MC"])
    oh_mc = float(np.mean(mc_oh))
    pca8 = [r for r in c3_rows if r["readout"] == "pca8"][0]["MC_mean"]
    c3_rows.append({"readout": f"onehot_top{M}", "dim": len(res.reservoir_sites) * (M + 1),
                    "MC_mean": oh_mc, "MC_std": float(np.std(mc_oh))})
    sane = oh_mc >= pca8 - 0.02
    print(f"  one-hot(top{M})  MC={oh_mc:.3f}  vs  PCA-8 MC={pca8:.3f}  "
          f"-> one-hot >= PCA-8 : {'PASS' if sane else 'FAIL (embedding readout may leak)'}")

    os.makedirs("results", exist_ok=True)
    with open("results/capacity_readout_robust_C4_codebook.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(c4_rows[0].keys())); w.writeheader(); w.writerows(c4_rows)
    with open("results/capacity_readout_robust_C3_pca.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(c3_rows[0].keys())); w.writeheader(); w.writerows(c3_rows)
    print("\n[wrote] results/capacity_readout_robust_{C4_codebook,C3_pca}.csv")
    # verdicts
    c4_mc = [r["MC_mean"] for r in c4_rows]
    print(f"[C4 verdict] MC across K=16/24/32 = {[round(x,3) for x in c4_mc]} "
          f"(spread {max(c4_mc)-min(c4_mc):.3f}) -> {'K-STABLE (not an encoding artifact)' if max(c4_mc)-min(c4_mc) < 0.1 else 'K-sensitive — investigate'}")
    pca_mc = [r['MC_mean'] for r in c3_rows if r['readout'].startswith('pca')]
    print(f"[C3 verdict] MC vs PCA-K = {[round(x,3) for x in pca_mc]} -> "
          f"{'SATURATES (not fitting noise)' if pca_mc[-1] <= pca_mc[-2] + 0.03 else 'still climbing — possible noise-fitting'}; "
          f"one-hot>=PCA8 {'PASS' if sane else 'FAIL'}")


if __name__ == "__main__":
    main()
