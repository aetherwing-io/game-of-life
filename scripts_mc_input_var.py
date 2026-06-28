"""Decision check (lead): use the ENCODED bin index b(t) as the input variable for
MC, so degree-1-IPC == MC EXACTLY and the study is self-consistent. Confirms:
  * MC over b(t-k)  ==  degree-1 IPC of phi_1(b(t-k))  (per lag, should match)
  * MC over u (continuous) vs MC over b  (the quantization gap, expected ~tiny)
Headline cell T=0.3, 3 seeds; reuses the same trajectory/split as the lock run.
"""
import os, warnings
warnings.filterwarnings("ignore", message=".*encountered in matmul.*")
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np
from llm_life import capacity as cap
from llm_life import reservoir as rz
from scripts_reservoir import build_causal, make_reservoir

T, NBINS, K, NIN, KMAX, TEMP = 3000, 16, 8, 2, 20, 0.3
SEEDS = [(7, 777), (11, 778), (15, 779)]


def main():
    auto, tok, model, dev, vocab, dead, emb = build_causal("EleutherAI/pythia-160m", "mps")
    L = 48
    table = rz.pca_readout_table(emb, K)
    basis = cap.symbol_poly_basis(NBINS, 3)
    sp = cap.make_split(T, 200, 1800, 400, 600)
    res = make_reservoir(auto, L, vocab, emb, dead, NIN, NBINS, TEMP, dev, codebook_mode="pc1")

    mc_u_tot, mc_b_tot, max_abs_diff = [], [], []
    for (us, ns) in SEEDS:
        u = np.random.default_rng(us).uniform(-1, 1, T)
        b = rz.bin_input(u, NBINS).astype(float)
        symbols = b.astype(np.int64)
        noises = rz.make_noise(T, L, vocab, dev, ns)
        states = res.run(u, res.random_init(123), noises)
        X = res.features(states, table); Xz, = rz.standardize(X[sp.train], X)
        rp = cap.RidgePath(Xz, sp)

        mc_u = cap.memory_capacity(Xz, u, sp, kmax=KMAX)        # continuous u target
        mc_b = cap.memory_capacity(Xz, b, sp, kmax=KMAX)        # encoded bin target
        # degree-1 IPC per lag = capacity of phi_1(b(t-k)); affine in b -> must equal MC_b_k
        ipc_d1 = []
        for k in range(KMAX + 1):
            sh = np.zeros(T, dtype=np.int64); sh[k:] = symbols[: T - k]
            y = basis[sh, 1].copy(); y[:k] = 0.0
            ipc_d1.append(rp.capacity(y)["capacity"])
        ipc_d1 = np.array(ipc_d1)
        diff = np.abs(mc_b["mc_k"] - ipc_d1)
        mc_u_tot.append(mc_u["MC"]); mc_b_tot.append(mc_b["MC"]); max_abs_diff.append(float(diff.max()))
        print(f"[seed {us}] MC_u={mc_u['MC']:.4f}  MC_b={mc_b['MC']:.4f}  "
              f"max|MC_b_k - IPCd1_k|={diff.max():.2e}  "
              f"(MC_b_0={mc_b['mc_k'][0]:.3f}==IPCd1_0={ipc_d1[0]:.3f})", flush=True)

    mu_u, mu_b = np.mean(mc_u_tot), np.mean(mc_b_tot)
    print(f"\nMC over continuous u : {mu_u:.4f} ± {np.std(mc_u_tot):.4f}")
    print(f"MC over encoded bin b: {mu_b:.4f} ± {np.std(mc_b_tot):.4f}")
    print(f"relative gap |u-b|/u : {abs(mu_u-mu_b)/mu_u*100:.2f}%")
    print(f"MC_b == degree1-IPC : max abs per-lag diff over seeds = {max(max_abs_diff):.2e} "
          f"({'EXACT (self-consistent)' if max(max_abs_diff) < 1e-9 else 'not exact'})")


if __name__ == "__main__":
    main()
