"""Robustness probe: does the near-zero capacity of the causal LLM reservoir
survive (a) the low-noise limit T->0 and (b) stronger drive (more input sites)?
If even the best-case config stays tiny, the 'too consistent to compute'
conclusion is intrinsic, not a weak-drive artifact. One model load; MC only."""
import os, warnings
warnings.filterwarnings("ignore", message=".*encountered in matmul.*")
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np
from llm_life import reservoir as rz
from llm_life import capacity as cap   # wide-α, interiority-checked MC (NOT the ≤100 grid)
from scripts_reservoir import build_causal, make_reservoir, iid_input

auto, tok, model, dev, vocab, dead, emb = build_causal("EleutherAI/pythia-160m", "mps")
L = 48; T = 2200; K = 8
table = rz.pca_readout_table(emb, K)
u = iid_input(T, seed=7)
b = rz.bin_input(u, 16).astype(float)   # MC reconstructs the encoded bin (self-consistent w/ IPC)
noises = rz.make_noise(T, L, vocab, dev, seed=777)
sp = cap.make_split(T, 200, 1200, 300, 500)

def probe(n_in, temp):
    res = make_reservoir(auto, L, vocab, emb, dead, n_in, 16, temp, dev, codebook_mode="pc1")
    res.temp = temp
    states = res.run(u, res.random_init(123), noises)
    X = res.features(states, table)
    Xz, = rz.standardize(X[sp.train], X)
    mc = cap.memory_capacity(Xz, b, sp, kmax=20)   # wide-α grid, interiority-asserted
    amax = mc["alpha_at_max_signif"]
    return mc["MC"], mc["mc_k"][0], mc["mc_k"][1], len(res.reservoir_sites) * K, amax

print("(a) LOW-NOISE LIMIT  (n_in=2)  [wide-α MC over bin b]:")
for temp in [0.0, 0.1, 0.2, 0.3]:
    MC, m0, m1, rd, amax = probe(2, temp)
    print(f"  T={temp:<4} MC={MC:.3f}  MC_0={m0:.3f} MC_1={m1:.3f}  readout_dim={rd}  "
          f"α@max_signif={amax}", flush=True)

print("(b) DRIVE STRENGTH  (T=0.2)  [wide-α MC over bin b]:")
for n_in in [2, 4, 8, 16]:
    MC, m0, m1, rd, amax = probe(n_in, 0.2)
    print(f"  n_in={n_in:<3} MC={MC:.3f}  MC_0={m0:.3f} MC_1={m1:.3f}  readout_dim={rd}  "
          f"α@max_signif={amax}", flush=True)
