"""Robustness probe: does the near-zero capacity of the causal LLM reservoir
survive (a) the low-noise limit T->0 and (b) stronger drive (more input sites)?
If even the best-case config stays tiny, the 'too consistent to compute'
conclusion is intrinsic, not a weak-drive artifact. One model load; MC only."""
import os, warnings
warnings.filterwarnings("ignore", message=".*encountered in matmul.*")
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np
from llm_life import reservoir as rz
from scripts_reservoir import build_causal, make_reservoir, iid_input

auto, tok, model, dev, vocab, dead, emb = build_causal("EleutherAI/pythia-160m", "mps")
L = 48; T = 2200; K = 8
table = rz.pca_readout_table(emb, K)
u = iid_input(T, seed=7)
noises = rz.make_noise(T, L, vocab, dev, seed=777)
sp = rz.make_split(T, 200, 1200, 300, 500)

def probe(n_in, temp):
    res = make_reservoir(auto, L, vocab, emb, dead, n_in, 16, temp, dev, codebook_mode="pc1")
    res.temp = temp
    states = res.run(u, res.random_init(123), noises)
    X = res.features(states, table)
    Xz, = rz.standardize(X[sp.train], X)
    mc = rz.memory_capacity(Xz, u, sp, kmax=20)
    return mc["MC"], mc["mc_k"][0], mc["mc_k"][1], len(res.reservoir_sites) * K

print("(a) LOW-NOISE LIMIT  (n_in=2):")
for temp in [0.0, 0.1, 0.2, 0.3]:
    MC, m0, m1, rd = probe(2, temp)
    print(f"  T={temp:<4} MC={MC:.3f}  MC_0={m0:.3f} MC_1={m1:.3f}  readout_dim={rd}", flush=True)

print("(b) DRIVE STRENGTH  (T=0.2):")
for n_in in [2, 4, 8, 16]:
    MC, m0, m1, rd = probe(n_in, 0.2)
    print(f"  n_in={n_in:<3} MC={MC:.3f}  MC_0={m0:.3f} MC_1={m1:.3f}  readout_dim={rd}", flush=True)
