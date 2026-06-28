"""G2 cross-process determinism: print a sha256 of a fixed short driven trajectory.
Run this in TWO separate python processes and compare the HASH lines — identical ⇒
F(state,input,noise) is reproducible across process boundaries (not just in-process).
Routed here by the reviewer to avoid contending for MPS with their own runs.

  python scripts_g2_xproc.py --device mps   # run twice, diff the HASH lines
"""
import argparse, hashlib, os, warnings
warnings.filterwarnings("ignore", message=".*encountered in matmul.*")
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np
from llm_life import reservoir as rz
from scripts_reservoir import build_causal, make_reservoir, iid_input

ap = argparse.ArgumentParser(); ap.add_argument("--device", default="mps"); a = ap.parse_args()
auto, tok, model, dev, vocab, dead, emb = build_causal("EleutherAI/pythia-160m", a.device)
L = 48
res = make_reservoir(auto, L, vocab, emb, dead, 2, 16, 0.7, dev, codebook_mode="pc1")
u = iid_input(120, seed=7)
noises = rz.make_noise(120, L, vocab, dev, seed=777)
states = res.run(u, res.random_init(123), noises)
print("HASH", hashlib.sha256(np.ascontiguousarray(states).tobytes()).hexdigest(), "shape", states.shape)
