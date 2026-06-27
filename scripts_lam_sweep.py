"""λ_cond synchronization-transition sweep (FINDINGS §8 firm-up, task #24).

Maps the asymptotic coupled-noise damage outcome (final Hamming separation) vs
temperature for causal full-attention vs masked bidirectional at MATCHED L, plus a
causal scale point. The question: does a causal model ever fail to synchronize
(final separation > 0) as T rises, or is causal universally consistent while only
bidirectional coupling breaks consistency?

Soft rule (no absorbing state) so synchronization is non-trivial (not drain-to-dead).
Run: python scripts_lam_sweep.py   (writes results/lam_sweep_T.csv)
"""
import csv
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch

from llm_life import metrics
from llm_life.automaton import StepNoise
from llm_life.sampler import gumbel_like
from llm_life.model import pick_device

DEVICE = pick_device("auto")
L, PAIRS, STEPS = 48, 10, 100


def gen(seed):
    g = torch.Generator(device="cpu" if DEVICE == "mps" else DEVICE)
    g.manual_seed(seed)
    return g


def build_causal(name):
    from llm_life.model import load, dead_token_id
    from llm_life.automaton import LLMAutomaton
    model, tok, dev = load(name, DEVICE)
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    dead = dead_token_id(model, tok, bos, dev)
    return LLMAutomaton(model, dead_token=dead, bos_token=bos, device=dev), model.config.vocab_size


def build_masked(name):
    from llm_life.model import load_mlm, dead_token_id_mlm
    from llm_life.automaton import MaskedLMAutomaton
    model, tok, dev = load_mlm(name, DEVICE)
    dead = dead_token_id_mlm(model, tok, dev)
    auto = MaskedLMAutomaton(model=model, dead_token=dead, mask_token=tok.mask_token_id,
                             cls_token=tok.cls_token_id, sep_token=tok.sep_token_id, device=dev)
    return auto, model.config.vocab_size


def damage(auto, vocab, T):
    finals, lams = [], []
    for p in range(PAIRS):
        g = gen(5000 * p + 11)
        init = torch.randint(0, vocab, (L,), generator=g).to(DEVICE)
        ref, pert = init.clone(), init.clone()
        site = (7 * p + 1) % L
        pert[site] = int((int(pert[site].item()) + 1 + p) % vocab)
        h = [float((ref != pert).sum().item())]
        template = torch.empty(L, vocab, dtype=torch.float32, device=DEVICE)
        for _ in range(STEPS):
            shared = StepNoise(gumbel=gumbel_like(template, generator=g))
            ref, _ = auto.step(ref, T, False, noise=shared)
            pert, _ = auto.step(pert, T, False, noise=shared)
            h.append(float((ref != pert).sum().item()))
        h = np.array(h)
        finals.append(h[-1])
        lams.append(metrics.lyapunov_estimate(h)["short_time_rate"])
    return float(np.mean(finals)), float(np.std(finals)), float(np.mean(lams))


TEMPS = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
CONFIGS = [
    ("causal", "EleutherAI/pythia-160m", TEMPS),
    ("masked", "distilroberta-base", TEMPS),
    ("causal", "Qwen/Qwen3-1.7B-Base", [0.8, 1.0, 1.2]),  # scale point
]

rows = []
for arch, name, ts in CONFIGS:
    auto, vocab = (build_causal if arch == "causal" else build_masked)(name)
    print(f"\n## {arch} {name}  (L={L}, {PAIRS} pairs, {STEPS} steps)")
    print(f"{'T':>5} {'final_sep':>10} {'std':>6} {'short_lam':>9}")
    for T in ts:
        fm, fs, lm = damage(auto, vocab, T)
        verdict = "SYNC" if fm <= 1.0 else ("partial" if fm < 0.5 * L else "CHAOTIC")
        print(f"{T:>5} {fm:>10.1f} {fs:>6.1f} {lm:>9.3f}   {verdict}", flush=True)
        rows.append({"arch": arch, "model": name.rsplit("/", 1)[-1], "temp": T,
                     "final_sep": round(fm, 2), "final_std": round(fs, 2),
                     "short_lam": round(lm, 4), "L": L, "verdict": verdict})

os.makedirs("results", exist_ok=True)
with open("results/lam_sweep_T.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("\n[wrote] results/lam_sweep_T.csv")
