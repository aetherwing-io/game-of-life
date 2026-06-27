"""Matched conditional-damage / consistency sweep.

This is the paper-facing follow-up to ``scripts_lam_sweep.py``:

* matched L across causal, masked, and local-window rules;
* raw per-pair Hamming curves, not only mean/std;
* multiple masked/local models where cached weights allow it;
* local-window interpolation (w) between causal synchronization and
  bidirectional/global masked chaos.

Default run is intentionally modest so it can finish on a laptop:

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts_consistency_sweep.py

Outputs:

* ``results/consistency_sweep_<tag>.csv``       aggregate per config/temp
* ``results/consistency_pairs_<tag>.csv``       per-pair metrics
* ``results/consistency_hamming_<tag>.csv``     per-generation Hamming curves
* ``results/consistency_skipped_<tag>.csv``     configs that could not load
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
from dataclasses import dataclass
from typing import Iterable

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch

from llm_life import metrics
from llm_life.automaton import StepNoise
from llm_life.run import _build, _gen, _model_tag
from llm_life.sampler import gumbel_like


@dataclass(frozen=True)
class Config:
    arch: str
    model: str
    window: int | None = None


PILOT_CONFIGS = [
    Config("causal", "EleutherAI/pythia-160m"),
    Config("masked", "distilroberta-base"),
    Config("masked", "bert-base-uncased"),
    Config("local", "distilroberta-base", 1),
    Config("local", "distilroberta-base", 2),
    Config("local", "distilroberta-base", 4),
]

SCALE_CONFIGS = [
    Config("causal", "Qwen/Qwen3-1.7B-Base"),
]

CODE_CONFIGS = [
    Config("masked", "huggingface/CodeBERTa-small-v1"),
    Config("local", "huggingface/CodeBERTa-small-v1", 2),
]


def parse_temps(raw: str) -> list[float]:
    if ":" not in raw:
        return [float(x) for x in raw.split(",") if x.strip()]
    lo, hi, step = map(float, raw.split(":"))
    vals = []
    x = lo
    while x <= hi + 1e-9:
        vals.append(round(x, 10))
        x += step
    return vals


def config_slug(cfg: Config) -> str:
    if cfg.arch == "local":
        return f"{cfg.arch}_w{cfg.window}_{_model_tag(cfg.model)}"
    return f"{cfg.arch}_{_model_tag(cfg.model)}"


def build_args(cfg: Config, args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        model=cfg.model,
        arch=cfg.arch,
        window=cfg.window if cfg.window is not None else args.window,
        device=args.device,
    )


def random_init(length: int, vocab: int, device: str, seed: int) -> torch.Tensor:
    g = _gen(device, seed)
    return torch.randint(0, vocab, (length,), generator=g).to(device)


def one_pair(auto, info: dict, temp: float, pair: int, args: argparse.Namespace) -> tuple[np.ndarray, dict]:
    device = info["device"]
    vocab = info["vocab"]
    init_seed = args.init_seed_stride * pair + args.init_seed_offset
    noise_seed = args.noise_seed_stride * pair + args.noise_seed_offset
    init = random_init(args.length, vocab, device, init_seed)
    g = _gen(device, noise_seed)

    ref = init.clone()
    pert = init.clone()
    site = (args.perturb_stride * pair + args.perturb_offset) % args.length
    pert[site] = int((int(pert[site].item()) + 1 + pair) % vocab)

    h = [float((ref != pert).sum().item())]
    template = torch.empty(args.length, vocab, dtype=torch.float32, device=device)
    for _ in range(args.steps):
        shared = StepNoise(gumbel=gumbel_like(template, generator=g))
        ref, _ = auto.step(ref, temp, args.absorbing, noise=shared)
        pert, _ = auto.step(pert, temp, args.absorbing, noise=shared)
        h.append(float((ref != pert).sum().item()))

    curve = np.asarray(h, dtype=float)
    est = metrics.lyapunov_estimate(curve)
    pair_row = {
        "pair": pair,
        "init_seed": init_seed,
        "noise_seed": noise_seed,
        "perturb_site": site,
        "short_lam": est["short_time_rate"],
        "peak_hamming": est["peak_hamming"],
        "time_to_peak": est["time_to_peak"],
        "final_hamming": est["final_hamming"],
        "synchronized": int(est["synchronized"]),
    }
    return curve, pair_row


def sweep_config(cfg: Config, temps: Iterable[float], args: argparse.Namespace):
    build_ns = build_args(cfg, args)
    auto, _tok, model, info = _build(build_ns)
    auto.freq_penalty = args.freq_penalty
    slug = config_slug(cfg)
    print(f"\n## {slug}  L={args.length} pairs={args.pairs} steps={args.steps}")
    print(f"[info] dead={info['dead_token_str']} device={info['device']} vocab={info['vocab']}")

    aggregate_rows = []
    pair_rows = []
    hamming_rows = []

    for temp in temps:
        curves = []
        temp_pair_rows = []
        for pair in range(args.pairs):
            curve, row = one_pair(auto, info, temp, pair, args)
            curves.append(curve)
            row.update({
                "arch": cfg.arch,
                "model": cfg.model,
                "model_tag": _model_tag(cfg.model),
                "window": cfg.window if cfg.window is not None else "",
                "temp": temp,
                "L": args.length,
                "steps": args.steps,
                "absorbing": int(args.absorbing),
                "freq_penalty": args.freq_penalty,
            })
            temp_pair_rows.append(row)
            for generation, hamming in enumerate(curve):
                hamming_rows.append({
                    "arch": cfg.arch,
                    "model": cfg.model,
                    "model_tag": _model_tag(cfg.model),
                    "window": cfg.window if cfg.window is not None else "",
                    "temp": temp,
                    "pair": pair,
                    "generation": generation,
                    "hamming": hamming,
                })

        H = np.vstack(curves)
        final = H[:, -1]
        peak = H.max(axis=1)
        short = np.asarray([r["short_lam"] for r in temp_pair_rows], dtype=float)
        sync = np.asarray([r["synchronized"] for r in temp_pair_rows], dtype=float)
        verdict = "SYNC" if final.mean() <= 1.0 else ("partial" if final.mean() < 0.5 * args.length else "CHAOTIC")
        agg = {
            "arch": cfg.arch,
            "model": cfg.model,
            "model_tag": _model_tag(cfg.model),
            "window": cfg.window if cfg.window is not None else "",
            "temp": temp,
            "L": args.length,
            "steps": args.steps,
            "pairs": args.pairs,
            "absorbing": int(args.absorbing),
            "freq_penalty": args.freq_penalty,
            "final_mean": float(final.mean()),
            "final_std": float(final.std()),
            "peak_mean": float(peak.mean()),
            "short_lam_mean": float(short.mean()),
            "short_lam_std": float(short.std()),
            "sync_fraction": float(sync.mean()),
            "verdict": verdict,
        }
        aggregate_rows.append(agg)
        pair_rows.extend(temp_pair_rows)
        print(
            f"  T={temp:<4} final={agg['final_mean']:.1f}+/-{agg['final_std']:.1f} "
            f"peak={agg['peak_mean']:.1f} short_lam={agg['short_lam_mean']:.2f} "
            f"sync={agg['sync_fraction']:.2f} {verdict}",
            flush=True,
        )

    del auto, model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return aggregate_rows, pair_rows, hamming_rows, info


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[wrote] {path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--preset",
        choices=["pilot", "code", "scale", "pilot+code", "pilot+scale", "all"],
        default="pilot",
    )
    p.add_argument("--temps", default="0.6,1.0,1.4", help="comma list or lo:hi:step")
    p.add_argument("--length", type=int, default=48)
    p.add_argument("--steps", type=int, default=80)
    p.add_argument("--pairs", type=int, default=8)
    p.add_argument("--device", default="auto")
    p.add_argument("--window", type=int, default=2, help="fallback window if a config omits one")
    p.add_argument("--freq-penalty", type=float, default=0.0, dest="freq_penalty")
    p.add_argument("--absorbing", action="store_true")
    p.add_argument("--out", default="results")
    p.add_argument("--tag", default="pilot")
    p.add_argument("--strict", action="store_true", help="fail instead of recording skipped load errors")
    p.add_argument("--init-seed-stride", type=int, default=5000)
    p.add_argument("--init-seed-offset", type=int, default=3)
    p.add_argument("--noise-seed-stride", type=int, default=5000)
    p.add_argument("--noise-seed-offset", type=int, default=11)
    p.add_argument("--perturb-stride", type=int, default=7)
    p.add_argument("--perturb-offset", type=int, default=1)
    args = p.parse_args()

    if args.preset == "code":
        configs = list(CODE_CONFIGS)
    elif args.preset == "scale":
        configs = list(SCALE_CONFIGS)
    else:
        configs = list(PILOT_CONFIGS)
    if args.preset in ("pilot+code", "all"):
        configs.extend(CODE_CONFIGS)
    if args.preset in ("pilot+scale", "all"):
        configs.extend(SCALE_CONFIGS)

    temps = parse_temps(args.temps)
    os.makedirs(args.out, exist_ok=True)

    aggregate_rows: list[dict] = []
    pair_rows: list[dict] = []
    hamming_rows: list[dict] = []
    skipped_rows: list[dict] = []
    prefix = os.path.join(args.out, f"consistency_{args.tag}")

    for cfg in configs:
        try:
            agg, pairs, hamming, info = sweep_config(cfg, temps, args)
            aggregate_rows.extend(agg)
            pair_rows.extend(pairs)
            hamming_rows.extend(hamming)
            write_csv(f"{prefix}_sweep.csv", aggregate_rows)
            write_csv(f"{prefix}_pairs.csv", pair_rows)
            write_csv(f"{prefix}_hamming.csv", hamming_rows)
        except Exception as exc:  # model cache misses should not invalidate completed configs
            if args.strict:
                raise
            row = {
                "arch": cfg.arch,
                "model": cfg.model,
                "model_tag": _model_tag(cfg.model),
                "window": cfg.window if cfg.window is not None else "",
                "error": repr(exc),
            }
            skipped_rows.append(row)
            print(f"[skip] {config_slug(cfg)}: {exc!r}", flush=True)
            write_csv(f"{prefix}_skipped.csv", skipped_rows)

    write_csv(f"{prefix}_sweep.csv", aggregate_rows)
    write_csv(f"{prefix}_pairs.csv", pair_rows)
    write_csv(f"{prefix}_hamming.csv", hamming_rows)
    write_csv(f"{prefix}_skipped.csv", skipped_rows)


if __name__ == "__main__":
    main()
