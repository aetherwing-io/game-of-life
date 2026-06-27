"""Tiny shared-noise null baselines for paired damage probes.

No models are loaded here. Each null automaton has fixed logits, and paired
replicas are driven by the same Gumbel noise tensor at every generation.

Default run:

    python3 scripts_null_probe.py

Outputs:

* results/null_probe_aggregate.csv
* results/null_probe_pairs.csv
* results/null_probe_hamming.csv
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np

EPS = 1e-20


@dataclass(frozen=True)
class Config:
    name: str
    vocab: int
    logits: np.ndarray
    init_p_live: float | None = None
    absorbing: str = ""
    dead_token: int = 0
    live_token: int = 1


def gumbel(shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    u = rng.random(shape)
    return -np.log(-np.log(u + EPS) + EPS)


def sample_fixed_logits(logits: np.ndarray, length: int, temp: float, noise: np.ndarray) -> np.ndarray:
    lg = np.broadcast_to(logits, (length, logits.shape[0]))
    if temp <= 0.0:
        return lg.argmax(axis=-1).astype(np.int64)
    return (lg / temp + noise).argmax(axis=-1).astype(np.int64)


def local_vacuum_mask(state: np.ndarray, live_token: int) -> np.ndarray:
    left_live = np.roll(state, 1) == live_token
    right_live = np.roll(state, -1) == live_token
    return ~(left_live | right_live)


def step(state: np.ndarray, cfg: Config, temp: float, noise: np.ndarray) -> np.ndarray:
    nxt = sample_fixed_logits(cfg.logits, len(state), temp, noise)
    if cfg.absorbing == "local_r1":
        mask = local_vacuum_mask(state, cfg.live_token)
        nxt = np.where(mask, cfg.dead_token, nxt)
    return nxt


def lyapunov_estimate(hamming: np.ndarray, short_frac: float = 0.1) -> dict:
    h = np.asarray(hamming, dtype=float)
    peak_idx = int(np.argmax(h))
    peak = float(h[peak_idx])
    final = float(h[-1])
    if peak_idx >= 1 and h[0] > 0 and peak > 0:
        lam = float((np.log(peak) - np.log(h[0])) / peak_idx)
    else:
        lam = 0.0
    n_short = max(1, int(h.size * short_frac))
    short_rate = lam if peak_idx <= n_short else float(
        (np.log(max(h[n_short], 1e-9)) - np.log(max(h[0], 1e-9))) / n_short
    )
    return {
        "short_lam": short_rate,
        "peak_hamming": peak,
        "time_to_peak": peak_idx,
        "final_hamming": final,
    }


def sync_step(curve: np.ndarray) -> int | None:
    zeros = np.flatnonzero(curve == 0)
    if zeros.size == 0:
        return None
    return int(zeros[0])


def initial_state(cfg: Config, length: int, rng: np.random.Generator) -> np.ndarray:
    if cfg.init_p_live is None:
        return rng.integers(0, cfg.vocab, size=length, dtype=np.int64)
    return (rng.random(length) < cfg.init_p_live).astype(np.int64)


def one_pair(cfg: Config, pair: int, args: argparse.Namespace) -> tuple[np.ndarray, dict]:
    init_seed = args.init_seed_stride * pair + args.init_seed_offset
    noise_seed = args.noise_seed_stride * pair + args.noise_seed_offset
    init_rng = np.random.default_rng(init_seed)
    noise_rng = np.random.default_rng(noise_seed)

    ref = initial_state(cfg, args.length, init_rng)
    pert = ref.copy()
    site = (args.perturb_stride * pair + args.perturb_offset) % args.length
    pert[site] = (pert[site] + 1) % cfg.vocab

    hamming = [float(np.count_nonzero(ref != pert))]
    for _ in range(args.steps):
        shared = gumbel((args.length, cfg.vocab), noise_rng)
        ref = step(ref, cfg, args.temp, shared)
        pert = step(pert, cfg, args.temp, shared)
        hamming.append(float(np.count_nonzero(ref != pert)))

    curve = np.asarray(hamming, dtype=float)
    est = lyapunov_estimate(curve)
    first_zero = sync_step(curve)
    row = {
        "null": cfg.name,
        "temp": args.temp,
        "L": args.length,
        "steps": args.steps,
        "pair": pair,
        "init_seed": init_seed,
        "noise_seed": noise_seed,
        "perturb_site": site,
        "short_lam": est["short_lam"],
        "peak_hamming": est["peak_hamming"],
        "time_to_peak": est["time_to_peak"],
        "final_hamming": est["final_hamming"],
        "exact_synchronized": int(est["final_hamming"] == 0.0),
        "sync_step": "" if first_zero is None else first_zero,
    }
    return curve, row


def aggregate(cfg: Config, curves: list[np.ndarray], pair_rows: list[dict], args: argparse.Namespace) -> dict:
    H = np.vstack(curves)
    final = H[:, -1]
    peak = H.max(axis=1)
    short = np.asarray([r["short_lam"] for r in pair_rows], dtype=float)
    exact = np.asarray([r["exact_synchronized"] for r in pair_rows], dtype=float)
    sync_steps = [r["sync_step"] for r in pair_rows if r["sync_step"] != ""]
    by_step1 = np.asarray([int(r["sync_step"] == 1) for r in pair_rows], dtype=float)
    verdict = "SYNC" if exact.mean() == 1.0 else ("partial" if final.mean() < 0.5 * args.length else "CHAOTIC")
    return {
        "null": cfg.name,
        "temp": args.temp,
        "L": args.length,
        "steps": args.steps,
        "pairs": args.pairs,
        "vocab": cfg.vocab,
        "init_p_live": "" if cfg.init_p_live is None else cfg.init_p_live,
        "absorbing": cfg.absorbing,
        "final_mean": float(final.mean()),
        "final_std": float(final.std()),
        "peak_mean": float(peak.mean()),
        "short_lam_mean": float(short.mean()),
        "short_lam_std": float(short.std()),
        "exact_sync_fraction": float(exact.mean()),
        "sync_by_step1_fraction": float(by_step1.mean()),
        "mean_sync_step": "" if not sync_steps else float(np.mean(sync_steps)),
        "verdict": verdict,
    }


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[wrote] {path}")


def configs(p_live: float, uniform_vocab: int) -> Iterable[Config]:
    yield Config(
        name=f"iid_uniform_v{uniform_vocab}",
        vocab=uniform_vocab,
        logits=np.zeros(uniform_vocab, dtype=float),
    )
    bern_logits = np.log(np.asarray([1.0 - p_live, p_live], dtype=float))
    yield Config(
        name=f"bernoulli_p{p_live:g}",
        vocab=2,
        logits=bern_logits,
        init_p_live=p_live,
    )
    yield Config(
        name=f"bernoulli_p{p_live:g}_absorbing_local_r1",
        vocab=2,
        logits=bern_logits,
        init_p_live=p_live,
        absorbing="local_r1",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=48)
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--pairs", type=int, default=16)
    parser.add_argument("--p-live", type=float, default=0.5)
    parser.add_argument("--uniform-vocab", type=int, default=16)
    parser.add_argument("--out", default="results")
    parser.add_argument("--prefix", default="null_probe")
    parser.add_argument("--init-seed-stride", type=int, default=5000)
    parser.add_argument("--init-seed-offset", type=int, default=3)
    parser.add_argument("--noise-seed-stride", type=int, default=5000)
    parser.add_argument("--noise-seed-offset", type=int, default=11)
    parser.add_argument("--perturb-stride", type=int, default=7)
    parser.add_argument("--perturb-offset", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    aggregate_rows: list[dict] = []
    pair_rows: list[dict] = []
    hamming_rows: list[dict] = []

    for cfg in configs(args.p_live, args.uniform_vocab):
        curves = []
        cfg_pair_rows = []
        for pair in range(args.pairs):
            curve, row = one_pair(cfg, pair, args)
            curves.append(curve)
            cfg_pair_rows.append(row)
            pair_rows.append(row)
            for generation, h in enumerate(curve):
                hamming_rows.append({
                    "null": cfg.name,
                    "temp": args.temp,
                    "pair": pair,
                    "generation": generation,
                    "hamming": h,
                })
        agg = aggregate(cfg, curves, cfg_pair_rows, args)
        aggregate_rows.append(agg)
        print(
            f"{cfg.name}: final={agg['final_mean']:.3g}+/-{agg['final_std']:.3g} "
            f"peak={agg['peak_mean']:.3g} sync={agg['exact_sync_fraction']:.2f} "
            f"step1={agg['sync_by_step1_fraction']:.2f} {agg['verdict']}",
            flush=True,
        )

    base = os.path.join(args.out, args.prefix)
    write_csv(f"{base}_aggregate.csv", aggregate_rows)
    write_csv(f"{base}_pairs.csv", pair_rows)
    write_csv(f"{base}_hamming.csv", hamming_rows)


if __name__ == "__main__":
    main()
