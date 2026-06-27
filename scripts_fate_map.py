"""Damage-fate map for local-window masked LLM cellular maps.

The map answers: for a local masked rule, which regions coalesce under shared
noise and which preserve damage?

Axes:
    x = local window radius w
    y = temperature T

Outputs:
    results/fate_map_<tag>_aggregate.csv
    results/fate_map_<tag>_pairs.csv
    results/fate_map_<tag>_hamming.csv
    results/fate_map_<tag>_psync.png
    results/fate_map_<tag>_late_damage.png
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
from types import SimpleNamespace

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch

from llm_life.automaton import StepNoise
from llm_life.metrics import lyapunov_estimate
from llm_life.run import _build, _gen, _model_tag, _random_init
from llm_life.sampler import gumbel_like


def parse_list(raw: str, cast=float):
    return [cast(x) for x in raw.split(",") if x.strip()]


def build_local(model: str, window: int, device: str):
    args = SimpleNamespace(model=model, arch="local", window=window, device=device)
    return _build(args)


def one_pair(auto, info, temp: float, pair: int, args):
    device = info["device"]
    vocab = info["vocab"]
    init_seed = args.init_seed_offset + args.init_seed_stride * pair
    noise_seed = args.noise_seed_offset + args.noise_seed_stride * pair
    init = _random_init(auto, args.length, vocab, device, init_seed)
    ref = init.clone()
    pert = init.clone()
    site = (args.perturb_offset + args.perturb_stride * pair) % args.length
    pert[site] = int((int(pert[site].item()) + 1 + pair) % vocab)

    g = _gen(device, noise_seed)
    template = torch.empty(args.length, vocab, dtype=torch.float32, device=device)
    h = [float((ref != pert).sum().item())]
    for _ in range(args.steps):
        shared = StepNoise(gumbel=gumbel_like(template, generator=g))
        ref, _ = auto.step(ref, temp, args.absorbing, noise=shared)
        pert, _ = auto.step(pert, temp, args.absorbing, noise=shared)
        h.append(float((ref != pert).sum().item()))

    h = np.asarray(h, dtype=float)
    est = lyapunov_estimate(h)
    last_k = h[-args.sustain_k :]
    nlate = max(1, int(np.ceil(args.late_frac * h.size)))
    sustained_zero = bool(np.all(last_k == 0))
    sync_time = ""
    for i in range(h.size):
        if np.all(h[i:] == 0):
            sync_time = i
            break
    return h, {
        "pair": pair,
        "init_seed": init_seed,
        "noise_seed": noise_seed,
        "perturb_site": site,
        "final_hamming": float(h[-1]),
        "peak_hamming": float(h.max()),
        "time_to_peak": int(h.argmax()),
        "short_lam": float(est["short_time_rate"]),
        "late_damage": float(h[-nlate:].mean() / args.length),
        "sustained_zero": int(sustained_zero),
        "sync_time": sync_time,
    }


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[wrote] {path}")


def plot_heatmap(path, matrix, windows, temps, title, cbar_label, vmin=0.0, vmax=1.0, cmap="viridis"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    im = ax.imshow(matrix, origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xticks(range(len(windows)), labels=[str(w) for w in windows])
    ax.set_yticks(range(len(temps)), labels=[str(t) for t in temps])
    ax.set_xlabel("local window radius w")
    ax.set_ylabel("temperature T")
    ax.set_title(title)
    for iy in range(len(temps)):
        for ix in range(len(windows)):
            val = matrix[iy, ix]
            ax.text(ix, iy, f"{val:.2f}", ha="center", va="center", color="white" if val < 0.55 else "black", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[wrote] {path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="distilroberta-base")
    p.add_argument("--windows", default="1,2,3,4,6,8")
    p.add_argument("--temps", default="0.6,0.8,1.0,1.2,1.4")
    p.add_argument("--length", type=int, default=48)
    p.add_argument("--steps", type=int, default=80)
    p.add_argument("--pairs", type=int, default=8)
    p.add_argument("--device", default="auto")
    p.add_argument("--absorbing", action="store_true")
    p.add_argument("--late-frac", type=float, default=0.2)
    p.add_argument("--sustain-k", type=int, default=10)
    p.add_argument("--out", default="results")
    p.add_argument("--tag", default="local_distilroberta_L48")
    p.add_argument("--init-seed-stride", type=int, default=5000)
    p.add_argument("--init-seed-offset", type=int, default=3)
    p.add_argument("--noise-seed-stride", type=int, default=5000)
    p.add_argument("--noise-seed-offset", type=int, default=11)
    p.add_argument("--perturb-stride", type=int, default=7)
    p.add_argument("--perturb-offset", type=int, default=1)
    args = p.parse_args()

    windows = parse_list(args.windows, int)
    temps = parse_list(args.temps, float)
    os.makedirs(args.out, exist_ok=True)

    aggregate_rows: list[dict] = []
    pair_rows: list[dict] = []
    hamming_rows: list[dict] = []
    psync = np.zeros((len(temps), len(windows)))
    late = np.zeros_like(psync)
    peak = np.zeros_like(psync)

    for ix, window in enumerate(windows):
        auto, _tok, model, info = build_local(args.model, window, args.device)
        print(f"\n## local w={window} {args.model} L={args.length} pairs={args.pairs} steps={args.steps}")
        print(f"[info] device={info['device']} dead={info['dead_token_str']} vocab={info['vocab']}", flush=True)
        for iy, temp in enumerate(temps):
            curves = []
            cell_pairs = []
            for pair in range(args.pairs):
                h, row = one_pair(auto, info, temp, pair, args)
                curves.append(h)
                row.update({
                    "model": args.model,
                    "model_tag": _model_tag(args.model),
                    "arch": "local",
                    "window": window,
                    "temp": temp,
                    "L": args.length,
                    "steps": args.steps,
                    "pairs": args.pairs,
                    "absorbing": int(args.absorbing),
                })
                cell_pairs.append(row)
                for generation, hamming in enumerate(h):
                    hamming_rows.append({
                        "model": args.model,
                        "model_tag": _model_tag(args.model),
                        "arch": "local",
                        "window": window,
                        "temp": temp,
                        "pair": pair,
                        "generation": generation,
                        "hamming": float(hamming),
                    })
            H = np.vstack(curves)
            p_sync = float(np.mean([r["sustained_zero"] for r in cell_pairs]))
            late_damage = float(np.mean([r["late_damage"] for r in cell_pairs]))
            peak_mean = float(np.mean([r["peak_hamming"] for r in cell_pairs]))
            psync[iy, ix] = p_sync
            late[iy, ix] = late_damage
            peak[iy, ix] = peak_mean / args.length
            aggregate_rows.append({
                "model": args.model,
                "model_tag": _model_tag(args.model),
                "arch": "local",
                "window": window,
                "temp": temp,
                "L": args.length,
                "steps": args.steps,
                "pairs": args.pairs,
                "p_sync": p_sync,
                "late_damage": late_damage,
                "peak_damage": peak_mean / args.length,
                "final_damage": float(H[:, -1].mean() / args.length),
                "sync_count": int(sum(r["sustained_zero"] for r in cell_pairs)),
            })
            pair_rows.extend(cell_pairs)
            print(
                f"  T={temp:<4} p_sync={p_sync:.2f} late={late_damage:.2f} "
                f"peak={peak_mean / args.length:.2f}",
                flush=True,
            )
        del auto, model
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        prefix = os.path.join(args.out, f"fate_map_{args.tag}")
        write_csv(f"{prefix}_aggregate.csv", aggregate_rows)
        write_csv(f"{prefix}_pairs.csv", pair_rows)
        write_csv(f"{prefix}_hamming.csv", hamming_rows)

    prefix = os.path.join(args.out, f"fate_map_{args.tag}")
    plot_heatmap(f"{prefix}_psync.png", psync, windows, temps, f"Coalescence probability: {_model_tag(args.model)} local rule", "P(sustained H=0)", vmin=0, vmax=1, cmap="magma_r")
    plot_heatmap(f"{prefix}_late_damage.png", late, windows, temps, f"Late damage density: {_model_tag(args.model)} local rule", "mean late H/L", vmin=0, vmax=1, cmap="inferno")
    plot_heatmap(f"{prefix}_peak_damage.png", peak, windows, temps, f"Peak damage density: {_model_tag(args.model)} local rule", "mean peak H/L", vmin=0, vmax=1, cmap="viridis")


if __name__ == "__main__":
    main()
