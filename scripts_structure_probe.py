"""Small raw-state/structure probe for local distilroberta ring dynamics.

Writes only results/structure_probe_* files:
  - raw token states and per-run arrays as NPZ
  - compact metrics CSV/JSON
  - support-width-over-time CSV
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch

from llm_life.automaton import LocalMaskedLMAutomaton
from llm_life.model import dead_token_id_mlm, load_mlm, pick_device


def ring_support_width(live: np.ndarray) -> int:
    """Smallest contiguous ring arc containing all live cells."""
    idx = np.flatnonzero(live)
    L = live.shape[0]
    if idx.size == 0:
        return 0
    if idx.size == 1:
        return 1
    gaps = np.diff(np.concatenate([idx, [idx[0] + L]]))
    return int(L - gaps.max() + 1)


def support_widths(states: np.ndarray, dead: int) -> np.ndarray:
    return np.array([ring_support_width(row != dead) for row in states], dtype=np.int16)


def exact_tail_period(states: np.ndarray, max_p: int, tail: int) -> int:
    """Smallest exact period p on the final tail, or 0 if none up to max_p."""
    n = states.shape[0]
    start = max(0, n - max(tail, max_p + 1))
    tail_states = states[start:]
    for p in range(1, max_p + 1):
        if tail_states.shape[0] > p and np.array_equal(tail_states[p:], tail_states[:-p]):
            return p
    return 0


def shift_tail_period(states: np.ndarray, max_p: int, max_abs_shift: int, tail: int) -> dict:
    """Smallest p and ring shift where state[t+p] == roll(state[t], shift)."""
    n = states.shape[0]
    start = max(0, n - max(tail, max_p + 1))
    tail_states = states[start:]
    shifts = [0]
    for s in range(1, max_abs_shift + 1):
        shifts.extend([s, -s])
    for p in range(1, max_p + 1):
        if tail_states.shape[0] <= p:
            continue
        for shift in shifts:
            ok = True
            for t in range(tail_states.shape[0] - p):
                if not np.array_equal(tail_states[t + p], np.roll(tail_states[t], shift)):
                    ok = False
                    break
            if ok:
                return {
                    "period": int(p),
                    "shift": int(shift),
                    "velocity": float(shift / p),
                }
    return {"period": 0, "shift": 0, "velocity": 0.0}


def configurational_equality_corr(states: np.ndarray, max_r: int, burn: int) -> np.ndarray:
    """C_eq(r) = mean_{t,i} 1[state[t,i] == state[t,i+r]] on the ring."""
    st = states[min(burn, states.shape[0] - 1):]
    out = np.empty(max_r, dtype=np.float64)
    for r in range(1, max_r + 1):
        out[r - 1] = (st == np.roll(st, -r, axis=1)).mean()
    return out


def encode_seed(tokenizer, seed_text: str, dead: int, length: int) -> list[int]:
    if seed_text:
        ids = tokenizer.encode(seed_text, add_special_tokens=False)
    else:
        ids = []
    ids = [int(i) for i in ids if int(i) != int(dead)]
    return ids[:length]


def make_init(tokenizer, dead: int, length: int, seed_text: str, device: str) -> tuple[torch.Tensor, list[int]]:
    ids = encode_seed(tokenizer, seed_text, dead, length)
    state = torch.full((length,), int(dead), dtype=torch.long, device=device)
    if ids:
        live = torch.tensor(ids, dtype=torch.long, device=device)
        lo = max(0, (length - live.numel()) // 2)
        state[lo:lo + live.numel()] = live
    return state, ids


def model_tag(name: str) -> str:
    base = name.rsplit("/", 1)[-1]
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in base)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilroberta-base")
    ap.add_argument("--length", type=int, default=64, choices=(64, 96))
    ap.add_argument("--steps", type=int, default=96)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--windows", default="1,2,4")
    ap.add_argument("--seed-text", default=" Comments Related Posts")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-period", type=int, default=16)
    ap.add_argument("--corr-r", type=int, default=8)
    ap.add_argument("--tail", type=int, default=48)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    windows = [int(x) for x in args.windows.split(",") if x.strip()]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device(args.device)
    model, tok, device = load_mlm(args.model, device)
    dead = dead_token_id_mlm(model, tok, device)
    vocab = int(model.config.vocab_size)
    init, seed_ids = make_init(tok, dead, args.length, args.seed_text, device)

    generator = torch.Generator(device="cpu" if device == "mps" else device)
    generator.manual_seed(args.seed)

    arrays: dict[str, np.ndarray] = {
        "windows": np.array(windows, dtype=np.int16),
        "seed_ids": np.array(seed_ids, dtype=np.int64),
        "dead_token": np.array([dead], dtype=np.int64),
    }
    metric_rows = []
    support_rows = []

    for w in windows:
        auto = LocalMaskedLMAutomaton(
            model=model,
            dead_token=dead,
            mask_token=tok.mask_token_id,
            cls_token=tok.cls_token_id,
            sep_token=tok.sep_token_id,
            window=w,
            device=device,
        )
        states_t, _ = auto.trajectory(
            init=init,
            steps=args.steps,
            temperature=args.temp,
            absorbing=True,
            generator=generator,
        )
        states = states_t.detach().cpu().numpy().astype(np.int64, copy=False)
        support = support_widths(states, dead)
        burn = max(0, states.shape[0] - args.tail)
        c_eq = configurational_equality_corr(states, args.corr_r, burn)
        period = exact_tail_period(states, args.max_period, args.tail)
        shift = shift_tail_period(
            states=states,
            max_p=args.max_period,
            max_abs_shift=min(args.length // 2, args.max_period * w),
            tail=args.tail,
        )

        arrays[f"states_w{w}"] = states
        arrays[f"support_w{w}"] = support
        arrays[f"c_eq_w{w}"] = c_eq

        live = states != dead
        tail_support = support[burn:]
        tail_live_density = live[burn:].mean(axis=1)
        changed = (states[1:] != states[:-1]).mean(axis=1) if states.shape[0] > 1 else np.array([0.0])
        tail_changed = changed[max(0, burn - 1):]

        row = {
            "window": w,
            "steps": args.steps,
            "length": args.length,
            "temperature": args.temp,
            "final_live_density": float(live[-1].mean()),
            "final_support_width": int(support[-1]),
            "tail_mean_live_density": float(tail_live_density.mean()),
            "tail_mean_support_width": float(tail_support.mean()),
            "tail_min_support_width": int(tail_support.min()),
            "tail_max_support_width": int(tail_support.max()),
            "tail_mean_activity": float(tail_changed.mean()),
            "exact_tail_period": int(period),
            "shift_period": int(shift["period"]),
            "shift": int(shift["shift"]),
            "velocity": float(shift["velocity"]),
        }
        for r, val in enumerate(c_eq, start=1):
            row[f"C_eq_{r}"] = float(val)
        metric_rows.append(row)

        for t, val in enumerate(support):
            support_rows.append({"window": w, "generation": t, "support_width": int(val)})

    tag = f"{model_tag(args.model)}_T{args.temp:g}_L{args.length}_s{args.seed}"
    npz_path = out_dir / f"structure_probe_{tag}_states.npz"
    metrics_csv = out_dir / f"structure_probe_{tag}_metrics.csv"
    support_csv = out_dir / f"structure_probe_{tag}_support.csv"
    summary_json = out_dir / f"structure_probe_{tag}_summary.json"

    np.savez_compressed(npz_path, **arrays)

    with metrics_csv.open("w", newline="") as f:
        fields = list(metric_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metric_rows)

    with support_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["window", "generation", "support_width"])
        writer.writeheader()
        writer.writerows(support_rows)

    summary = {
        "model": args.model,
        "device": device,
        "length": args.length,
        "steps": args.steps,
        "temperature": args.temp,
        "absorbing": True,
        "windows": windows,
        "seed_text": args.seed_text,
        "seed_ids": seed_ids,
        "seed_decoded": tok.decode(seed_ids),
        "dead_token": int(dead),
        "dead_token_str": tok.decode([dead]),
        "vocab": vocab,
        "metrics": metric_rows,
        "files": {
            "states_npz": str(npz_path),
            "metrics_csv": str(metrics_csv),
            "support_csv": str(support_csv),
            "summary_json": str(summary_json),
        },
    }
    with summary_json.open("w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
