"""Animated coupled-damage visualization.

Renders a shared-noise damage-spreading pair as an animated three-panel GIF:

    reference replica | perturbed replica | damage mask

This is the visual companion to the consistency CSVs. It makes coalescence vs
damage survival visible without mentally reconstructing it from Hamming curves.
"""

from __future__ import annotations

import argparse
import csv
import os
from types import SimpleNamespace

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
from PIL import Image, ImageDraw

from llm_life.automaton import StepNoise
from llm_life.run import _build, _gen, _model_tag, _random_init
from llm_life.sampler import gumbel_like
from llm_life.viz import embedding_rgb_table, hash_rgb_table


def color_table(model, info):
    if info["arch"] == "mlx":
        try:
            from llm_life.model import mlx_input_embeddings
            return embedding_rgb_table(mlx_input_embeddings(model, info["vocab"]))
        except Exception:
            return hash_rgb_table(info["vocab"])
    emb = model.get_input_embeddings().weight.detach().to("cpu").float().numpy()
    return embedding_rgb_table(emb)


def run_pair(auto, info, args):
    device = info["device"]
    vocab = info["vocab"]
    init = _random_init(auto, args.length, vocab, device, args.init_seed)
    ref = init.clone()
    pert = init.clone()
    site = args.perturb_site % args.length
    pert[site] = int((int(pert[site].item()) + 1 + args.pair) % vocab)

    ref_states = torch.empty(args.steps + 1, args.length, dtype=torch.long, device=device)
    pert_states = torch.empty_like(ref_states)
    ref_states[0] = ref
    pert_states[0] = pert
    hamming = [float((ref != pert).sum().item())]

    g = _gen(device, args.noise_seed)
    template = torch.empty(args.length, vocab, dtype=torch.float32, device=device)
    for t in range(args.steps):
        shared = StepNoise(gumbel=gumbel_like(template, generator=g))
        ref, _ = auto.step(ref, args.temp, args.absorbing, noise=shared)
        pert, _ = auto.step(pert, args.temp, args.absorbing, noise=shared)
        ref_states[t + 1] = ref
        pert_states[t + 1] = pert
        hamming.append(float((ref != pert).sum().item()))

    return (
        ref_states.detach().to("cpu").numpy(),
        pert_states.detach().to("cpu").numpy(),
        np.asarray(hamming, dtype=float),
        site,
    )


def pad_window(img, window):
    if img.shape[0] >= window:
        return img[-window:]
    pad = np.ones((window - img.shape[0], img.shape[1], img.shape[2]), dtype=img.dtype)
    return np.vstack([pad, img])


def render_frames(ref_states, pert_states, rgb, hamming, title, args):
    ref_img = rgb[ref_states]
    pert_img = rgb[pert_states]
    damage = ref_states != pert_states
    damage_img = np.ones((*damage.shape, 3), dtype=float)
    damage_img[damage] = np.array([0.92, 0.05, 0.02])

    frames = []
    gap = 6
    header_h = 46
    cell = args.cell
    panel_w = args.length
    T1 = ref_states.shape[0]
    peak = int(hamming.max())
    final = int(hamming[-1])

    for t in range(T1):
        lo = max(0, t - args.window + 1)
        ref_chunk = pad_window(ref_img[lo : t + 1], args.window)
        pert_chunk = pad_window(pert_img[lo : t + 1], args.window)
        dmg_chunk = pad_window(damage_img[lo : t + 1], args.window)
        chunk = np.hstack([
            ref_chunk,
            np.ones((args.window, gap, 3)),
            pert_chunk,
            np.ones((args.window, gap, 3)),
            dmg_chunk,
        ])
        arr = (np.clip(chunk, 0, 1) * 255).astype(np.uint8)
        im = Image.fromarray(arr, "RGB").resize((arr.shape[1] * cell, arr.shape[0] * cell), Image.Resampling.NEAREST)
        canvas = Image.new("RGB", (im.width, im.height + header_h), "white")
        canvas.paste(im, (0, header_h))
        d = ImageDraw.Draw(canvas)
        d.text((8, 4), title, fill=(0, 0, 0))
        d.text((8, 22), f"gen {t:03d} | H={int(hamming[t])}/{args.length} | peak={peak} | final={final}", fill=(0, 0, 0))
        scale = cell
        x1 = panel_w * scale
        x2 = (panel_w * 2 + gap) * scale
        d.text((panel_w * scale // 2 - 24, header_h - 16), "ref", fill=(0, 0, 0))
        d.text((x1 + gap * scale + panel_w * scale // 2 - 34, header_h - 16), "perturbed", fill=(0, 0, 0))
        d.text((x2 + gap * scale + panel_w * scale // 2 - 34, header_h - 16), "damage", fill=(0, 0, 0))
        frames.append(canvas)
    return frames


def write_hamming(path, hamming):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["generation", "hamming"])
        for i, h in enumerate(hamming):
            w.writerow([i, h])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="EleutherAI/pythia-160m")
    p.add_argument("--arch", choices=["causal", "masked", "local", "mlx"], default="causal")
    p.add_argument("--window-radius", type=int, default=2, dest="window_radius")
    p.add_argument("--device", default="auto")
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--length", type=int, default=48)
    p.add_argument("--steps", type=int, default=80)
    p.add_argument("--pair", type=int, default=0)
    p.add_argument("--init-seed", type=int, default=3)
    p.add_argument("--noise-seed", type=int, default=11)
    p.add_argument("--perturb-site", type=int, default=1)
    p.add_argument("--absorbing", action="store_true")
    p.add_argument("--window", type=int, default=80, help="scrolling history rows shown")
    p.add_argument("--cell", type=int, default=5)
    p.add_argument("--fps", type=int, default=18)
    p.add_argument("--out", default="results")
    p.add_argument("--tag", default=None)
    args = p.parse_args()

    build_args = SimpleNamespace(
        model=args.model,
        arch=args.arch,
        window=args.window_radius,
        device=args.device,
    )
    auto, _tok, model, info = _build(build_args)
    rgb = color_table(model, info)
    ref, pert, hamming, site = run_pair(auto, info, args)

    win_tag = f"_w{info['window']}" if info.get("window") else ""
    mode = "absorbing" if args.absorbing else "soft"
    tag = args.tag or (
        f"{info['arch']}{win_tag}_{_model_tag(info['model'])}_T{args.temp}_L{args.length}"
        f"_pair{args.pair}_s{args.init_seed}_n{args.noise_seed}_{mode}"
    )
    os.makedirs(args.out, exist_ok=True)
    title = f"{info['model']} ({info['arch']}{win_tag}) T={args.temp} L={args.length} shared-noise pair, perturb site {site}"
    frames = render_frames(ref, pert, rgb, hamming, title, args)
    gif_path = os.path.join(args.out, f"pairdamage_{tag}.gif")
    png_path = os.path.join(args.out, f"pairdamage_{tag}.png")
    csv_path = os.path.join(args.out, f"pairdamage_{tag}.csv")
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=int(1000 / args.fps), loop=0)
    frames[-1].save(png_path)
    write_hamming(csv_path, hamming)
    print(f"[result] final={int(hamming[-1])}/{args.length} peak={int(hamming.max())} frames={len(frames)}")
    print(f"[wrote] {gif_path}")
    print(f"[wrote] {png_path}")
    print(f"[wrote] {csv_path}")


if __name__ == "__main__":
    main()
