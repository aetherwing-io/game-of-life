"""Command-line driver for the LLM cellular-automaton experiments.

Subcommands
-----------
  single   one trajectory at a fixed temperature -> space-time PNG + metrics CSV
  sweep    a temperature sweep with N seeds each -> phase-diagram PNG + raw CSV
  damage   coupled-noise damage spreading at a fixed temperature -> Hamming PNG

Everything is seeded and reproducible. Raw per-generation and per-run numbers
are written to CSV so the plots can be regenerated / re-analysed independently.

Examples
--------
  python -m llm_life.run single --temp 0.8 --steps 300 --length 128 --absorbing
  python -m llm_life.run sweep  --temps 0.2:1.6:0.1 --seeds 5 --steps 300
  python -m llm_life.run damage --temp 0.8 --steps 200 --pairs 16
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np

from . import metrics, reference_ca
from .viz import embedding_rgb_table, save_animation, save_spacetime


def _gen(device: str, seed: int):
    import torch
    # MPS doesn't support a device-side generator for randint; use CPU and move.
    g = torch.Generator(device="cpu" if device == "mps" else device)
    g.manual_seed(seed)
    return g


def _build(args):
    from .model import pick_device

    device = pick_device(args.device)
    arch = getattr(args, "arch", "causal")
    if arch in ("masked", "local"):
        from .model import dead_token_id_mlm, load_mlm

        model_name = "distilroberta-base" if args.model == "gpt2" else args.model
        model, tok, device = load_mlm(model_name, device)
        dead = dead_token_id_mlm(model, tok, device)
        common = dict(
            model=model, dead_token=dead, mask_token=tok.mask_token_id,
            cls_token=tok.cls_token_id, sep_token=tok.sep_token_id, device=device,
        )
        if arch == "local":
            from .automaton import LocalMaskedLMAutomaton
            auto = LocalMaskedLMAutomaton(window=getattr(args, "window", 4), **common)
            extra = {"arch": "local", "model": model_name, "window": getattr(args, "window", 4)}
        else:
            from .automaton import MaskedLMAutomaton
            auto = MaskedLMAutomaton(**common)
            extra = {"arch": "masked", "model": model_name}
    else:
        from .automaton import LLMAutomaton
        from .model import dead_token_id, load

        model, tok, device = load(args.model, device)
        bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
        dead = dead_token_id(model, tok, bos, device)
        auto = LLMAutomaton(model, dead_token=dead, bos_token=bos, device=device)
        extra = {"arch": "causal", "model": args.model}

    info = {
        "device": device,
        "vocab": model.config.vocab_size,
        "dead_token": dead,
        "dead_token_str": repr(tok.decode([dead])),
        **extra,
    }
    return auto, tok, model, info


def _random_init(auto, length: int, vocab: int, device: str, seed: int):
    import torch
    g = _gen(device, seed)
    s = torch.randint(0, vocab, (length,), generator=g)
    return s.to(device)


def _to_np(states) -> np.ndarray:
    return states.detach().to("cpu").numpy()


def _make_init(args, info, tok=None):
    """Build the initial generation. seed-mode 'random' = full random lattice;
    'single' = one live cell on a dead background (the classic glider seed);
    'patch' = a small block of live cells. If --seed-text is given, the live
    region is those exact tokens instead of random ones (control the input)."""
    import torch
    mode = getattr(args, "seed_mode", "random")
    L, vocab, device, dead = args.length, info["vocab"], info["device"], info["dead_token"]
    seed_text = getattr(args, "seed_text", None)
    if mode == "random" and not seed_text:
        return _random_init(None, L, vocab, device, args.seed)
    g = _gen(device, args.seed)
    state = torch.full((L,), dead, dtype=torch.long, device=device)
    if seed_text:
        ids = tok.encode(seed_text, add_special_tokens=False)
        live = torch.tensor(ids, dtype=torch.long)
    else:
        width = 1 if mode == "single" else max(1, getattr(args, "patch", 5))
        live = torch.randint(0, vocab, (width,), generator=g)
    width = live.shape[0]
    lo = max(0, (L - width) // 2)
    state[lo:lo + width] = live[:L].to(device)
    return state


# --------------------------------------------------------------------------- #
# single
# --------------------------------------------------------------------------- #
def cmd_single(args):
    auto, tok, model, info = _build(args)
    auto.freq_penalty = getattr(args, "freq_penalty", 0.0)
    info["freq_penalty"] = auto.freq_penalty
    print(f"[info] {info}")
    init = _make_init(args, info, tok)
    g = _gen(info["device"], args.seed + 1)
    states_t, _ = auto.trajectory(
        init, args.steps, args.temp, args.absorbing, generator=g,
        refractory=getattr(args, "refractory", 0),
    )
    states = _to_np(states_t)

    rho = metrics.activity(states)
    live = metrics.live_density(states, info["dead_token"])
    ent = metrics.token_entropy(states)
    tau = metrics.integrated_autocorr_time(rho)
    xi = metrics.spatial_corr_length(states)
    print(f"[result] mean rho={rho.mean():.4f}  mean entropy={ent.mean():.4f} bits  "
          f"tau_int={tau:.2f}  xi={xi:.2f}  final live={live[-1]:.4f}")

    os.makedirs(args.out, exist_ok=True)
    pen_tag = f"_p{args.freq_penalty}" if getattr(args, "freq_penalty", 0.0) else ""
    win_tag = f"_w{info['window']}" if info.get("window") else ""
    sm = getattr(args, "seed_mode", "random")
    seed_tag = "_txt" if getattr(args, "seed_text", None) else ("" if sm == "random" else f"_{sm}")
    ref_tag = f"_r{args.refractory}" if getattr(args, "refractory", 0) else ""
    tag = (f"{info['arch']}{win_tag}_T{args.temp}{pen_tag}{seed_tag}{ref_tag}"
           f"_L{args.length}_s{args.seed}{'_abs' if args.absorbing else ''}")

    # per-generation CSV
    csv_path = os.path.join(args.out, f"single_{tag}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["generation", "activity", "live_density", "token_entropy_bits"])
        w.writerow([0, "", live[0], ent[0]])
        for t in range(len(rho)):
            w.writerow([t + 1, rho[t], live[t + 1], ent[t + 1]])
    print(f"[wrote] {csv_path}")

    if getattr(args, "dump_tokens", False):
        dead = info["dead_token"]
        txt_path = os.path.join(args.out, f"tokens_{tag}.txt")
        with open(txt_path, "w") as f:
            f.write(f"# {info}\n# dead token {dead} = {tok.decode([dead])!r} shown as '.'\n")
            for t in range(states.shape[0]):
                cells = ["." if int(x) == dead else (tok.decode([int(x)]).strip() or "_")
                         for x in states[t]]
                f.write(f"g{t:>4} | " + " ".join(c[:8] for c in cells) + "\n")
        print(f"[wrote] {txt_path}")

    # space-time diagram
    emb = model.get_input_embeddings().weight.detach().to("cpu").float().numpy()
    rgb = embedding_rgb_table(emb)
    title = (f"{info['model']} ({info['arch']})  T={args.temp}  L={args.length}  "
             f"{'absorbing' if args.absorbing else 'soft'}")
    png = os.path.join(args.out, f"spacetime_{tag}.png")
    save_spacetime(states, rgb, png, title=title)
    print(f"[wrote] {png}")
    if args.animate:
        save_animation(states, rgb, os.path.join(args.out, f"spacetime_{tag}.gif"),
                       title=title)


# --------------------------------------------------------------------------- #
# reference  (known-class CA baselines, same pipeline)
# --------------------------------------------------------------------------- #
def cmd_reference(args):
    os.makedirs(args.out, exist_ok=True)
    rgb = reference_ca.binary_rgb_table()
    classes = {110: "Class 4 (edge of chaos)", 30: "Class 3 (chaotic)",
               90: "Class 3 (Sierpinski)", 250: "Class 2 (periodic)", 0: "Class 1 (dead)"}
    for rule in args.rules:
        seed_mode = "single" if rule in (110, 90) else "random"
        st = reference_ca.elementary(rule, args.length, args.steps, seed=1, init=seed_mode)
        cls = classes.get(rule, "")
        title = f"Elementary CA rule {rule} — {cls}"
        save_spacetime(st, rgb, os.path.join(args.out, f"ref_rule{rule}.png"), title=title)
        rho = metrics.activity(st)[args.steps // 4:]
        ent = metrics.token_entropy(st)[args.steps // 4:]
        tau = metrics.integrated_autocorr_time(metrics.activity(st))
        print(f"rule {rule:>3} {cls:<26} rho={rho.mean():.3f}  H={ent.mean():.3f} bits  "
              f"tau_int={tau:.2f}")
        if args.animate:
            save_animation(st, rgb, os.path.join(args.out, f"ref_rule{rule}.gif"), title=title)


# --------------------------------------------------------------------------- #
# sweep  (the phase diagram)
# --------------------------------------------------------------------------- #
def _parse_temps(spec: str) -> list[float]:
    if ":" in spec:
        lo, hi, step = (float(x) for x in spec.split(":"))
        n = int(round((hi - lo) / step)) + 1
        return [round(lo + i * step, 6) for i in range(n)]
    return [float(x) for x in spec.split(",")]


def cmd_sweep(args):
    auto, tok, model, info = _build(args)
    print(f"[info] {info}")
    temps = _parse_temps(args.temps)
    print(f"[sweep] temperatures: {temps}")

    rows = []  # one per (temp, seed)
    for T in temps:
        for s in range(args.seeds):
            init = _random_init(auto, args.length, info["vocab"], info["device"], 1000 * s + 7)
            g = _gen(info["device"], 1000 * s + 99)
            states = _to_np(auto.trajectory(init, args.steps, T, args.absorbing, generator=g)[0])

            # discard a burn-in transient before measuring steady-state stats
            burn = min(args.burn, args.steps // 2)
            rho = metrics.activity(states)[burn:]
            ent = metrics.token_entropy(states)[burn:]
            live = metrics.live_density(states, info["dead_token"])[burn:]
            # tau_int / xi must be measured on the post-burn STEADY STATE, not
            # the full trajectory -- otherwise the slow absorbing transient
            # inflates the autocorrelation and masquerades as critical slowing
            # down. (Edge-of-chaos shows up as a *peak* here, not a plateau.)
            tau = metrics.integrated_autocorr_time(metrics.activity(states)[burn:])
            xi = metrics.spatial_corr_length(states[burn:])
            rows.append({
                "temp": T, "seed": s,
                "activity": float(rho.mean()),
                "entropy": float(ent.mean()),
                "live": float(live.mean()),
                "tau_int": float(tau),
                "xi": float(xi),
            })
            print(f"  T={T:<5} seed={s} rho={rho.mean():.4f} H={ent.mean():.3f} "
                  f"tau={tau:.2f} xi={xi:.2f}")

    os.makedirs(args.out, exist_ok=True)
    raw = os.path.join(args.out, "sweep_raw.csv")
    with open(raw, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[wrote] {raw}")

    # aggregate mean +/- std across seeds
    def agg(key):
        m, e = [], []
        for T in temps:
            vals = np.array([r[key] for r in rows if r["temp"] == T])
            m.append(vals.mean())
            e.append(vals.std(ddof=1) if vals.size > 1 else 0.0)
        return np.array(m), np.array(e)

    _plot_sweep(temps, agg, args, info)


def _plot_sweep(temps, agg, args, info):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        ("activity", "activity  rho", "tab:blue"),
        ("entropy", "token entropy (bits)", "tab:green"),
        ("tau_int", "autocorr time  tau_int", "tab:red"),
        ("xi", "spatial corr length  xi", "tab:purple"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for ax, (key, label, col) in zip(axes.ravel(), panels):
        m, e = agg(key)
        ax.errorbar(temps, m, yerr=e, marker="o", ms=4, lw=1.2, capsize=3, color=col)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("temperature T")
    mode = "absorbing" if args.absorbing else "soft"
    fig.suptitle(f"LLM-CA phase sweep — {args.model}, L={args.length}, "
                 f"{args.seeds} seeds, {mode}\n"
                 f"tau_int / xi peaks locate the edge-of-chaos band")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    png = os.path.join(args.out, "sweep_phasediagram.png")
    fig.savefig(png, dpi=140)
    plt.close(fig)
    print(f"[wrote] {png}")


# --------------------------------------------------------------------------- #
# sweep2d  (temperature x frequency-penalty phase plane)
# --------------------------------------------------------------------------- #
def cmd_sweep2d(args):
    auto, tok, model, info = _build(args)
    print(f"[info] {info}")
    temps = _parse_temps(args.temps)
    pens = _parse_temps(args.penalties)
    print(f"[sweep2d] {len(temps)} temps x {len(pens)} penalties x {args.seeds} seeds")

    burn = min(args.burn, args.steps // 2)
    # grids[key][i_pen, j_temp] = seed-averaged steady-state metric
    keys = ["activity", "entropy", "live", "tau_int", "xi"]
    grids = {k: np.zeros((len(pens), len(temps))) for k in keys}
    rows = []
    for ip, pen in enumerate(pens):
        for jt, T in enumerate(temps):
            vals = {k: [] for k in keys}
            for s in range(args.seeds):
                auto.freq_penalty = float(pen)
                init = _random_init(auto, args.length, info["vocab"], info["device"], 1000 * s + 7)
                g = _gen(info["device"], 1000 * s + 99)
                states = _to_np(auto.trajectory(init, args.steps, T, args.absorbing, generator=g)[0])
                vals["activity"].append(float(metrics.activity(states)[burn:].mean()))
                vals["entropy"].append(float(metrics.token_entropy(states)[burn:].mean()))
                vals["live"].append(float(metrics.live_density(states, info["dead_token"])[burn:].mean()))
                vals["tau_int"].append(float(metrics.integrated_autocorr_time(metrics.activity(states)[burn:])))
                vals["xi"].append(float(metrics.spatial_corr_length(states[burn:])))
            for k in keys:
                grids[k][ip, jt] = np.mean(vals[k])
            rows.append({"penalty": pen, "temp": T, **{k: grids[k][ip, jt] for k in keys}})
            print(f"  pen={pen:<5} T={T:<5} rho={grids['activity'][ip,jt]:.3f} "
                  f"H={grids['entropy'][ip,jt]:.2f} tau={grids['tau_int'][ip,jt]:.1f} "
                  f"xi={grids['xi'][ip,jt]:.1f}")

    os.makedirs(args.out, exist_ok=True)
    raw = os.path.join(args.out, f"sweep2d_raw_{info['arch']}.csv")
    with open(raw, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[wrote] {raw}")
    _plot_sweep2d(grids, temps, pens, args, info)


def _plot_sweep2d(grids, temps, pens, args, info):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("activity", "activity rho"), ("entropy", "entropy (bits)"),
              ("tau_int", "autocorr time tau_int"), ("xi", "spatial corr len xi")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    extent = [min(temps), max(temps), min(pens), max(pens)]
    for ax, (key, label) in zip(axes.ravel(), panels):
        im = ax.imshow(grids[key], origin="lower", aspect="auto", extent=extent,
                       cmap="viridis", interpolation="nearest")
        ax.set_xlabel("temperature T")
        ax.set_ylabel("frequency penalty")
        ax.set_title(label)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    mode = "absorbing" if args.absorbing else "soft"
    fig.suptitle(f"LLM-CA phase plane — {info['model']} ({info['arch']}), L={args.length}, "
                 f"{args.seeds} seeds, {mode}\n"
                 f"disorder axis (T) x balance axis (penalty); look for a tau_int/xi ridge")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    png = os.path.join(args.out, f"sweep2d_{info['arch']}_{mode}.png")
    fig.savefig(png, dpi=140)
    plt.close(fig)
    print(f"[wrote] {png}")


# --------------------------------------------------------------------------- #
# damage  (coupled-noise Lyapunov / butterfly)
# --------------------------------------------------------------------------- #
def cmd_damage(args):
    import torch

    from .automaton import StepNoise
    from .sampler import gumbel_like

    auto, tok, model, info = _build(args)
    print(f"[info] {info}")
    vocab = info["vocab"]
    curves = []
    for p in range(args.pairs):
        init = _random_init(auto, args.length, vocab, info["device"], 5000 * p + 3)
        g = _gen(info["device"], 5000 * p + 11)
        # two replicas differing in a single site; driven in lockstep with the
        # SAME noise tensor each step (coupled noise) so any divergence is due
        # to the perturbation alone. One Gumbel tensor is held at a time.
        ref = init.clone()
        pert = init.clone()
        site = (7 * p + 1) % args.length
        pert[site] = int((int(pert[site].item()) + 1 + p) % vocab)
        h = [float((ref != pert).sum().item())]
        template = torch.empty(args.length, vocab, dtype=torch.float32, device=info["device"])
        for _ in range(args.steps):
            shared = StepNoise(gumbel=gumbel_like(template, generator=g))  # shape only
            ref, _ = auto.step(ref, args.temp, args.absorbing, noise=shared)
            pert, _ = auto.step(pert, args.temp, args.absorbing, noise=shared)
            h.append(float((ref != pert).sum().item()))
        h = np.array(h, dtype=float)
        curves.append(h)
        est = metrics.lyapunov_estimate(h)
        print(f"  pair {p}: lambda={est['lyapunov']:+.4f}  "
              f"final_hamming={est['final_hamming']:.0f}/{args.length}")

    H = np.vstack(curves)
    mean_h = H.mean(axis=0)
    std_h = H.std(axis=0)
    overall = metrics.lyapunov_estimate(mean_h)
    print(f"[result] mean lambda={overall['lyapunov']:+.4f}  "
          f"mean final separation={mean_h[-1]:.1f}/{args.length} sites")

    os.makedirs(args.out, exist_ok=True)
    tag = f"{args.arch}_T{args.temp}_L{args.length}{'_abs' if args.absorbing else ''}"
    csv_path = os.path.join(args.out, f"damage_{tag}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["generation", "mean_hamming", "std_hamming"])
        for t in range(mean_h.size):
            w.writerow([t, mean_h[t], std_h[t]])
    print(f"[wrote] {csv_path}")

    _plot_damage(mean_h, std_h, args, overall)


def _plot_damage(mean_h, std_h, args, overall):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(mean_h.size)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.fill_between(t, mean_h - std_h, mean_h + std_h, alpha=0.25, color="tab:orange")
    ax1.plot(t, mean_h, color="tab:orange", lw=1.5)
    ax1.set(xlabel="generation", ylabel="Hamming distance (sites)",
            title="damage spreading (linear)")
    ax1.grid(alpha=0.3)
    ax2.semilogy(t, np.clip(mean_h, 1e-1, None), color="tab:orange", lw=1.5)
    ax2.set(xlabel="generation", ylabel="Hamming distance (log)",
            title=f"log scale — lambda={overall['lyapunov']:+.4f}")
    ax2.grid(alpha=0.3, which="both")
    mode = "absorbing" if args.absorbing else "soft"
    fig.suptitle(f"coupled-noise damage spreading — {args.model} ({args.arch}), "
                 f"T={args.temp}, L={args.length}, {args.pairs} pairs, {mode}")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    png = os.path.join(args.out, f"damage_{args.arch}_{mode}_T{args.temp}.png")
    fig.savefig(png, dpi=140)
    plt.close(fig)
    print(f"[wrote] {png}")


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description="LLM-as-cellular-automaton experiments")
    p.add_argument("--model", default="gpt2",
                   help="model name; default gpt2 (causal) or distilroberta-base (masked)")
    p.add_argument("--arch", default="causal", choices=["causal", "masked", "local"],
                   help="causal (leftward) | masked (bidirectional global) | local (windowed ring)")
    p.add_argument("--window", type=int, default=4,
                   help="neighborhood radius w for --arch local (window = 2w+1)")
    p.add_argument("--device", default="auto", help="auto|mps|cuda|cpu")
    p.add_argument("--out", default="results", help="output directory")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("single", help="one trajectory -> space-time diagram")
    sp.add_argument("--temp", type=float, default=0.8)
    sp.add_argument("--length", type=int, default=128)
    sp.add_argument("--steps", type=int, default=300)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--absorbing", action="store_true")
    sp.add_argument("--freq-penalty", type=float, default=0.0, dest="freq_penalty",
                    help="overpopulation-death balance knob (0 = off)")
    sp.add_argument("--seed-mode", choices=["random", "single", "patch"], default="random",
                    dest="seed_mode", help="initial generation (single/patch = glider seed)")
    sp.add_argument("--patch", type=int, default=5, help="live-block width for seed-mode patch")
    sp.add_argument("--seed-text", default=None, dest="seed_text",
                    help="seed the live region with these exact tokens (controls the input)")
    sp.add_argument("--refractory", type=int, default=0,
                    help="cells die after this many gens alive (Brian's-Brain decay; enables gliders)")
    sp.add_argument("--dump-tokens", action="store_true", dest="dump_tokens",
                    help="also write the decoded generation-by-generation token grid to a .txt")
    sp.add_argument("--animate", action="store_true", help="also write an animated GIF")
    sp.set_defaults(func=cmd_single)

    s2 = sub.add_parser("sweep2d", help="temperature x frequency-penalty phase plane")
    s2.add_argument("--temps", default="0.2:1.6:0.2", help="lo:hi:step or comma list")
    s2.add_argument("--penalties", default="0.0:5.0:0.5", help="lo:hi:step or comma list")
    s2.add_argument("--length", type=int, default=128)
    s2.add_argument("--steps", type=int, default=200)
    s2.add_argument("--burn", type=int, default=100, help="burn-in gens discarded")
    s2.add_argument("--seeds", type=int, default=2)
    s2.add_argument("--absorbing", action="store_true")
    s2.set_defaults(func=cmd_sweep2d)

    rf = sub.add_parser("reference", help="known-class CA baselines (same pipeline)")
    rf.add_argument("--rules", type=int, nargs="+", default=[110, 30, 90, 250])
    rf.add_argument("--length", type=int, default=200)
    rf.add_argument("--steps", type=int, default=200)
    rf.add_argument("--animate", action="store_true")
    rf.set_defaults(func=cmd_reference)

    sw = sub.add_parser("sweep", help="temperature sweep -> phase diagram")
    sw.add_argument("--temps", default="0.2:1.6:0.1", help="lo:hi:step or comma list")
    sw.add_argument("--length", type=int, default=128)
    sw.add_argument("--steps", type=int, default=300)
    sw.add_argument("--burn", type=int, default=100, help="burn-in gens discarded")
    sw.add_argument("--seeds", type=int, default=5)
    sw.add_argument("--absorbing", action="store_true")
    sw.set_defaults(func=cmd_sweep)

    dm = sub.add_parser("damage", help="coupled-noise damage spreading")
    dm.add_argument("--temp", type=float, default=0.8)
    dm.add_argument("--length", type=int, default=128)
    dm.add_argument("--steps", type=int, default=200)
    dm.add_argument("--pairs", type=int, default=16, help="perturbation pairs to average")
    dm.add_argument("--absorbing", action="store_true")
    dm.set_defaults(func=cmd_damage)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
