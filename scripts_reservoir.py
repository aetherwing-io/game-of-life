"""Reservoir-computing apparatus over the causal LLM cellular-automaton map.

The causal full-attention map is a *consistent* (echo-state) map under shared
Gumbel noise (FINDINGS §23). This script turns it into a driven reservoir,
verifies the driven echo-state property, and measures linear Memory Capacity.

Subcommands
-----------
  apparatus  build the reservoir, run it, and sanity-check determinism + readout (#32)
  esp        driven echo-state property: replicas with different inits, same
             input + same noise; sweep T x injection strength (#33)
  mc         linear Memory Capacity (Jaeger) with trivial baselines (#34)

Run (offline, MPS):
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts_reservoir.py esp --device mps
"""

from __future__ import annotations

import argparse
import csv
import os
import time
import warnings

import numpy as np

# Apple Accelerate (numpy 2.x on macOS) raises spurious FP-exception flags inside
# BLAS matmul ("divide by zero / overflow / invalid encountered in matmul") even
# when the result is finite and correct (verified: the shift-register calibration
# still returns MC=20.000 exactly). Silence just those to keep logs readable.
warnings.filterwarnings("ignore", message=".*encountered in matmul.*")

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from llm_life import reservoir as rz

OUT = "results"
DEFAULT_MODEL = "EleutherAI/pythia-160m"


# --------------------------------------------------------------------------- #
# model / reservoir construction (load the model ONCE, reuse for every cell)
# --------------------------------------------------------------------------- #
def build_causal(model_name: str, device: str):
    import torch
    from llm_life.automaton import LLMAutomaton
    from llm_life.model import dead_token_id, load, pick_device

    dev = pick_device(device)
    model, tok, dev = load(model_name, dev)
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    dead = dead_token_id(model, tok, bos, dev)
    vocab = getattr(model.config, "vocab_size", None) or model.config.text_config.vocab_size
    auto = LLMAutomaton(model, dead_token=dead, bos_token=bos, device=dev)
    emb = model.get_input_embeddings().weight.detach().to("cpu").float().numpy()
    return auto, tok, model, dev, vocab, dead, emb


def make_reservoir(auto, L, vocab, emb, dead, n_in, n_bins, temp, device,
                   codebook_mode="pc1", absorbing=False, codebook_seed=0):
    """A DrivenReservoir with input sites at the far left and a fixed codebook."""
    input_sites = np.arange(n_in)
    codebook = rz.build_codebook(n_bins, vocab, emb, mode=codebook_mode,
                                 seed=codebook_seed, exclude=(dead,))
    return rz.DrivenReservoir(
        auto=auto, L=L, vocab=vocab, codebook=codebook, input_sites=input_sites,
        temp=temp, absorbing=absorbing, device=device,
    )


def iid_input(T, seed):
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=T)


# --------------------------------------------------------------------------- #
# #32  apparatus: determinism + readout sanity
# --------------------------------------------------------------------------- #
def cmd_apparatus(args):
    auto, tok, model, dev, vocab, dead, emb = build_causal(args.model, args.device)
    L = args.length
    table = rz.pca_readout_table(emb, args.k)
    res = make_reservoir(auto, L, vocab, emb, dead, args.n_in, args.n_bins,
                         args.temp, dev, codebook_mode=args.codebook)
    print(f"[info] model={args.model} dev={dev} L={L} vocab={vocab} "
          f"dead={dead}={tok.decode([dead])!r}")
    print(f"[info] n_in={args.n_in} input_sites={res.input_sites.tolist()} "
          f"n_reservoir={len(res.reservoir_sites)} n_bins={res.n_bins} K={args.k} "
          f"readout_dim={len(res.reservoir_sites)*args.k}")
    print(f"[codebook:{args.codebook}] tokens={res.codebook.tolist()}")
    print(f"           decoded={[tok.decode([int(c)]) for c in res.codebook]}")

    T = args.steps
    u = iid_input(T, seed=1)
    noises = rz.make_noise(T, L, vocab, dev, seed=args.noise_seed)

    # determinism: identical (init, input, noise) -> byte-identical trajectory
    init = res.random_init(seed=10)
    t0 = time.time()
    s1 = res.run(u, init, noises)
    s2 = res.run(u, init.clone(), noises)
    dt = time.time() - t0
    identical = bool(np.array_equal(s1, s2))
    print(f"[determinism] two runs identical={identical}  ({dt:.1f}s for {2*T} steps)")

    # input sensitivity: flip one input symbol mid-stream -> divergence downstream
    u2 = u.copy(); u2[T // 2] = -u2[T // 2]
    s3 = res.run(u2, init.clone(), noises)
    diff_res = (s1[:, res.reservoir_sites] != s3[:, res.reservoir_sites]).sum(axis=1)
    print(f"[input-sensitivity] reservoir Hamming after single-symbol flip at t={T//2}: "
          f"max={int(diff_res.max())}/{len(res.reservoir_sites)} "
          f"final={int(diff_res[-1])}")

    # readout sanity: feature matrix shape + a quick lag-0 / lag-1 linear probe
    X = res.features(s1, table)
    print(f"[readout] feature matrix X shape={X.shape}  "
          f"(rank~{np.linalg.matrix_rank(X[args.washout:args.washout+200] - X[args.washout:args.washout+200].mean(0)):d})")
    sp = rz.make_split(T, args.washout, T - args.washout - 400, 200, 200)
    Xz, = rz.standardize(X[sp.train], X)
    mc = rz.memory_capacity(Xz, u, sp, kmax=args.kmax)
    print(f"[probe] MC={mc['MC']:.2f} over k=0..{args.kmax} (readout_dim={mc['readout_dim']})")
    print(f"        MC_k head: " + " ".join(f"{v:.2f}" for v in mc["mc_k"][:8]))
    print("[ok] apparatus built and validated.")


# --------------------------------------------------------------------------- #
# #33  driven echo-state property
# --------------------------------------------------------------------------- #
def cmd_esp(args):
    auto, tok, model, dev, vocab, dead, emb = build_causal(args.model, args.device)
    L = args.length
    temps = [float(x) for x in args.temps.split(",")]
    n_ins = [int(x) for x in args.n_ins.split(",")]
    T = args.steps
    print(f"[info] model={args.model} dev={dev} L={L} vocab={vocab} steps={T} "
          f"pairs={args.pairs}  temps={temps} n_ins={n_ins}")

    os.makedirs(OUT, exist_ok=True)
    rows = []
    curves = {}  # (T, n_in) -> mean reservoir-Hamming curve
    for n_in in n_ins:
        for temp in temps:
            res = make_reservoir(auto, L, vocab, emb, dead, n_in, args.n_bins,
                                 temp, dev, codebook_mode=args.codebook,
                                 absorbing=args.absorbing)
            nres = len(res.reservoir_sites)
            pair_curves = []
            for p in range(args.pairs):
                u = iid_input(T, seed=100 + p)
                noises = rz.make_noise(T, L, vocab, dev, seed=4000 + p)
                initA = res.random_init(seed=10 * p + 1)
                initB = res.random_init(seed=10 * p + 2)
                sA = res.run(u, initA, noises)
                sB = res.run(u, initB, noises)
                ham = (sA[:, res.reservoir_sites] != sB[:, res.reservoir_sites]).sum(axis=1)
                pair_curves.append(ham.astype(float))
            H = np.vstack(pair_curves)            # (pairs, T)
            mean_h = H.mean(axis=0)
            curves[(temp, n_in)] = mean_h
            tail = args.tail
            late = float(mean_h[-tail:].mean())
            late_frac = late / nres
            # per-pair: synchronized if last `tail` gens are exactly 0
            sync = np.mean([1.0 if H[p, -tail:].max() == 0 else 0.0 for p in range(args.pairs)])
            # time to (near) sync of the mean curve: first gen with mean_h < 0.5
            below = np.where(mean_h < 0.5)[0]
            t_sync = int(below[0]) if below.size else -1
            verdict = "ESP" if late_frac < 0.02 else ("partial" if late_frac < 0.3 else "no-ESP")
            rows.append({
                "n_in": n_in, "temp": temp, "n_reservoir": nres,
                "init_hamming": float(mean_h[0]), "peak_hamming": float(mean_h.max()),
                "late_hamming": late, "late_frac": late_frac,
                "sync_fraction": float(sync), "t_sync": t_sync, "verdict": verdict,
            })
            print(f"  n_in={n_in} T={temp:<4} nres={nres} init_H={mean_h[0]:.0f} "
                  f"peak={mean_h.max():.0f} late_H={late:.2f} ({late_frac:.3f}) "
                  f"sync={sync:.2f} t_sync={t_sync} {verdict}", flush=True)

    tag = f"{rz_model_tag(args.model)}_L{L}"
    raw = os.path.join(OUT, f"reservoir_esp_{tag}.csv")
    with open(raw, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[wrote] {raw}")

    # per-generation curves CSV
    hcsv = os.path.join(OUT, f"reservoir_esp_{tag}_curves.csv")
    with open(hcsv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_in", "temp", "generation", "mean_reservoir_hamming"])
        for (temp, n_in), c in curves.items():
            for g, v in enumerate(c):
                w.writerow([n_in, temp, g, v])
    print(f"[wrote] {hcsv}")

    _plot_esp(curves, temps, n_ins, os.path.join(OUT, f"reservoir_esp_{tag}.png"), args)


def _plot_esp(curves, temps, n_ins, png, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(n_ins), figsize=(5 * len(n_ins), 4.2), squeeze=False)
    for j, n_in in enumerate(n_ins):
        ax = axes[0][j]
        for temp in temps:
            c = curves[(temp, n_in)]
            ax.semilogy(np.arange(len(c)), np.clip(c, 1e-1, None), lw=1.4, label=f"T={temp}")
        ax.set_title(f"injection n_in={n_in}")
        ax.set_xlabel("generation")
        ax.set_ylabel("reservoir Hamming (replica A vs B)")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    fig.suptitle(f"Driven echo-state property — {args.model}, L={args.length}\n"
                 "different inits, same input + same fixed noise; ESP <=> Hamming -> 0")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(png, dpi=140)
    plt.close(fig)
    print(f"[wrote] {png}")


# --------------------------------------------------------------------------- #
# #34  Memory Capacity + baselines
# --------------------------------------------------------------------------- #
def effective_rank(X: np.ndarray) -> int:
    """Numerical rank of the centered feature matrix -- the true ceiling on
    linear MC (<= readout_dim, often far below it for a contracting reservoir)."""
    Xc = X - X.mean(axis=0, keepdims=True)
    return int(np.linalg.matrix_rank(Xc))


def cmd_mc(args):
    auto, tok, model, dev, vocab, dead, emb = build_causal(args.model, args.device)
    L = args.length
    table = rz.pca_readout_table(emb, args.k)
    T = args.steps
    temps = [float(x) for x in str(args.temp).split(",")]
    sp = rz.make_split(T, args.washout, args.train, args.val, args.test)
    kmax = args.kmax
    u = iid_input(T, seed=7)
    noises = rz.make_noise(T, L, vocab, dev, seed=args.noise_seed)
    rng = np.random.default_rng(0)
    perm = rng.permutation(T)

    # build a reservoir once to get geometry (temp set per-run below)
    res = make_reservoir(auto, L, vocab, emb, dead, args.n_in, args.n_bins,
                         temps[0], dev, codebook_mode=args.codebook,
                         absorbing=args.absorbing)
    nres = len(res.reservoir_sites)
    readout_dim = nres * args.k
    print(f"[info] model={args.model} dev={dev} L={L} n_in={args.n_in} nres={nres} "
          f"n_bins={res.n_bins} K={args.k} readout_dim={readout_dim} T={T} temps={temps}")

    # temp-INDEPENDENT baselines (computed once)
    rand_states = rng.integers(0, vocab, size=(T, L))
    Xr = res.features(rand_states, table)
    Xrz, = rz.standardize(Xr[sp.train], Xr)
    base_random = rz.memory_capacity(Xrz, u, sp, kmax=kmax)
    esn = linear_esn_features(u, readout_dim, spectral_radius=args.esn_rho, seed=5)
    Xez, = rz.standardize(esn[sp.train], esn)
    base_esn = rz.memory_capacity(Xez, u, sp, kmax=kmax)
    print(f"[baseline] linear_esn(dim={readout_dim}, rho={args.esn_rho}) MC={base_esn['MC']:.2f}  "
          f"random_token MC={base_random['MC']:.3f}")

    os.makedirs(OUT, exist_ok=True)
    all_rows = []
    per_temp = {}      # temp -> {"reservoir":..., "shuffled_input":...}
    summary = []       # one row per temp for the MC-vs-T plot
    for temp in temps:
        res.temp = temp
        init = res.random_init(seed=123)
        t0 = time.time()
        states = res.run(u, init, noises)
        X = res.features(states, table)
        Xz, = rz.standardize(X[sp.train], X)
        r_res = rz.memory_capacity(Xz, u, sp, kmax=kmax)
        r_shuf = rz.memory_capacity(Xz, u[perm], sp, kmax=kmax)
        erank = effective_rank(X[sp.train])
        # INPUT-LEAK CONTROL (lead req #1): MC read from ONLY the clamped input
        # sites. The headline readout excludes them; this quantifies how much
        # "capacity" is just the injected symbol copied off its own sites
        # (expected: concentrated at lag 0, ~0 at lag>=1 since input sites are
        # overwritten each step). It is the leak we are NOT counting.
        Xin = res.features(states, table, sites="input")
        Xinz, = rz.standardize(Xin[sp.train], Xin)
        r_leak = rz.memory_capacity(Xinz, u, sp, kmax=kmax)
        # capacity vs distance-from-input: restrict readout to the first m reservoir
        # sites (closest to the left-edge input). Rules out "MC is just the site
        # adjacent to the clamp copying the input."
        dist_mc = {}
        for m in args.dist_sites:
            if m <= nres:
                cols = np.arange(m * args.k)
                dist_mc[m] = rz.memory_capacity(Xz[:, cols], u, sp, kmax=kmax)["MC"]
        per_temp[temp] = {"reservoir": r_res, "shuffled_input": r_shuf}
        summary.append({
            "temp": temp, "MC": r_res["MC"], "MC_shuffled": r_shuf["MC"],
            "MC_net": r_res["MC"] - r_shuf["MC"],
            "MC_inputleak": r_leak["MC"], "MC_inputleak_0": r_leak["mc_k"][0],
            "MC_inputleak_1": r_leak["mc_k"][1], "eff_rank": erank,
            "readout_dim": readout_dim,
            **{f"MC_first{m}": dist_mc.get(m, "") for m in args.dist_sites},
        })
        for name, r in (("reservoir", r_res), ("shuffled_input", r_shuf)):
            for k in range(kmax + 1):
                all_rows.append({"temp": temp, "system": name, "k": k,
                                 "mc_k": r["mc_k"][k], "alpha_k": r["alpha_k"][k],
                                 "MC": r["MC"], "readout_dim": readout_dim,
                                 "eff_rank": erank})
        dist_str = " ".join(f"d{m}={dist_mc[m]:.2f}" for m in args.dist_sites if m in dist_mc)
        print(f"  T={temp:<4} MC={r_res['MC']:.2f} (shuffled {r_shuf['MC']:.2f}, "
              f"net {r_res['MC']-r_shuf['MC']:.2f})  eff_rank={erank}/{readout_dim}  "
              f"MC_0={r_res['mc_k'][0]:.2f} MC_1={r_res['mc_k'][1]:.2f}  "
              f"inputleak(MC={r_leak['MC']:.2f},k0={r_leak['mc_k'][0]:.2f},k1={r_leak['mc_k'][1]:.2f})  "
              f"[{dist_str}]  ({time.time()-t0:.0f}s)", flush=True)

    # add the temp-independent baselines to the per-k table
    for name, r in (("linear_esn", base_esn), ("random_token", base_random)):
        for k in range(kmax + 1):
            all_rows.append({"temp": "", "system": name, "k": k, "mc_k": r["mc_k"][k],
                             "alpha_k": r["alpha_k"][k], "MC": r["MC"],
                             "readout_dim": readout_dim, "eff_rank": ""})

    tag = f"{rz_model_tag(args.model)}_L{L}_nin{args.n_in}_K{args.k}"
    csv_path = os.path.join(OUT, f"reservoir_mc_{tag}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["temp", "system", "k", "mc_k", "alpha_k",
                                          "MC", "readout_dim", "eff_rank"])
        w.writeheader(); w.writerows(all_rows)
    print(f"[wrote] {csv_path}")
    scsv = os.path.join(OUT, f"reservoir_mc_{tag}_summary.csv")
    with open(scsv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader(); w.writerows(summary)
    print(f"[wrote] {scsv}")

    print(f"\n[ceiling] linear MC <= effective readout rank <= readout_dim={readout_dim}")
    print(f"[calibration] linear_esn MC={base_esn['MC']:.1f}  random_token MC={base_random['MC']:.2f}")

    # pick the best temp for the forgetting-curve panel
    best = max(summary, key=lambda r: r["MC_net"])
    best_results = {
        "reservoir": per_temp[best["temp"]]["reservoir"],
        "linear_esn": base_esn,
        "random_token": base_random,
        "shuffled_input": per_temp[best["temp"]]["shuffled_input"],
    }
    args._best_temp = best["temp"]
    _plot_mc_sweep(summary, best_results, temps, readout_dim, args,
                   os.path.join(OUT, f"reservoir_mc_{tag}.png"))


def linear_esn_features(u, dim, spectral_radius=0.9, seed=0):
    """Textbook linear echo-state network: x(t) = W x(t-1) + w_in u(t), with W a
    random matrix scaled to ``spectral_radius`` < 1 (echo-state condition). Used
    as a calibration baseline -- its linear Memory Capacity is known to approach
    the reservoir dimension."""
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((dim, dim))
    eig = np.max(np.abs(np.linalg.eigvals(W)))
    W = W / eig * spectral_radius
    w_in = rng.uniform(-1, 1, size=dim)
    T = len(u)
    X = np.zeros((T, dim))
    x = np.zeros(dim)
    for t in range(T):
        x = np.tanh(W @ x + w_in * u[t])
        X[t] = x
    return X


def _plot_mc_sweep(summary, best_results, temps, readout_dim, args, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"reservoir": "tab:purple", "linear_esn": "tab:blue",
              "random_token": "0.6", "shuffled_input": "tab:red"}
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(16, 4.6))

    # panel 0: MC vs temperature (net of shuffled-input floor)
    ts = [r["temp"] for r in summary]
    ax0.plot(ts, [r["MC"] for r in summary], "o-", color="tab:purple", label="MC (raw)")
    ax0.plot(ts, [r["MC_shuffled"] for r in summary], "s--", color="tab:red",
             label="shuffled-input floor")
    ax0.plot(ts, [r["MC_net"] for r in summary], "^-", color="tab:green", label="MC net")
    ax0.set_xlabel("temperature T")
    ax0.set_ylabel("total linear MC")
    ax0.set_title("memory capacity vs temperature")
    ax0.grid(alpha=0.3); ax0.legend(fontsize=8)

    # panel 1: forgetting curve at the best temp + baselines
    for name, r in best_results.items():
        k = np.arange(len(r["mc_k"]))
        ax1.plot(k, r["mc_k"], marker="o", ms=3, lw=1.3, color=colors.get(name),
                 label=f"{name} (MC={r['MC']:.1f})")
    ax1.set_xlabel("lag k")
    ax1.set_ylabel("MC_k  (test R²)")
    ax1.set_title(f"forgetting curve (T={args._best_temp})")
    ax1.grid(alpha=0.3); ax1.legend(fontsize=7)

    # panel 2: total MC vs the readout-rank ceiling
    names = list(best_results.keys())
    mcs = [best_results[n]["MC"] for n in names]
    ax2.bar(range(len(names)), mcs, color=[colors.get(n) for n in names])
    erank = summary[[r["temp"] for r in summary].index(args._best_temp)]["eff_rank"]
    ax2.axhline(readout_dim, color="k", ls="--", lw=1, label=f"readout_dim={readout_dim}")
    ax2.axhline(erank, color="tab:gray", ls=":", lw=1.2, label=f"eff_rank={erank}")
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax2.set_ylabel("total MC")
    ax2.set_title(f"MC vs readout ceiling (T={args._best_temp})")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3, axis="y")

    fig.suptitle(f"Memory Capacity — {args.model} causal reservoir, L={args.length}, "
                 f"n_in={args.n_in}, K={args.k}")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(png, dpi=140)
    plt.close(fig)
    print(f"[wrote] {png}")


# --------------------------------------------------------------------------- #
# #36  Information Processing Capacity (Dambre) + degree decomposition
# --------------------------------------------------------------------------- #
def _ipc_one(name, X, u, sp, args):
    Xz, = rz.standardize(X[sp.train], X)
    r = rz.information_processing_capacity(
        Xz, u, sp, max_degree=args.max_degree, max_delay=args.max_delay,
        max_vars=args.max_vars, n_surrogate=args.n_surrogate)
    deg = r["per_degree"]
    degstr = " ".join(f"d{g}={deg.get(g,0.0):.2f}" for g in range(1, args.max_degree + 1))
    print(f"  {name:<16} IPC_total={r['total']:.2f}  [{degstr}]  thr={r['threshold']:.3f} "
          f"readout_dim={r['readout_dim']}", flush=True)
    return r


def cmd_ipc(args):
    # calibration first: shift-register IPC must be degree-1 ~ N, degree>=2 ~ 0
    if args.calibrate:
        print("[calibrate] literal shift register (degree-1 should ~ N, higher ~ 0):")
        T = args.steps
        u = iid_input(T, seed=7)
        N = args.cal_taps
        Xsr = np.zeros((T, N))
        for k in range(N):
            Xsr[k:, k] = u[: T - k]
        sp = rz.make_split(T, args.washout, args.train, args.val, args.test)
        _ipc_one(f"shift_reg(N={N})", Xsr, u, sp, args)

    auto, tok, model, dev, vocab, dead, emb = build_causal(args.model, args.device)
    L = args.length
    table = rz.pca_readout_table(emb, args.k)
    T = args.steps
    temps = [float(x) for x in str(args.temp).split(",")]
    sp = rz.make_split(T, args.washout, args.train, args.val, args.test)
    u = iid_input(T, seed=7)
    noises = rz.make_noise(T, L, vocab, dev, seed=args.noise_seed)
    res = make_reservoir(auto, L, vocab, emb, dead, args.n_in, args.n_bins,
                         temps[0], dev, codebook_mode=args.codebook, absorbing=args.absorbing)
    nres = len(res.reservoir_sites)
    readout_dim = nres * args.k
    print(f"[info] IPC: model={args.model} L={L} n_in={args.n_in} nres={nres} K={args.k} "
          f"readout_dim={readout_dim} max_degree={args.max_degree} max_delay={args.max_delay} "
          f"max_vars={args.max_vars} temps={temps}")

    # baselines
    rng = np.random.default_rng(0)
    print("[baselines]")
    rand_states = rng.integers(0, vocab, size=(T, L))
    base_random = _ipc_one("random_token", res.features(rand_states, table), u, sp, args)
    esn = linear_esn_features(u, readout_dim, spectral_radius=args.esn_rho, seed=5)
    base_esn = _ipc_one("linear_esn", esn, u, sp, args)

    all_rows = []
    summary = []
    print("[reservoir by temperature]")
    res_by_temp = {}
    for temp in temps:
        res.temp = temp
        init = res.random_init(seed=123)
        t0 = time.time()
        states = res.run(u, init, noises)
        X = res.features(states, table)
        r = _ipc_one(f"reservoir T={temp}", X, u, sp, args)
        res_by_temp[temp] = r
        deg = r["per_degree"]
        summary.append({"temp": temp, "IPC_total": r["total"], "threshold": r["threshold"],
                        "readout_dim": readout_dim,
                        **{f"deg{g}": deg.get(g, 0.0) for g in range(1, args.max_degree + 1)}})
        for row in r["rows"]:
            all_rows.append({"system": f"reservoir_T{temp}", "degree": row["degree"],
                             "config": str(row["config"]), "capacity": row["capacity"],
                             "capacity_thr": row["capacity_thr"]})
        print(f"    ({time.time()-t0:.0f}s)")

    # write
    os.makedirs(OUT, exist_ok=True)
    tag = f"{rz_model_tag(args.model)}_L{L}_nin{args.n_in}_K{args.k}_d{args.max_degree}"
    scsv = os.path.join(OUT, f"reservoir_ipc_{tag}_summary.csv")
    # add baselines to summary
    for nm, r in (("linear_esn", base_esn), ("random_token", base_random)):
        deg = r["per_degree"]
        summary.append({"temp": nm, "IPC_total": r["total"], "threshold": r["threshold"],
                        "readout_dim": readout_dim,
                        **{f"deg{g}": deg.get(g, 0.0) for g in range(1, args.max_degree + 1)}})
    with open(scsv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader(); w.writerows(summary)
    print(f"[wrote] {scsv}")
    ccsv = os.path.join(OUT, f"reservoir_ipc_{tag}_configs.csv")
    with open(ccsv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["system", "degree", "config", "capacity", "capacity_thr"])
        w.writeheader(); w.writerows(all_rows)
    print(f"[wrote] {ccsv}")

    _plot_ipc(summary, res_by_temp, base_esn, base_random, temps, args,
              os.path.join(OUT, f"reservoir_ipc_{tag}.png"))


def _plot_ipc(summary, res_by_temp, base_esn, base_random, temps, args, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    degrees = list(range(1, args.max_degree + 1))
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 4.8))

    # panel 0: stacked degree decomposition vs temperature (+ esn baseline bar)
    width = 0.6
    bottoms = np.zeros(len(temps))
    cmap = plt.get_cmap("viridis")
    for i, g in enumerate(degrees):
        vals = np.array([res_by_temp[t]["per_degree"].get(g, 0.0) for t in temps])
        ax0.bar(range(len(temps)), vals, width, bottom=bottoms,
                color=cmap(i / max(1, len(degrees) - 1)), label=f"degree {g}")
        bottoms += vals
    ax0.set_xticks(range(len(temps)))
    ax0.set_xticklabels([f"T={t}" for t in temps], fontsize=8)
    ax0.set_ylabel("IPC (sum of thresholded R²)")
    ax0.set_title("IPC degree decomposition — LLM reservoir")
    ax0.legend(fontsize=8); ax0.grid(alpha=0.3, axis="y")

    # panel 1: degree profile, LLM (best temp) vs linear_esn baseline
    best_t = max(temps, key=lambda t: res_by_temp[t]["total"])
    llm = [res_by_temp[best_t]["per_degree"].get(g, 0.0) for g in degrees]
    esn = [base_esn["per_degree"].get(g, 0.0) for g in degrees]
    x = np.arange(len(degrees))
    ax1.bar(x - 0.2, llm, 0.4, color="tab:purple", label=f"LLM reservoir (T={best_t})")
    ax1.bar(x + 0.2, esn, 0.4, color="tab:blue", label="linear_esn")
    ax1.set_xticks(x); ax1.set_xticklabels([f"deg {g}" for g in degrees])
    ax1.set_ylabel("capacity")
    ax1.set_title("linear (deg 1) vs nonlinear (deg ≥ 2) processing")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3, axis="y")

    fig.suptitle(f"Information Processing Capacity (Dambre) — {args.model} causal reservoir, "
                 f"L={args.length}, n_in={args.n_in}")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(png, dpi=140)
    plt.close(fig)
    print(f"[wrote] {png}")


def rz_model_tag(name: str) -> str:
    base = name.rsplit("/", 1)[-1]
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in base)


# --------------------------------------------------------------------------- #
def main():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", default=DEFAULT_MODEL)
    common.add_argument("--device", default="mps")
    common.add_argument("--length", type=int, default=48)
    common.add_argument("--codebook", default="pc1", choices=["pc1", "random", "evenly"])
    common.add_argument("--n-bins", type=int, default=16, dest="n_bins")
    common.add_argument("--absorbing", action="store_true")

    p = argparse.ArgumentParser(description=__doc__, parents=[common])
    sub = p.add_subparsers(dest="cmd", required=True)

    ap = sub.add_parser("apparatus", parents=[common], help="build + sanity-check the reservoir (#32)")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--n-in", type=int, default=2, dest="n_in")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--washout", type=int, default=100)
    ap.add_argument("--kmax", type=int, default=15)
    ap.add_argument("--noise-seed", type=int, default=777, dest="noise_seed")
    ap.set_defaults(func=cmd_apparatus)

    es = sub.add_parser("esp", parents=[common], help="driven echo-state property (#33)")
    es.add_argument("--steps", type=int, default=150)
    es.add_argument("--pairs", type=int, default=4)
    es.add_argument("--temps", default="0.5,0.7,0.9,1.1")
    es.add_argument("--n-ins", default="1,2,3", dest="n_ins")
    es.add_argument("--tail", type=int, default=10)
    es.set_defaults(func=cmd_esp)

    mc = sub.add_parser("mc", parents=[common], help="memory capacity + baselines (#34)")
    mc.add_argument("--steps", type=int, default=2200)
    mc.add_argument("--n-in", type=int, default=2, dest="n_in")
    mc.add_argument("--k", type=int, default=8)
    mc.add_argument("--temp", default="0.7", help="single temp or comma list to sweep")
    mc.add_argument("--washout", type=int, default=200)
    mc.add_argument("--train", type=int, default=1200)
    mc.add_argument("--val", type=int, default=300)
    mc.add_argument("--test", type=int, default=500)
    mc.add_argument("--kmax", type=int, default=40)
    mc.add_argument("--esn-rho", type=float, default=0.95, dest="esn_rho")
    mc.add_argument("--noise-seed", type=int, default=777, dest="noise_seed")
    mc.add_argument("--dist-sites", type=int, nargs="+", default=[1, 4, 12],
                    dest="dist_sites", help="capacity-vs-distance: first-m reservoir sites")
    mc.set_defaults(func=cmd_mc)

    ip = sub.add_parser("ipc", parents=[common], help="Dambre IPC + degree decomposition (#36)")
    ip.add_argument("--steps", type=int, default=3000)
    ip.add_argument("--n-in", type=int, default=2, dest="n_in")
    ip.add_argument("--k", type=int, default=8)
    ip.add_argument("--temp", default="0.3", help="single temp or comma list")
    ip.add_argument("--washout", type=int, default=200)
    ip.add_argument("--train", type=int, default=1800)
    ip.add_argument("--val", type=int, default=400)
    ip.add_argument("--test", type=int, default=600)
    ip.add_argument("--max-degree", type=int, default=4, dest="max_degree")
    ip.add_argument("--max-delay", type=int, default=8, dest="max_delay")
    ip.add_argument("--max-vars", type=int, default=2, dest="max_vars")
    ip.add_argument("--n-surrogate", type=int, default=48, dest="n_surrogate")
    ip.add_argument("--esn-rho", type=float, default=0.95, dest="esn_rho")
    ip.add_argument("--noise-seed", type=int, default=777, dest="noise_seed")
    ip.add_argument("--calibrate", action="store_true",
                    help="run shift-register IPC calibration first")
    ip.add_argument("--cal-taps", type=int, default=20, dest="cal_taps")
    ip.set_defaults(func=cmd_ipc)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
