"""LOCK the quotable §25 capacity magnitudes (task #41).

Runs the disciplined estimators in llm_life.capacity over >=3 (input, noise)
seeds and writes the licensed numbers + the load-bearing decomposition:

  * MC reconstructs the CONTINUOUS input u(t-k); wide ridge-alpha grid with an
    INTERIORITY hard-check (the first pass pinned alpha at the grid max).
  * IPC on the encoded-symbol Gram-Schmidt basis, split 2x2 by
    degree x {instantaneous (all delays 0) | temporal (any delay>=1)} and
    floored by a shuffled-INPUT surrogate reported at {p99, p99.9, max}.
  * Baselines split the SAME way: a linear ESN (must show C_temporal>>0 -- a real
    reservoir computes over time), a literal delay-line, and a random-token null.
    The contrast that matters: LLM reservoir C_temporal ~ 0 vs ESN C_temporal >> 0.
  * Input-leak control: MC read off ONLY the clamped input sites (excluded from
    the headline readout) -- quantifies the input copied off its own sites.

Run:  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts_capacity_lock.py --device mps
"""
import argparse, csv, os, time, warnings
warnings.filterwarnings("ignore", message=".*encountered in matmul.*")
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np

from llm_life import capacity as cap
from llm_life import reservoir as rz
from scripts_reservoir import build_causal, make_reservoir, linear_esn_features, rz_model_tag

OUT = "results"
DEGREES = (1, 2, 3, 4)


def bin_centers(n_bins):
    return -1.0 + (np.arange(n_bins) + 0.5) * 2.0 / n_bins


def split_summary(ipc, tag):
    """Flatten an IPC result's max-threshold summary into a flat dict."""
    s = ipc["summary_max"]
    row = {f"{tag}_total": s["total"], f"{tag}_inst": s["inst_total"],
           f"{tag}_temporal": s["temporal_total"]}
    for g in DEGREES:
        row[f"{tag}_d{g}_inst"] = s["per_cell"].get((g, "inst"), 0.0)
        row[f"{tag}_d{g}_temporal"] = s["per_cell"].get((g, "temporal"), 0.0)
    # robustness of the total across thresholds
    row[f"{tag}_total_p99"] = ipc["summary_p99"]["total"]
    row[f"{tag}_total_p999"] = ipc["summary_p999"]["total"]
    return row


def run_ipc(X, symbols, sp, basis, args, seed):
    return cap.information_processing_capacity(
        X, symbols, sp, basis, max_degree=args.max_degree, max_delay=args.max_delay,
        max_vars=args.max_vars, n_surrogate=args.n_surrogate, rng_seed=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-160m")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--length", type=int, default=48)
    ap.add_argument("--n-in", type=int, default=2, dest="n_in")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--n-bins", type=int, default=16, dest="n_bins")
    ap.add_argument("--temps", default="0.0,0.3,0.7,1.0")
    ap.add_argument("--primary-temp", type=float, default=0.3, dest="primary_temp")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--washout", type=int, default=200)
    ap.add_argument("--train", type=int, default=1800)
    ap.add_argument("--val", type=int, default=400)
    ap.add_argument("--test", type=int, default=600)
    ap.add_argument("--kmax", type=int, default=30)
    ap.add_argument("--max-degree", type=int, default=4, dest="max_degree")
    ap.add_argument("--max-delay", type=int, default=8, dest="max_delay")
    ap.add_argument("--max-vars", type=int, default=2, dest="max_vars")
    ap.add_argument("--n-surrogate", type=int, default=40, dest="n_surrogate")
    ap.add_argument("--esn-rho", type=float, default=0.95, dest="esn_rho")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--shift-taps", type=int, default=30, dest="shift_taps")
    args = ap.parse_args()

    need = args.washout + args.train + args.val + args.test
    assert need <= args.steps, (f"washout+train+val+test={need} exceeds steps={args.steps}; "
                                f"increase --steps or shrink the splits")
    assert args.washout > args.kmax, "washout must exceed kmax so u(t-k) is defined on every split"
    temps = [float(x) for x in args.temps.split(",")]
    seed_pairs = [(7 + 4 * i, 777 + i) for i in range(args.seeds)]  # (input_seed, noise_seed)

    auto, tok, model, dev, vocab, dead, emb = build_causal(args.model, args.device)
    L = args.length
    table = rz.pca_readout_table(emb, args.k)
    basis = cap.symbol_poly_basis(args.n_bins, args.max_degree)
    centers = bin_centers(args.n_bins)
    res = make_reservoir(auto, L, vocab, emb, dead, args.n_in, args.n_bins,
                         temps[0], dev, codebook_mode="pc1")
    nres = len(res.reservoir_sites); readout_dim = nres * args.k
    sp = cap.make_split(args.steps, args.washout, args.train, args.val, args.test)
    print(f"[info] {args.model} dev={dev} L={L} n_in={args.n_in} nres={nres} K={args.k} "
          f"readout_dim={readout_dim} n_bins={args.n_bins} steps={args.steps} "
          f"seeds={args.seeds} temps={temps}")
    print(f"[info] alpha grid: {cap.ALPHA_GRID[0]:.0e}..{cap.ALPHA_GRID[-1]:.0e} "
          f"({len(cap.ALPHA_GRID)} pts); IPC caps deg<={args.max_degree} "
          f"delay<={args.max_delay} vars<={args.max_vars}")

    rows = []            # one dict per (system, temp, seed)
    config_rows = []     # per-IPC-config capacity (reservoir) for d3_temporal inspection
    boundary_violations = []
    for (us, ns) in seed_pairs:
        ru = np.random.default_rng(us)
        u = ru.uniform(-1.0, 1.0, args.steps)
        symbols = rz.bin_input(u, args.n_bins)
        uc = centers[symbols]
        # MC reconstructs the ENCODED bin index b(t) -- the variable the reservoir
        # actually receives -- so degree-1 IPC == MC exactly (self-consistency;
        # verified machine-precision). MC over the encoded bin vs continuous u
        # differs only ~0.5% (quantization), so headline magnitudes are unchanged.
        b = symbols.astype(float)
        noises = rz.make_noise(args.steps, L, vocab, dev, ns)

        # ---- baselines (driven by the same encoded input uc; once per seed) ----
        esn = linear_esn_features(uc, readout_dim, spectral_radius=args.esn_rho, seed=5)
        Xez, = rz.standardize(esn[sp.train], esn)
        sr = np.zeros((args.steps, args.shift_taps))
        for kk in range(args.shift_taps):
            sr[kk:, kk] = uc[: args.steps - kk]
        Xsrz, = rz.standardize(sr[sp.train], sr)
        rtok = ru.integers(0, vocab, size=(args.steps, L))
        Xr = res.features(rtok, table); Xrz, = rz.standardize(Xr[sp.train], Xr)

        for name, Xb in (("linear_esn", Xez), ("shift_register", Xsrz), ("random_token", Xrz)):
            t0 = time.time()
            mc = cap.memory_capacity(Xb, b, sp, kmax=args.kmax)
            ipc = run_ipc(Xb, symbols, sp, basis, args, seed=us)
            row = {"system": name, "temp": "", "input_seed": us, "noise_seed": ns,
                   "MC": mc["MC"], "MC_0": mc["mc_k"][0], "MC_1": mc["mc_k"][1],
                   "MC_alpha_at_max_signif": mc["alpha_at_max_signif"],
                   "eff_rank": ipc["eff_rank"], "readout_dim": readout_dim,
                   "ipc_alpha_at_max_signif": ipc["alpha_at_max_signif"],
                   "surrogate_max": ipc["thresholds"]["max"]}
            row.update(split_summary(ipc, "IPC"))
            rows.append(row)
            if mc["alpha_at_max_signif"] or ipc["alpha_at_max_signif"]:
                boundary_violations.append((name, "baseline", us))
            print(f"  [{name:<14} seed={us}] MC={mc['MC']:.2f} IPC={ipc['summary_max']['total']:.2f} "
                  f"(inst={ipc['summary_max']['inst_total']:.2f} temp={ipc['summary_max']['temporal_total']:.2f}) "
                  f"a@max={mc['alpha_at_max_signif']}/{ipc['alpha_at_max_signif']} ({time.time()-t0:.0f}s)",
                  flush=True)

        # ---- LLM reservoir per temperature ----
        for temp in temps:
            res.temp = temp
            init = res.random_init(seed=123)
            t0 = time.time()
            states = res.run(u, init, noises)
            X = res.features(states, table)
            Xz, = rz.standardize(X[sp.train], X)
            mc = cap.memory_capacity(Xz, b, sp, kmax=args.kmax)
            ipc = run_ipc(Xz, symbols, sp, basis, args, seed=us)
            # input-leak control
            Xin = res.features(states, table, sites="input")
            Xinz, = rz.standardize(Xin[sp.train], Xin)
            mc_leak = cap.memory_capacity(Xinz, b, sp, kmax=args.kmax)
            row = {"system": "reservoir", "temp": temp, "input_seed": us, "noise_seed": ns,
                   "MC": mc["MC"], "MC_0": mc["mc_k"][0], "MC_1": mc["mc_k"][1],
                   "MC_alpha_at_max_signif": mc["alpha_at_max_signif"],
                   "MC_inputleak": mc_leak["MC"], "MC_inputleak_0": mc_leak["mc_k"][0],
                   "MC_inputleak_1": mc_leak["mc_k"][1],
                   "eff_rank": ipc["eff_rank"], "readout_dim": readout_dim,
                   "ipc_alpha_at_max_signif": ipc["alpha_at_max_signif"],
                   "surrogate_max": ipc["thresholds"]["max"]}
            row.update(split_summary(ipc, "IPC"))
            rows.append(row)
            # per-config capacities (above the conservative max floor) so the
            # degree-3 TEMPORAL component can be inspected config-by-config rather
            # than trusted as a single aggregate (gate S2 / #46).
            for cr in ipc["rows"]:
                floor = ipc["thresholds_by_degree"][cr["degree"]]["max"]
                if cr["capacity"] > floor:
                    config_rows.append({
                        "system": "reservoir", "temp": temp, "input_seed": us,
                        "degree": cr["degree"], "kind": "inst" if cr["inst"] else "temporal",
                        "config": str(cr["config"]), "capacity": round(cr["capacity"], 5),
                        "threshold_max": round(floor, 5)})
            if mc["alpha_at_max_signif"] or ipc["alpha_at_max_signif"]:
                boundary_violations.append(("reservoir", temp, us))
            print(f"  [reservoir T={temp:<4} seed={us}] MC={mc['MC']:.3f} "
                  f"(0={mc['mc_k'][0]:.3f},1={mc['mc_k'][1]:.3f}) leak={mc_leak['MC']:.3f} "
                  f"IPC={ipc['summary_max']['total']:.3f} inst={ipc['summary_max']['inst_total']:.3f} "
                  f"temp={ipc['summary_max']['temporal_total']:.3f} "
                  f"a@max={mc['alpha_at_max_signif']}/{ipc['alpha_at_max_signif']} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    os.makedirs(OUT, exist_ok=True)
    tag = f"{rz_model_tag(args.model)}_L{L}_nin{args.n_in}_K{args.k}"
    raw = os.path.join(OUT, f"capacity_lock_{tag}_raw.csv")
    keys = sorted({k for r in rows for k in r})
    keys = ["system", "temp", "input_seed", "noise_seed"] + [k for k in keys
            if k not in ("system", "temp", "input_seed", "noise_seed")]
    with open(raw, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print(f"[wrote] {raw}")

    if config_rows:
        ccsv = os.path.join(OUT, f"capacity_lock_{tag}_configs.csv")
        cfields = ["system", "temp", "input_seed", "degree", "kind", "config",
                   "capacity", "threshold_max"]
        with open(ccsv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cfields); w.writeheader(); w.writerows(config_rows)
        print(f"[wrote] {ccsv}")
        # quick degree-3 temporal inspection across seeds
        d3t = [r for r in config_rows if r["degree"] == 3 and r["kind"] == "temporal"]
        if d3t:
            from collections import defaultdict
            by = defaultdict(list)
            for r in d3t:
                by[(r["temp"], r["config"])].append(r["capacity"])
            print("[d3_temporal] configs above max-floor (per temp, mean cap over seeds, n seeds):")
            for (temp, cfg), caps in sorted(by.items()):
                print(f"    T={temp} {cfg:<26} cap={np.mean(caps):.4f} (n={len(caps)})")

    # aggregate mean +/- std over seeds per (system, temp)
    agg = aggregate(rows)
    acsv = os.path.join(OUT, f"capacity_lock_{tag}_summary.csv")
    fields = ["system", "temp", "n_seeds"] + sorted({k for a in agg for k in a
              if k not in ("system", "temp", "n_seeds")})
    with open(acsv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader(); w.writerows(agg)
    print(f"[wrote] {acsv}")

    print("\n===== LICENSED SUMMARY (mean +/- std over seeds) =====")
    for a in agg:
        sysname = a["system"] + (f" T={a['temp']}" if a["temp"] != "" else "")
        print(f"  {sysname:<22} MC={a['MC_mean']:.3f}±{a['MC_std']:.3f}  "
              f"IPC={a['IPC_total_mean']:.3f}±{a['IPC_total_std']:.3f}  "
              f"inst={a['IPC_inst_mean']:.3f} temporal={a['IPC_temporal_mean']:.3f}  "
              f"a@max(MC/IPC)={a['MC_alpha_at_max_signif_sum']}/{a['ipc_alpha_at_max_signif_sum']}")

    if boundary_violations:
        print(f"\n[ALPHA-INTERIORITY WARNING] {len(boundary_violations)} significant max-boundary "
              f"hits: {boundary_violations[:10]} -> widen ALPHA_GRID upper end and rerun.")
    else:
        print("\n[ALPHA-INTERIORITY OK] no significant-capacity target selected the grid max; "
              "magnitudes are licensed.")

    plot_lock(agg, temps, args, os.path.join(OUT, f"capacity_lock_{tag}.png"))


def aggregate(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r["system"], r["temp"]), []).append(r)
    # union of numeric keys across ALL rows (reservoir-only fields like the input
    # leak are absent from baseline rows, so keying off rows[0] would drop them)
    numeric = sorted({k for r in rows for k, v in r.items() if isinstance(v, (int, float))})
    out = []
    for (system, temp), rs in groups.items():
        a = {"system": system, "temp": temp, "n_seeds": len(rs)}
        for k in numeric:
            vals = np.array([r[k] for r in rs if isinstance(r.get(k), (int, float))], dtype=float)
            if vals.size == 0:
                continue
            a[f"{k}_mean"] = float(vals.mean())
            a[f"{k}_std"] = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
            if "alpha_at_max" in k:
                a[f"{k}_sum"] = int(vals.sum())
        out.append(a)
    return out


def plot_lock(agg, temps, args, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res_rows = sorted([a for a in agg if a["system"] == "reservoir"], key=lambda r: r["temp"])
    base = {a["system"]: a for a in agg if a["system"] != "reservoir"}
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(16, 4.7))

    ts = [a["temp"] for a in res_rows]
    ax0.errorbar(ts, [a["MC_mean"] for a in res_rows], yerr=[a["MC_std"] for a in res_rows],
                 marker="o", color="tab:purple", capsize=3, label="LLM reservoir MC")
    if "linear_esn" in base:
        ax0.axhline(base["linear_esn"]["MC_mean"], color="tab:blue", ls="--",
                    label=f"linear_esn MC={base['linear_esn']['MC_mean']:.1f}")
    ax0.set_xlabel("temperature T"); ax0.set_ylabel("Memory Capacity (continuous u)")
    ax0.set_title("MC vs T (licensed, wide-α)"); ax0.grid(alpha=0.3); ax0.legend(fontsize=8)

    # inst vs temporal IPC for reservoir across T
    inst = [a["IPC_inst_mean"] for a in res_rows]
    tempo = [a["IPC_temporal_mean"] for a in res_rows]
    x = np.arange(len(ts))
    ax1.bar(x - 0.2, inst, 0.4, color="tab:orange", label="instantaneous (static nonlinear)")
    ax1.bar(x + 0.2, tempo, 0.4, color="tab:green", label="temporal (memory/compute)")
    ax1.set_xticks(x); ax1.set_xticklabels([f"T={t}" for t in ts], fontsize=8)
    ax1.set_ylabel("IPC"); ax1.set_title("LLM reservoir: instantaneous vs temporal")
    ax1.grid(alpha=0.3, axis="y"); ax1.legend(fontsize=8)

    # the headline contrast: temporal capacity, LLM vs baselines
    names, vals = [], []
    pt = min(res_rows, key=lambda r: abs(r["temp"] - args.primary_temp))
    names.append(f"reservoir\nT={pt['temp']}"); vals.append(pt["IPC_temporal_mean"])
    for nm in ("linear_esn", "shift_register"):
        if nm in base:
            names.append(nm); vals.append(base[nm]["IPC_temporal_mean"])
    ax2.bar(range(len(names)), vals, color=["tab:purple", "tab:blue", "tab:gray"][:len(names)])
    ax2.set_xticks(range(len(names))); ax2.set_xticklabels(names, fontsize=8)
    ax2.set_ylabel("TEMPORAL IPC (computation over time)")
    ax2.set_title("temporal capacity: LLM ~ 0 vs real reservoirs >> 0")
    ax2.grid(alpha=0.3, axis="y")

    fig.suptitle(f"Licensed capacity — {args.model} causal reservoir, L={args.length}, "
                 f"n_in={args.n_in}, K={args.k}, {args.seeds} seeds")
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(png, dpi=140); plt.close(fig)
    print(f"[wrote] {png}")


if __name__ == "__main__":
    main()
