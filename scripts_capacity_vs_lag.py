"""HEADLINE TEST — capacity-vs-lag stratified by degree (task #48).

The discrepancy worth everything: initial-condition memory persists ~t_sync
(≈40-87 steps) under shared noise, yet linear input correlation hits the noise
floor by lag ~2. Hypothesis: information about inputs at lags 2…t_sync is in the
state but encoded NONLINEARLY. The resolving measurement is single-variable IPC
capacity of P_d(u(t-k)) at every lag k, per degree d — does degree>=2 capacity
persist to lags where linear (d=1) has already decayed?

  YES  -> "the map remembers its input nonlinearly beyond its linear horizon".
  NO   -> "strongly contracting, low total capacity: a consistent map but a poor
           reservoir" (degree>=2 also dies by ~lag 2).

Both are real, publishable characterizations — we are LOCATING this map on the
memory x nonlinearity plane, not chasing a big number.

Also emits two checks the lead asked for BEFORE any drive redesign:
  (a) low-T input-perturbation-decay: flip one input symbol, same init+noise,
      watch the reservoir Hamming response — confirms the input still PERTURBS the
      state (not frozen) and measures the STATE-SPACE memory horizon (how long an
      input perturbation survives in the lattice, regardless of readout).
  (b) lag-1 capacity vs site-distance from the clamp — is the weak memory just the
      sites adjacent to the input clamp, or distributed?

Run: HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts_capacity_vs_lag.py --device mps
"""
import argparse, csv, os, time, warnings
warnings.filterwarnings("ignore", message=".*encountered in matmul.*")
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np

from llm_life import capacity as cap
from llm_life import reservoir as rz
from scripts_reservoir import build_causal, make_reservoir

OUT = "results"
DEGREES = [1, 2, 3]


def single_var_target(symbols, basis, degree, lag):
    """P_degree(symbol(t-lag)) as a (T,) target (unit variance, zero mean)."""
    T = len(symbols)
    sh = np.zeros(T, dtype=np.int64)
    sh[lag:] = symbols[: T - lag]
    y = basis[sh, degree].copy()
    y[:lag] = 0.0
    return y


def degree_floor(rp, symbols, basis, degree, kmax, n_shuffle, rng):
    """Conservative by-chance floor for a single-var degree-d target: shuffle the
    symbol stream and reconstruct P_d(shuffled(t-k)) for a few lags; take the max."""
    nulls = []
    lags = [k for k in (1, 2, 4, 8, kmax) if k <= kmax]
    for _ in range(n_shuffle):
        ss = rng.permutation(symbols)
        for k in lags:
            nulls.append(rp.capacity(single_var_target(ss, basis, degree, k))["capacity"])
    return float(np.max(nulls))


def input_perturbation_decay(res, u, base_states, init, noises, t0_list, horizon=40):
    """Flip one input symbol at t0, same init+noise, measure reservoir Hamming
    response vs steps-after-perturbation. Reuses the unperturbed ``base_states``
    and truncates each perturbed run to t0+horizon (cheap)."""
    curves = []
    for t0 in t0_list:
        end = t0 + horizon
        u2 = u[:end].copy()
        b0 = rz.bin_input(np.array([u[t0]]), res.n_bins)[0]
        b1 = (b0 + res.n_bins // 2) % res.n_bins          # move to the opposite bin
        u2[t0] = -1.0 + (b1 + 0.5) * 2.0 / res.n_bins
        s2 = res.run(u2, init.clone(), noises[:end])
        ham = (base_states[:end, res.reservoir_sites] != s2[:, res.reservoir_sites]).sum(axis=1)
        curves.append(ham[t0:t0 + horizon].astype(float))
    mean = np.vstack(curves).mean(axis=0)
    above = np.where(mean > 0.5)[0]                        # last offset with >0.5 site response
    horizon_steps = int(above[-1]) if above.size else 0
    return mean, horizon_steps, float(mean.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--temps", default="0.3,0.0")
    ap.add_argument("--length", type=int, default=48)
    ap.add_argument("--n-in", type=int, default=2, dest="n_in")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--n-bins", type=int, default=16, dest="n_bins")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--kmax", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-shuffle", type=int, default=40, dest="n_shuffle")
    args = ap.parse_args()

    temps = [float(x) for x in args.temps.split(",")]
    seed_pairs = [(7 + 4 * i, 777 + i) for i in range(args.seeds)]
    auto, tok, model, dev, vocab, dead, emb = build_causal("EleutherAI/pythia-160m", args.device)
    L = args.length
    table = rz.pca_readout_table(emb, args.k)
    basis = cap.symbol_poly_basis(args.n_bins, max(DEGREES))
    sp = cap.make_split(args.steps, 200, 1800, 400, 600)
    # split adapts to --steps: washout fixed, then 60/15/25 train/val/test of the rest
    washout = 200
    rest = args.steps - washout
    train, val = int(0.60 * rest), int(0.15 * rest)
    test = rest - train - val
    sp = cap.make_split(args.steps, washout, train, val, test)
    res = make_reservoir(auto, L, vocab, emb, dead, args.n_in, args.n_bins,
                         temps[0], dev, codebook_mode="pc1")
    nres = len(res.reservoir_sites)
    print(f"[info] L={L} n_in={args.n_in} nres={nres} K={args.k} kmax={args.kmax} "
          f"seeds={args.seeds} temps={temps}")

    rows = []           # capacity-vs-lag per (temp,degree,lag): mean/std + floor
    decay_rows = []     # input-perturbation-decay curves
    dist_rows = []      # lag-1 capacity vs site-distance
    for temp in temps:
        res.temp = temp
        # per-seed arrays
        cap_dk = {d: {k: [] for k in range(args.kmax + 1)} for d in DEGREES}
        floors = {d: [] for d in DEGREES}
        decay_curves, horizons, peaks = [], [], []
        dist_lag1 = {}  # m -> list over seeds
        for (us, ns) in seed_pairs:
            t0 = time.time()
            ru = np.random.default_rng(us)
            u = ru.uniform(-1, 1, args.steps)
            symbols = rz.bin_input(u, args.n_bins)
            noises = rz.make_noise(args.steps, L, vocab, dev, ns)
            init = res.random_init(123)
            states = res.run(u, init, noises)
            X = res.features(states, table); Xz, = rz.standardize(X[sp.train], X)
            rp = cap.RidgePath(Xz, sp)
            for d in DEGREES:
                for k in range(args.kmax + 1):
                    cap_dk[d][k].append(rp.capacity(single_var_target(symbols, basis, d, k))["capacity"])
                floors[d].append(degree_floor(rp, symbols, basis, d, args.kmax, args.n_shuffle, ru))
            # (a) input-perturbation-decay (reuses base states; truncated runs)
            t0_list = list(np.linspace(300, args.steps - 60, 5).round().astype(int))
            mean_decay, hz, pk = input_perturbation_decay(
                res, u, states, init, noises, t0_list=t0_list)
            decay_curves.append(mean_decay); horizons.append(hz); peaks.append(pk)
            # (b) lag-1 capacity vs site-distance (first-m reservoir sites)
            for m in (1, 2, 4, 8, 16, nres):
                cols = np.arange(m * args.k)
                rpm = cap.RidgePath(Xz[:, cols], sp)
                c1 = rpm.capacity(single_var_target(symbols, basis, 1, 1))["capacity"]
                dist_lag1.setdefault(m, []).append(c1)
            print(f"  [T={temp} seed={us}] done ({time.time()-t0:.0f}s)", flush=True)

        # aggregate capacity-vs-lag
        print(f"\n=== T={temp}: capacity-vs-lag by degree (mean±std over {args.seeds} seeds) ===")
        horizon_by_deg = {}
        for d in DEGREES:
            floor = float(np.mean(floors[d]))
            means = np.array([np.mean(cap_dk[d][k]) for k in range(args.kmax + 1)])
            stds = np.array([np.std(cap_dk[d][k]) for k in range(args.kmax + 1)])
            above = [k for k in range(1, args.kmax + 1) if means[k] > floor]  # lags>=1 above floor
            horizon_by_deg[d] = max(above) if above else 0
            for k in range(args.kmax + 1):
                rows.append({"temp": temp, "degree": d, "lag": k,
                             "capacity_mean": means[k], "capacity_std": stds[k], "floor": floor})
            head = " ".join(f"{means[k]:.3f}" for k in range(min(7, args.kmax + 1)))
            print(f"  deg {d}: floor={floor:.4f}  cap(k=0..6)= {head}  "
                  f"-> last lag>=1 above floor = {horizon_by_deg[d]}")
        # the headline verdict
        lin_h, nl_h = horizon_by_deg[1], max(horizon_by_deg[2], horizon_by_deg[3])
        verdict = ("NONLINEAR MEMORY BEYOND LINEAR HORIZON" if nl_h > lin_h
                   else "all degrees decay together (poor reservoir, low capacity)")
        print(f"  >>> linear horizon={lin_h}  nonlinear horizon={nl_h}  => {verdict}")

        # (a) decay summary
        mean_decay = np.vstack(decay_curves).mean(axis=0)
        print(f"  [input-perturbation-decay] peak={np.mean(peaks):.1f}/{nres} sites, "
              f"state-space horizon={int(np.mean(horizons))} steps "
              f"(response>0.5 site); curve[0:8]= " + " ".join(f"{v:.1f}" for v in mean_decay[:8]))
        for off, v in enumerate(mean_decay):
            decay_rows.append({"temp": temp, "offset": off, "mean_reservoir_hamming": v})
        # (b) distance summary
        print(f"  [lag-1 cap vs site-distance] " + " ".join(
            f"first{m}={np.mean(v):.3f}" for m, v in sorted(dist_lag1.items())))
        for m, v in sorted(dist_lag1.items()):
            dist_rows.append({"temp": temp, "first_m_sites": m, "lag1_capacity_mean": float(np.mean(v))})

    os.makedirs(OUT, exist_ok=True)
    for name, data in (("capacity_vs_lag", rows), ("capacity_perturbation_decay", decay_rows),
                       ("capacity_lag1_vs_distance", dist_rows)):
        p = os.path.join(OUT, f"{name}_pythia160m_L{L}.csv")
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys())); w.writeheader(); w.writerows(data)
        print(f"[wrote] {p}")
    plot_all(rows, decay_rows, temps, args, os.path.join(OUT, f"capacity_vs_lag_pythia160m_L{L}.png"))


def plot_all(rows, decay_rows, temps, args, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(temps) + 1, figsize=(6 * (len(temps) + 1), 4.6), squeeze=False)
    colors = {1: "tab:blue", 2: "tab:orange", 3: "tab:green"}
    for j, temp in enumerate(temps):
        ax = axes[0][j]
        for d in DEGREES:
            rs = [r for r in rows if r["temp"] == temp and r["degree"] == d]
            rs.sort(key=lambda r: r["lag"])
            ks = [r["lag"] for r in rs]
            mu = [r["capacity_mean"] for r in rs]
            sd = [r["capacity_std"] for r in rs]
            floor = rs[0]["floor"]
            ax.errorbar(ks, mu, yerr=sd, marker="o", ms=3, lw=1.3, color=colors[d],
                        capsize=2, label=f"degree {d}")
            ax.axhline(floor, color=colors[d], ls=":", lw=0.9, alpha=0.7)
        ax.set_xlabel("lag k"); ax.set_ylabel("single-var capacity (test R²)")
        ax.set_title(f"capacity-vs-lag by degree (T={temp})")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
        ax.set_yscale("symlog", linthresh=0.01)
    # decay panel
    ax = axes[0][-1]
    for temp in temps:
        rs = [r for r in decay_rows if r["temp"] == temp]
        rs.sort(key=lambda r: r["offset"])
        ax.semilogy([r["offset"] for r in rs], [max(r["mean_reservoir_hamming"], 0.1) for r in rs],
                    marker="o", ms=2, lw=1.2, label=f"T={temp}")
    ax.set_xlabel("steps after input perturbation"); ax.set_ylabel("reservoir Hamming response")
    ax.set_title("state-space memory: input-perturbation decay")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    fig.suptitle("Locating the causal LLM-CA on the memory × nonlinearity plane "
                 "(pythia-160m, L=%d, n_in=%d)" % (args.length, args.n_in))
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(png, dpi=140); plt.close(fig)
    print(f"[wrote] {png}")


if __name__ == "__main__":
    main()
