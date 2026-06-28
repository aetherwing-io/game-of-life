"""T→0 sweep: MC vs INPUT-SENSITIVITY on one axis (lead steer #1 / task #50).

As T→0 the map deterministizes; §24 says the *undriven* T=0 map freezes to a fixed
point. Driven, it should still track the clamped input — but we must CONFIRM that,
else a high low-T MC could be measuring a frozen, input-independent state rather
than memory. At each T we report, over the SAME trajectory:

  * MC over the encoded bin b(t)           — the memory number.
  * input-sensitivity                      — peak reservoir Hamming response to a
                                             single input-symbol flip (same init,
                                             same fixed noise), normalized by
                                             n_reservoir. ~0 ⇒ frozen / input-blind.
  * free-running change-rate               — fraction of reservoir sites that change
                                             per step in the unperturbed run; a
                                             second frozen check.

The honest memory bound is max MC over the range where input-sensitivity is clearly
nonzero. Below that, MC is measuring freeze, not a memory ceiling.

Run: HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts_T_sweep_sensitivity.py --device mps
"""
import argparse, csv, os, time, warnings
warnings.filterwarnings("ignore", message=".*encountered in matmul.*")
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np

from llm_life import capacity as cap
from llm_life import reservoir as rz
from scripts_reservoir import build_causal, make_reservoir
from scripts_capacity_vs_lag import input_perturbation_decay

OUT = "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--temps", default="0.0,0.1,0.2,0.3,0.5,0.7,1.0")
    ap.add_argument("--length", type=int, default=48)
    ap.add_argument("--n-in", type=int, default=2, dest="n_in")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--n-bins", type=int, default=16, dest="n_bins")
    ap.add_argument("--steps", type=int, default=2200)
    ap.add_argument("--kmax", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    temps = [float(x) for x in args.temps.split(",")]
    seed_pairs = [(7 + 4 * i, 777 + i) for i in range(args.seeds)]
    auto, tok, model, dev, vocab, dead, emb = build_causal("EleutherAI/pythia-160m", args.device)
    L = args.length
    table = rz.pca_readout_table(emb, args.k)
    # split adapts to steps
    wash = 200; rest = args.steps - wash
    tr, va = int(0.6 * rest), int(0.15 * rest); te = rest - tr - va
    sp = cap.make_split(args.steps, wash, tr, va, te)
    res = make_reservoir(auto, L, vocab, emb, dead, args.n_in, args.n_bins, temps[0], dev,
                         codebook_mode="pc1")
    nres = len(res.reservoir_sites)
    print(f"[info] L={L} n_in={args.n_in} nres={nres} steps={args.steps} seeds={args.seeds} "
          f"temps={temps}")

    rows = []
    for temp in temps:
        res.temp = temp
        mc_s, sens_s, chg_s = [], [], []
        for (us, ns) in seed_pairs:
            u = np.random.default_rng(us).uniform(-1, 1, args.steps)
            b = rz.bin_input(u, args.n_bins).astype(float)
            noises = rz.make_noise(args.steps, L, vocab, dev, ns)
            init = res.random_init(123)
            states = res.run(u, init, noises)
            X = res.features(states, table); Xz, = rz.standardize(X[sp.train], X)
            mc = cap.memory_capacity(Xz, b, sp, kmax=args.kmax)["MC"]
            # input-sensitivity: peak reservoir Hamming from a single input flip
            t0_list = list(np.linspace(300, args.steps - 60, 4).round().astype(int))
            _, _, peak = input_perturbation_decay(res, u, states, init, noises, t0_list, horizon=25)
            sens = peak / nres
            # free-running change-rate (post-washout)
            R = states[wash:, res.reservoir_sites]
            chg = float((R[1:] != R[:-1]).mean())
            mc_s.append(mc); sens_s.append(sens); chg_s.append(chg)
        row = {"temp": temp, "MC_mean": float(np.mean(mc_s)), "MC_std": float(np.std(mc_s)),
               "input_sensitivity_mean": float(np.mean(sens_s)),
               "input_sensitivity_std": float(np.std(sens_s)),
               "free_change_rate_mean": float(np.mean(chg_s))}
        rows.append(row)
        print(f"  T={temp:<4} MC={row['MC_mean']:.3f}±{row['MC_std']:.3f}  "
              f"input_sensitivity={row['input_sensitivity_mean']:.3f} (peak frac of {nres})  "
              f"free_change_rate={row['free_change_rate_mean']:.3f}", flush=True)

    os.makedirs(OUT, exist_ok=True)
    csv_path = os.path.join(OUT, f"capacity_Tsweep_sensitivity_pythia160m_L{L}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"[wrote] {csv_path}")

    # honest memory bound: max MC where sensitivity is clearly nonzero (>0.05)
    valid = [r for r in rows if r["input_sensitivity_mean"] > 0.05]
    if valid:
        best = max(valid, key=lambda r: r["MC_mean"])
        print(f"[honest memory bound] max MC where input-sensitivity>0.05: "
              f"MC={best['MC_mean']:.3f} at T={best['temp']} "
              f"(sensitivity={best['input_sensitivity_mean']:.3f})")
    frozen = [r["temp"] for r in rows if r["input_sensitivity_mean"] <= 0.05]
    print(f"[freeze check] temps with input-sensitivity<=0.05 (possible freeze): {frozen or 'none'}")
    plot(rows, args, os.path.join(OUT, f"capacity_Tsweep_sensitivity_pythia160m_L{L}.png"))


def plot(rows, args, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ts = [r["temp"] for r in rows]
    fig, ax1 = plt.subplots(figsize=(8, 5))
    c1, c2 = "tab:purple", "tab:orange"
    ax1.errorbar(ts, [r["MC_mean"] for r in rows], yerr=[r["MC_std"] for r in rows],
                 marker="o", color=c1, capsize=3, label="MC (over bin b)")
    ax1.set_xlabel("temperature T"); ax1.set_ylabel("Memory Capacity", color=c1)
    ax1.tick_params(axis="y", labelcolor=c1); ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.errorbar(ts, [r["input_sensitivity_mean"] for r in rows],
                 yerr=[r["input_sensitivity_std"] for r in rows],
                 marker="s", color=c2, capsize=3, label="input-sensitivity (peak frac)")
    ax2.plot(ts, [r["free_change_rate_mean"] for r in rows], "^--", color="tab:gray",
             alpha=0.7, label="free-running change-rate")
    ax2.axhline(0.05, color=c2, ls=":", lw=0.8, alpha=0.6)
    ax2.set_ylabel("input-sensitivity / change-rate", color=c2)
    ax2.tick_params(axis="y", labelcolor=c2); ax2.set_ylim(0, 1.02)
    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, fontsize=8, loc="center right")
    ax1.set_title(f"MC vs input-sensitivity across T — {args.length}-site causal reservoir\n"
                  "(MC is a real memory ceiling only where input still perturbs the state)")
    fig.tight_layout(); fig.savefig(png, dpi=140); plt.close(fig)
    print(f"[wrote] {png}")


if __name__ == "__main__":
    main()
