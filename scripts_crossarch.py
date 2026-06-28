"""Cross-architecture consistency-vs-capacity contrast (task #43).

§23 established a CONSISTENCY axis across map architectures: causal full-attention
and local window w=1 COALESCE (echo-state property holds); local w>=3 and global
masked do NOT (damage saturates). This script adds the CAPACITY axis: drive each
map as a reservoir and measure MC + temporal IPC with the licensed estimators.

The hypothesis it tests: is there a consistency-vs-capacity tradeoff? A reservoir
is only usable where ESP holds (state is a function of input history, not the
initial condition). The §23-consistent maps have ESP but (we found for causal)
~0 temporal capacity; the non-consistent maps may look "richer" but ESP fails, so
their capacity is initial-condition-dependent and not a usable reservoir number.
We measure both and flag ESP per arch, turning the single-map result into a curve.

Run: HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts_crossarch.py --device mps
"""
import argparse, csv, os, time, warnings
warnings.filterwarnings("ignore", message=".*encountered in matmul.*")
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np

from llm_life import capacity as cap
from llm_life import reservoir as rz
from scripts_reservoir import make_reservoir, rz_model_tag

OUT = "results"
DEGREES = (1, 2, 3, 4)

# (label, arch, model, window, temp)
CONFIGS = [
    ("causal pythia",     "causal", "EleutherAI/pythia-160m", None, 0.7),
    ("local w1 distilR",  "local",  "distilroberta-base",     1,    0.7),
    ("local w2 distilR",  "local",  "distilroberta-base",     2,    0.7),
    ("local w4 distilR",  "local",  "distilroberta-base",     4,    0.7),
    ("masked distilR",    "masked", "distilroberta-base",     None, 0.7),
]


def build_any(arch, model_name, window, device):
    from llm_life.model import pick_device
    dev = pick_device(device)
    if arch in ("masked", "local"):
        from llm_life.model import load_mlm, dead_token_id_mlm
        model, tok, dev = load_mlm(model_name, dev)
        dead = dead_token_id_mlm(model, tok, dev)
        vocab = model.config.vocab_size
        common = dict(model=model, dead_token=dead, mask_token=tok.mask_token_id,
                      cls_token=tok.cls_token_id, sep_token=tok.sep_token_id, device=dev)
        if arch == "local":
            from llm_life.automaton import LocalMaskedLMAutomaton
            auto = LocalMaskedLMAutomaton(window=window, **common)
        else:
            from llm_life.automaton import MaskedLMAutomaton
            auto = MaskedLMAutomaton(**common)
    else:
        from llm_life.automaton import LLMAutomaton
        from llm_life.model import dead_token_id, load
        model, tok, dev = load(model_name, dev)
        bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
        dead = dead_token_id(model, tok, bos, dev)
        vocab = getattr(model.config, "vocab_size", None) or model.config.text_config.vocab_size
        auto = LLMAutomaton(model, dead_token=dead, bos_token=bos, device=dev)
    emb = model.get_input_embeddings().weight.detach().to("cpu").float().numpy()
    return auto, tok, model, dev, vocab, dead, emb


def measure_esp(res, T, pairs, L, vocab, dev):
    """Replicas from DIFFERENT inits, same input + same fixed noise. Return mean
    final reservoir Hamming (fraction) and p_sync over pairs."""
    nres = len(res.reservoir_sites)
    finals, syncs = [], []
    for p in range(pairs):
        u = np.random.default_rng(100 + p).uniform(-1, 1, T)
        noises = rz.make_noise(T, L, vocab, dev, seed=4000 + p)
        sA = res.run(u, res.random_init(10 * p + 1), noises)
        sB = res.run(u, res.random_init(10 * p + 2), noises)
        ham = (sA[:, res.reservoir_sites] != sB[:, res.reservoir_sites]).sum(axis=1)
        finals.append(ham[-10:].mean() / nres)
        syncs.append(1.0 if ham[-10:].max() == 0 else 0.0)
    return float(np.mean(finals)), float(np.mean(syncs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--length", type=int, default=48)
    ap.add_argument("--n-in", type=int, default=2, dest="n_in")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--n-bins", type=int, default=16, dest="n_bins")
    ap.add_argument("--esp-steps", type=int, default=120, dest="esp_steps")
    ap.add_argument("--esp-pairs", type=int, default=4, dest="esp_pairs")
    ap.add_argument("--steps", type=int, default=1600)
    ap.add_argument("--washout", type=int, default=200)
    ap.add_argument("--train", type=int, default=900)
    ap.add_argument("--val", type=int, default=250)
    ap.add_argument("--test", type=int, default=250)
    ap.add_argument("--kmax", type=int, default=15)
    ap.add_argument("--max-degree", type=int, default=3, dest="max_degree")
    ap.add_argument("--max-delay", type=int, default=6, dest="max_delay")
    ap.add_argument("--max-vars", type=int, default=2, dest="max_vars")
    ap.add_argument("--n-surrogate", type=int, default=32, dest="n_surrogate")
    args = ap.parse_args()

    L = args.length
    sp = cap.make_split(args.steps, args.washout, args.train, args.val, args.test)
    basis = cap.symbol_poly_basis(args.n_bins, args.max_degree)
    rows = []
    for (label, arch, model_name, window, temp) in CONFIGS:
        try:
            t0 = time.time()
            auto, tok, model, dev, vocab, dead, emb = build_any(arch, model_name, window, args.device)
            table = rz.pca_readout_table(emb, args.k)
            res = make_reservoir(auto, L, vocab, emb, dead, args.n_in, args.n_bins,
                                 temp, dev, codebook_mode="pc1")
            # ESP
            esp_h, p_sync = measure_esp(res, args.esp_steps, args.esp_pairs, L, vocab, dev)
            esp_ok = esp_h < 0.02
            # capacity (single seed; this is a breadth scan, not the locked number)
            u = np.random.default_rng(7).uniform(-1, 1, args.steps)
            symbols = rz.bin_input(u, args.n_bins)
            noises = rz.make_noise(args.steps, L, vocab, dev, seed=777)
            states = res.run(u, res.random_init(123), noises)
            X = res.features(states, table)
            Xz, = rz.standardize(X[sp.train], X)
            mc = cap.memory_capacity(Xz, u, sp, kmax=args.kmax)
            ipc = cap.information_processing_capacity(
                Xz, symbols, sp, basis, max_degree=args.max_degree, max_delay=args.max_delay,
                max_vars=args.max_vars, n_surrogate=args.n_surrogate, rng_seed=7)
            s = ipc["summary_max"]
            row = {"label": label, "arch": arch, "model": model_name,
                   "window": window if window is not None else "", "temp": temp,
                   "esp_final_hamming_frac": esp_h, "p_sync": p_sync, "esp_ok": int(esp_ok),
                   "MC": mc["MC"], "MC_0": mc["mc_k"][0], "MC_1": mc["mc_k"][1],
                   "IPC_total": s["total"], "IPC_inst": s["inst_total"],
                   "IPC_temporal": s["temporal_total"], "eff_rank": ipc["eff_rank"],
                   "readout_dim": len(res.reservoir_sites) * args.k,
                   "MC_alpha_at_max_signif": mc["alpha_at_max_signif"],
                   "surrogate_max": ipc["thresholds"]["max"]}
            rows.append(row)
            print(f"  {label:<18} ESP_h={esp_h:.3f} p_sync={p_sync:.2f} {'ESP' if esp_ok else 'no-ESP':<7} "
                  f"| MC={mc['MC']:.3f} (0={mc['mc_k'][0]:.2f},1={mc['mc_k'][1]:.2f}) "
                  f"IPC={s['total']:.2f} inst={s['inst_total']:.2f} temporal={s['temporal_total']:.2f} "
                  f"a@max={mc['alpha_at_max_signif']} ({time.time()-t0:.0f}s)", flush=True)
            del auto, model
            import gc, torch
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception as e:
            print(f"  [skip] {label}: {type(e).__name__}: {e}", flush=True)

    os.makedirs(OUT, exist_ok=True)
    csv_path = os.path.join(OUT, "crossarch_capacity.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"[wrote] {csv_path}")
    plot_crossarch(rows, os.path.join(OUT, "crossarch_capacity.png"), args)


def plot_crossarch(rows, png, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5))
    labels = [r["label"] for r in rows]
    x = np.arange(len(rows))
    # panel 0: consistency (p_sync) and capacity (MC, temporal IPC) side by side
    ax0.bar(x - 0.27, [r["p_sync"] for r in rows], 0.27, color="tab:gray", label="p_sync (consistency)")
    ax0.bar(x, [r["MC"] for r in rows], 0.27, color="tab:purple", label="MC")
    ax0.bar(x + 0.27, [r["IPC_temporal"] for r in rows], 0.27, color="tab:green", label="temporal IPC")
    ax0.set_xticks(x); ax0.set_xticklabels(labels, rotation=25, ha="right", fontsize=7)
    ax0.set_title("consistency vs capacity by architecture"); ax0.legend(fontsize=8)
    ax0.grid(alpha=0.3, axis="y")
    # panel 1: ESP final Hamming vs temporal IPC scatter, colored by ESP status
    for r in rows:
        c = "tab:blue" if r["esp_ok"] else "tab:red"
        ax1.scatter(r["esp_final_hamming_frac"], r["IPC_temporal"], s=80, color=c, zorder=3)
        ax1.annotate(r["label"], (r["esp_final_hamming_frac"], r["IPC_temporal"]),
                     textcoords="offset points", xytext=(6, 3), fontsize=7)
    ax1.axvspan(0, 0.02, color="tab:blue", alpha=0.08)
    ax1.set_xlabel("ESP final reservoir Hamming (0 = consistent)")
    ax1.set_ylabel("temporal IPC")
    ax1.set_title("usable reservoirs live at left edge (ESP holds)")
    ax1.grid(alpha=0.3)
    fig.suptitle(f"Cross-architecture consistency-vs-capacity — L={args.length}, n_in={args.n_in}")
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(png, dpi=140); plt.close(fig)
    print(f"[wrote] {png}")


if __name__ == "__main__":
    main()
