"""Place iterated-LLM dynamics on a VALIDATED complexity-entropy plane.

The repo's structural metrics (tau_int, xi) provably fail to separate Wolfram
class (FINDINGS sec 1/sec 20). This script uses the validated measures in
llm_life/complexity.py instead:

  x = entropy rate  h_mu   (randomness / disorder)
  y = excess entropy E_exc (bias-corrected spatial predictive information)

Reference CAs of KNOWN class are the anchors (Class 4 = high E & moderate h_mu;
Class 3 = high h_mu, ~0 E; Class 1/2 = low h_mu). Then every LLM variant is run
under the SAME matched protocol (same L, random init, post-burn steady state)
and dropped onto the same plane. The question the whole project has been asking
visually -- "does any iterated-LLM rule reach the Class-4 corner?" -- becomes a
coordinate.

Pure analysis on top of the existing automaton API; no new model behavior.
Run:  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts_complexity_plane.py
"""
import os, csv, time, argparse
import numpy as np

L, STEPS, BURN, NMAX = 120, 250, 120, 8   # L=120: non-2^k so rule-90 stays chaotic
NSEED = 2
OUT = "results"


# --------------------------------------------------------------------------- #
def _agg(accum):
    keys = [k for k in accum[0] if isinstance(accum[0][k], (int, float))]
    out = {k: float(np.mean([a[k] for a in accum])) for k in keys}
    out.update({k + "_sd": float(np.std([a[k] for a in accum])) for k in ("h_mu", "E_excess")})
    return out


def ref_anchors():
    from llm_life import reference_ca as rc, complexity as cx
    rows = []
    for rule, cls in [(110, "4"), (54, "4"), (30, "3"), (90, "3"),
                      (184, "2"), (250, "2"), (0, "1")]:
        accum = []
        for s in range(NSEED):
            st = rc.elementary(rule, L, STEPS, seed=s, init="random")[BURN:]
            accum.append(cx.complexity_summary(st, nmax=NMAX, seed=s, dead_token=0))
        agg = _agg(accum)
        agg.update(kind="ref", name=f"rule {rule}", cls=cls)
        rows.append(agg)
        print(f"  rule {rule:>3} (C{cls}): h_mu={agg['h_mu']:.2f}  E_exc={agg['E_excess']:.2f}"
              f"±{agg['E_excess_sd']:.2f}  MI1bin={agg['MI1_bin']:.3f}")
    return rows


# --------------------------------------------------------------------------- #
def _build(arch, model_name, window=2, device="mps"):
    import torch
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
        from llm_life.model import load, dead_token_id
        from llm_life.automaton import LLMAutomaton
        model, tok, dev = load(model_name, dev)
        bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
        dead = dead_token_id(model, tok, bos, dev)
        vocab = getattr(model.config, "vocab_size", None) or model.config.text_config.vocab_size
        auto = LLMAutomaton(model, dead_token=dead, bos_token=bos, device=dev)
    return auto, vocab, dev


def _seeded_init(seed_text, tok, dead, vocab, dev, rng_seed):
    """A dead background with a seed motif in the centre (sparse, dead-background
    regimes) or a full random lattice if seed_text is None."""
    import torch
    if seed_text is None:
        g = torch.Generator(device="cpu"); g.manual_seed(rng_seed)
        return torch.randint(0, vocab, (L,), generator=g).to(dev)
    ids = tok.encode(seed_text, add_special_tokens=False)[:L]
    init = torch.full((L,), dead, dtype=torch.long)
    start = (L - len(ids)) // 2
    init[start:start + len(ids)] = torch.tensor(ids, dtype=torch.long)
    return init.to(dev)


def run_llm(label, arch, model_name, temp, window=2, device="mps",
            absorbing=False, penalty=0.0, refractory=0, seed_text=None):
    """Run NSEED trajectories, return (aggregated row, seed-0 field). seed_text
    set => sparse dead-background regime (absorbing/penalty/refractory); else
    random-init soft."""
    import torch
    from llm_life import complexity as cx
    from llm_life.run import _to_np
    t0 = time.time()
    auto, vocab, dev = _build(arch, model_name, window, device)
    auto.freq_penalty = penalty
    dead = auto.dead_token
    # need a tokenizer for seeded init
    tok = None
    if seed_text is not None:
        from llm_life.model import load_mlm, load
        if arch in ("masked", "local"):
            _, tok, _ = load_mlm(model_name, dev)
        else:
            _, tok, _ = load(model_name, dev)
    nseed = 1 if seed_text is not None else NSEED   # seeded runs are deterministic
    accum, field0 = [], None
    for seed in range(nseed):
        gen_dev = "cpu" if dev == "mps" else dev
        g = torch.Generator(device=gen_dev); g.manual_seed(seed + 1)
        init = _seeded_init(seed_text, tok, dead, vocab, dev, seed + 1)
        states, _ = auto.trajectory(init, STEPS, temp, absorbing, generator=g,
                                    refractory=refractory)
        st = _to_np(states)[BURN:]
        accum.append(cx.complexity_summary(st, nmax=NMAX, seed=seed, dead_token=dead))
        if seed == 0:
            field0 = _to_np(states)
    s = _agg(accum)
    s.update(kind="llm", name=label, cls="?", dead_token=int(dead))
    print(f"  {label:<26} h_mu={s['h_mu']:.2f}±{s['h_mu_sd']:.2f}  "
          f"E_exc={s['E_excess']:.2f}±{s['E_excess_sd']:.2f}  "
          f"MI1bin={s['MI1_bin']:.3f} MItok={s['MI1_tok']:.2f} live={s['live_frac']:.2f}  ({time.time()-t0:.0f}s)")
    return s, field0


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    print("[1] reference-CA anchors (known Wolfram class), matched L=%d:" % L)
    rows = ref_anchors()

    # The LLM menu: causal (synchronizing), masked (the 'closest to edge'
    # drifting-cluster regime, sec 8), local windowed ring (the sec 23 damage
    # boundary: w=1 coalesces, w=2 boundary, w>=3 preserves damage).
    # (a) ACTIVE random-init soft regimes: causal active + masked/local fills.
    # (b) STRUCTURE-BEARING sparse regimes (dead background, where live/dead is
    #     meaningful and a glider would be visible): the sec 14/21 bert local-
    #     penalty oscillators + a sec 12 propagating refractory cone.
    menu = [
        dict(label="causal pythia T0.5",  arch="causal", model="EleutherAI/pythia-160m", temp=0.5, window=2),
        dict(label="causal pythia T1.0",  arch="causal", model="EleutherAI/pythia-160m", temp=1.0, window=2),
        dict(label="masked distilR T1.2", arch="masked", model="distilroberta-base",     temp=1.2, window=2),
        # sparse, dead-background structure regimes (the localized 'lifeforms'):
        dict(label="local bert osc pen2", arch="local", model="bert-base-uncased", temp=0.0, window=2,
             absorbing=True, penalty=2.0, refractory=0, seed_text="( a )"),
        dict(label="local bert osc pen2 r3", arch="local", model="bert-base-uncased", temp=0.0, window=2,
             absorbing=True, penalty=2.0, refractory=3, seed_text="one two three"),
        dict(label="local distilR cone r3", arch="local", model="distilroberta-base", temp=0.0, window=2,
             absorbing=True, penalty=0.0, refractory=3, seed_text=" Comments Related Posts"),
    ]
    print(f"\n[2] LLM variants: (a) active random-init, (b) sparse dead-background structures:")
    fields = {}
    for m in menu:
        try:
            s, st = run_llm(m["label"], m["arch"], m["model"], m["temp"], m.get("window", 2),
                            device=args.device, absorbing=m.get("absorbing", False),
                            penalty=m.get("penalty", 0.0), refractory=m.get("refractory", 0),
                            seed_text=m.get("seed_text"))
            rows.append(s); fields[m["label"]] = st
        except Exception as e:
            print(f"  [skip] {m['label']}: {type(e).__name__}: {e}")

    # write CSV
    csv_path = os.path.join(OUT, "complexity_plane.csv")
    keys = ["kind", "name", "cls", "h_mu", "h_mu_sd", "E_raw", "E_shuffle", "E_excess",
            "E_excess_sd", "MI1_bin", "MI1_tok", "TE_right_bin", "live_frac",
            "site_entropy_tok"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print(f"\n[wrote] {csv_path}")

    plot_plane(rows, os.path.join(OUT, "complexity_plane.png"))
    # save the most edge-like LLM field for the glider-filter figure
    np.savez(os.path.join(OUT, "complexity_llm_fields.npz"),
             **{k.replace(" ", "_"): v for k, v in fields.items()})
    print(f"[wrote] {OUT}/complexity_llm_fields.npz  ({len(fields)} fields)")


def plot_plane(rows, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 7))
    cls_color = {"4": "tab:red", "3": "tab:blue", "2": "tab:green", "1": "0.5"}
    for r in rows:
        xe, ye = r.get("h_mu_sd", 0), r.get("E_excess_sd", 0)
        if r["kind"] == "ref":
            ax.errorbar(r["h_mu"], r["E_excess"], xerr=xe, yerr=ye, fmt="none",
                        ecolor="0.6", elinewidth=0.8, zorder=4)
            ax.scatter(r["h_mu"], r["E_excess"], s=200, marker="*",
                       color=cls_color.get(r["cls"], "k"), edgecolor="k", zorder=5)
            ax.annotate(f"{r['name']} (C{r['cls']})", (r["h_mu"], r["E_excess"]),
                        textcoords="offset points", xytext=(8, 4), fontsize=9, fontweight="bold")
        else:
            ax.errorbar(r["h_mu"], r["E_excess"], xerr=xe, yerr=ye, fmt="none",
                        ecolor="tab:purple", alpha=0.4, elinewidth=0.8, zorder=3)
            ax.scatter(r["h_mu"], r["E_excess"], s=70, marker="o",
                       facecolor="none", edgecolor="tab:purple", linewidth=1.6, zorder=4)
            ax.annotate(r["name"], (r["h_mu"], r["E_excess"]),
                        textcoords="offset points", xytext=(6, -3), fontsize=7.5, color="tab:purple")
    ax.set_xlabel("entropy rate  h_mu  (bits/site)  —  disorder →")
    ax.set_ylabel("excess entropy  E_excess  (bits)  —  structure / predictive info ↑")
    ax.set_title("Complexity–entropy plane: iterated LLMs vs known-class CA anchors\n"
                 "(Class 4 / edge of chaos = high E AND moderate h_mu — upper middle)")
    ax.grid(alpha=0.3)
    ax.axhline(0, color="k", lw=0.5)
    # annotate the corners
    ax.text(0.98, 0.97, "Class 4\n(edge of chaos)", transform=ax.transAxes,
            ha="right", va="top", color="tab:red", fontsize=9, alpha=0.6)
    ax.text(0.98, 0.03, "Class 3 (chaos)", transform=ax.transAxes,
            ha="right", va="bottom", color="tab:blue", fontsize=9, alpha=0.6)
    ax.text(0.02, 0.03, "Class 1/2\n(frozen/periodic)", transform=ax.transAxes,
            ha="left", va="bottom", color="0.4", fontsize=9, alpha=0.6)
    fig.tight_layout()
    fig.savefig(png, dpi=140)
    print(f"[wrote] {png}")


if __name__ == "__main__":
    main()
