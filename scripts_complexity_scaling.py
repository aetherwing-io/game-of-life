"""Finite-size scaling E(L) of the validated complexity diagnostic (§24 firm-up).

Two questions:
  (A) Is the (h_mu, E) class discrimination L-STABLE, or a finite-size artifact?
      (tau_int/xi failed precisely because of finite-size confounds, FINDINGS §1.)
      Scan reference CAs of known class across L and confirm E(110/54) stays high,
      E(30) stays ~0, at every L.
  (B) Is the sparse bert 'lifeform' a COMMENSURATE standing wave (L-sensitive E)
      or a free structure (L-independent E)? §14 found its clean oscillator only at
      L=64,80 -> a free life-form would keep the same E at every L; a standing wave
      pinned to commensurate L would not. This is the decisive test, on the
      validated instrument, of §14/§21's 'standing waves, not life-forms' verdict.

Run: HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts_complexity_scaling.py
"""
import os, csv, time, argparse
import numpy as np
from llm_life import reference_ca as rc, complexity as cx

OUT = "results"
NMAX = 8
L_REF = [32, 48, 64, 96, 128, 160, 200, 256]
L_LLM = [40, 48, 56, 64, 72, 80, 96, 112, 128]


def E_of(field, dead, seed):
    s = cx.complexity_summary(field, nmax=NMAX, seed=seed, dead_token=dead)
    return s["E_excess"], s["h_mu"], s["live_frac"]


def ref_scaling():
    print("[A] reference-CA E(L) — class discrimination vs lattice size")
    rows = []
    for rule, cls in [(110, "4"), (54, "4"), (30, "3"), (184, "2")]:
        line = []
        for L in L_REF:
            steps = max(400, 3 * L); burn = steps // 2
            Es = []
            for s in range(2):
                st = rc.elementary(rule, L, steps, seed=s, init="random")[burn:]
                Es.append(E_of(st, 0, s)[0])
            E = float(np.mean(Es))
            line.append((L, E))
            rows.append(dict(kind="ref", name=f"rule {rule}", cls=cls, L=L, E_excess=E))
        print(f"  rule {rule:>3} (C{cls}): " + "  ".join(f"L{L}:{E:.2f}" for L, E in line))
    return rows


def _build(arch, model_name, window, device):
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
    return auto, tok, vocab, dev


def llm_scaling(device):
    import torch
    from llm_life.run import _to_np
    rows = []
    STEPS, BURN = 250, 125

    # (B1) the sparse bert 'lifeform' oscillator across L (the commensurability test)
    print("\n[B1] bert local oscillator (w=2, T=0, absorbing, penalty=2.0, seed '( a )') E(L):")
    auto, tok, vocab, dev = _build("local", "bert-base-uncased", 2, device)
    auto.freq_penalty = 2.0
    dead = auto.dead_token
    ids = tok.encode("( a )", add_special_tokens=False)
    line = []
    for L in L_LLM:
        init = torch.full((L,), dead, dtype=torch.long)
        s0 = (L - len(ids)) // 2; init[s0:s0 + len(ids)] = torch.tensor(ids); init = init.to(dev)
        g = torch.Generator(device="cpu" if dev == "mps" else dev); g.manual_seed(1)
        st = _to_np(auto.trajectory(init, STEPS, 0.0, True, generator=g)[0])[BURN:]
        E, h, live = E_of(st, dead, 0)
        line.append((L, E)); rows.append(dict(kind="llm", name="bert osc pen2", cls="?", L=L,
                                              E_excess=E, h_mu=h, live=live))
        print(f"    L={L:>3}: E={E:.2f}  h_mu={h:.2f}  live={live:.2f}")

    # (B2) causal pythia T0.5 across L (active control)
    print("\n[B2] causal pythia T0.5 (random init) E(L):")
    auto, tok, vocab, dev = _build("causal", "EleutherAI/pythia-160m", 2, device)
    dead = auto.dead_token
    for L in L_LLM:
        g = torch.Generator(device="cpu" if dev == "mps" else dev); g.manual_seed(1)
        init = torch.randint(0, vocab, (L,), generator=g).to(dev)
        st = _to_np(auto.trajectory(init, STEPS, 0.5, False, generator=g)[0])[BURN:]
        E, h, live = E_of(st, dead, 0)
        rows.append(dict(kind="llm", name="causal pythia T0.5", cls="?", L=L,
                         E_excess=E, h_mu=h, live=live))
        print(f"    L={L:>3}: E={E:.2f}  h_mu={h:.2f}  live={live:.2f}")
    return rows


def plot(rows, png):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, (axr, axl) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    # reference panel
    names = {}
    for r in rows:
        names.setdefault((r["kind"], r["name"], r.get("cls", "?")), []).append((r["L"], r["E_excess"]))
    cls_color = {"4": "tab:red", "3": "tab:blue", "2": "tab:green"}
    for (kind, name, cls), pts in names.items():
        pts = sorted(pts); Ls = [p[0] for p in pts]; Es = [p[1] for p in pts]
        if kind == "ref":
            axr.plot(Ls, Es, "o-", color=cls_color.get(cls, "k"), label=f"{name} (C{cls})")
        else:
            axl.plot(Ls, Es, "s-", label=name)
    axr.set(title="(A) reference CAs — discrimination is L-stable",
            xlabel="lattice size L", ylabel="excess entropy E (bits)")
    axr.legend(fontsize=8); axr.grid(alpha=0.3)
    # rule-110 band on the LLM panel for reference
    ref110 = sorted(names.get(("ref", "rule 110", "4"), []))
    if ref110:
        axl.axhspan(min(e for _, e in ref110), max(e for _, e in ref110),
                    color="tab:red", alpha=0.08, label="rule-110 (Class 4) band")
    axl.set(title="(B) LLM cells — commensurate (L-sensitive) vs free (flat)",
            xlabel="lattice size L")
    axl.legend(fontsize=8); axl.grid(alpha=0.3)
    fig.suptitle("Finite-size scaling E(L) of excess entropy (§24 firm-up)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(png, dpi=140); print(f"\n[wrote] {png}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="mps")
    args = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    rows = ref_scaling()
    try:
        rows += llm_scaling(args.device)
    except Exception as e:
        print(f"[warn] LLM scaling failed: {type(e).__name__}: {e}")
    with open(os.path.join(OUT, "complexity_scaling.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["kind", "name", "cls", "L", "E_excess", "h_mu", "live"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in w.fieldnames})
    print(f"[wrote] {OUT}/complexity_scaling.csv")
    plot(rows, os.path.join(OUT, "complexity_scaling.png"))


if __name__ == "__main__":
    main()
