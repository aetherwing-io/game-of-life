"""GATE #47 — settle the nonlinear-temporal claim with a DEGREE-STRATIFIED floor.

The degree-agnostic floor under-catches high-degree finite-sample bias (heavier
Legendre tails), and that bias is seed-reproducible, so cross-seed stability does
not save a high-degree config. Here we recompute the shuffled-input null PER
DEGREE and ask, for the IN-DOMAIN (ESP-verified) temperatures T=0.3 and T=0.7,
whether each degree's TEMPORAL capacity survives its own degree-matched floor.

Decision (per the gate):
  * d>=2 temporal drops below degree-matched floor -> "weak LINEAR lag memory only".
  * d3 (odd) temporal survives                     -> "weak linear + genuine odd nonlinear tail".

T=0 is excluded (ESP was only verified for T in [0.3,1.1]; T=0 is the noiseless
argmax limit, a different regime).
"""
import os, warnings
warnings.filterwarnings("ignore", message=".*encountered in matmul.*")
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np
from collections import defaultdict

from llm_life import capacity as cap
from llm_life import reservoir as rz
from scripts_reservoir import build_causal, make_reservoir

TEMPS = [0.3, 0.7]
SEEDS = [(7, 777), (11, 778), (15, 779)]
STEPS, NBINS, K, NIN = 3000, 16, 8, 2
MAXDEG, MAXDELAY, MAXVARS, NSURR = 4, 8, 2, 80


def raw_per_cell(rows):
    """Unthresholded per-(degree,kind) capacity sums."""
    out = defaultdict(float)
    for r in rows:
        out[(r["degree"], "inst" if r["inst"] else "temporal")] += r["capacity"]
    return out


def main():
    auto, tok, model, dev, vocab, dead, emb = build_causal("EleutherAI/pythia-160m", "mps")
    L = 48
    table = rz.pca_readout_table(emb, K)
    basis = cap.symbol_poly_basis(NBINS, MAXDEG)
    sp = cap.make_split(STEPS, 200, 1800, 400, 600)
    res = make_reservoir(auto, L, vocab, emb, dead, NIN, NBINS, TEMPS[0], dev, codebook_mode="pc1")

    print(f"[info] in-domain temps {TEMPS}, seeds {[s for s,_ in SEEDS]}, "
          f"degree-stratified floor (max over shuffled-input null per degree), n_surrogate={NSURR}")
    csv_rows = []
    degree_robust = defaultdict(list)    # degree -> [robust? per temp] (cross-temp criterion)
    for temp in TEMPS:
        res.temp = temp
        raw_t = defaultdict(list)        # degree -> [raw temporal cap per seed]
        thr_t = defaultdict(list)        # degree -> [degree-matched max floor per seed]
        strat_t = defaultdict(list)      # degree -> [stratified-thresholded temporal per seed]
        agnostic_t = defaultdict(list)   # degree -> [agnostic-thresholded temporal per seed]
        mc1 = []
        for (us, ns) in SEEDS:
            u = np.random.default_rng(us).uniform(-1, 1, STEPS)
            symbols = rz.bin_input(u, NBINS)
            noises = rz.make_noise(STEPS, L, vocab, dev, ns)
            states = res.run(u, res.random_init(123), noises)
            X = res.features(states, table); Xz, = rz.standardize(X[sp.train], X)
            mc = cap.memory_capacity(Xz, u, sp, kmax=20); mc1.append(mc["mc_k"][1])
            ipc = cap.information_processing_capacity(Xz, symbols, sp, basis, max_degree=MAXDEG,
                                                      max_delay=MAXDELAY, max_vars=MAXVARS,
                                                      n_surrogate=NSURR, rng_seed=us)
            raw = raw_per_cell(ipc["rows"])
            strat = ipc["summary_max"]["per_cell"]                 # degree-matched floor
            agn_floor = ipc["thresholds"]["max"]                   # single global floor
            for d in range(1, MAXDEG + 1):
                raw_t[d].append(raw.get((d, "temporal"), 0.0))
                thr_t[d].append(ipc["thresholds_by_degree"][d]["max"])
                strat_t[d].append(strat.get((d, "temporal"), 0.0))
                # agnostic: re-threshold raw temporal configs against the single floor
                agn = sum(r["capacity"] for r in ipc["rows"]
                          if r["degree"] == d and not r["inst"] and r["capacity"] > agn_floor)
                agnostic_t[d].append(agn)
        print(f"\n=== T={temp}  (MC_1={np.mean(mc1):.3f}±{np.std(mc1):.3f}) ===")
        print(f"  {'deg':>3} | {'raw_temporal':>14} | {'deg-floor(max)':>14} | "
              f"{'agnostic_thr':>13} | {'STRATIFIED_thr':>14}")
        for d in range(1, MAXDEG + 1):
            rt, st = np.array(raw_t[d]), np.array(strat_t[d])
            fl, ag = np.array(thr_t[d]), np.array(agnostic_t[d])
            # TIGHTENED survival criterion (reviewer #47): a degree is a GENUINE
            # signal only if its stratified temporal clears the degree-matched MAX
            # floor by MORE than its own seed-std — i.e. (mean − std) > floor — and
            # does so at EVERY temp. The old `st.mean()>1e-6` produced a false
            # "survival" for d3 (a floor-HEIGHT artifact: d3≈d4 magnitude, but the
            # d3 floor sits below it and the d4 floor above). margin = mean − floor.
            margin = float(st.mean() - fl.mean())
            robust = (st.mean() - st.std()) > fl.mean()      # clears floor by >seed-std
            degree_robust[d].append(bool(robust))
            verdict = "CLEARS>std" if robust else ("marginal" if st.mean() > fl.mean() else "below_floor")
            tag = "(linear)" if d == 1 else f"(deg-{d} {'odd' if d % 2 else 'even'})"
            print(f"  {d:>3} | {rt.mean():>7.3f}±{rt.std():.3f} | {fl.mean():>14.4f} | "
                  f"margin={margin:>+.4f} | {st.mean():>7.3f}±{st.std():.3f}  {verdict} {tag}")
            csv_rows.append({"temp": temp, "degree": d,
                             "parity": "linear" if d == 1 else ("odd" if d % 2 else "even"),
                             "MC_1": round(float(np.mean(mc1)), 4),
                             "raw_temporal_mean": round(float(rt.mean()), 4),
                             "raw_temporal_std": round(float(rt.std()), 4),
                             "deg_matched_floor_max": round(float(fl.mean()), 4),
                             "stratified_temporal_mean": round(float(st.mean()), 4),
                             "stratified_temporal_std": round(float(st.std()), 4),
                             "margin_above_floor": round(margin, 4),
                             "clears_by_seedstd": int(robust),
                             "verdict": verdict})

    import csv as _csv, os as _os
    _os.makedirs("results", exist_ok=True)
    _p = _os.path.join("results", "capacity_degree_floor_pythia160m_L48.csv")
    with open(_p, "w", newline="") as _f:
        _w = _csv.DictWriter(_f, fieldnames=list(csv_rows[0].keys())); _w.writeheader(); _w.writerows(csv_rows)
    print(f"[wrote] {_p}")
    # headline decision (TIGHTENED, reviewer #47): a nonlinear degree (≥2) is GENUINE
    # only if it clears the degree-matched MAX floor by > seed-std at EVERY temp.
    genuine = [d for d in range(2, MAXDEG + 1) if degree_robust[d] and all(degree_robust[d])]
    print(f"\n[cross-temp robustness] degree d: clears-floor-by->seedstd at ALL temps?")
    for d in range(2, MAXDEG + 1):
        print(f"  deg {d} ({'odd' if d%2 else 'even'}): per-temp robust = {degree_robust[d]} "
              f"-> {'GENUINE' if (degree_robust[d] and all(degree_robust[d])) else 'not established'}")
    if genuine:
        print(f"\n[VERDICT] nonlinear temporal ESTABLISHED at degrees {genuine} (clear floor by "
              f">seed-std at every temp) → 'weak linear + genuine nonlinear tail'.")
    else:
        print("\n[VERDICT] NO degree≥2 clears the degree-matched floor by >seed-std at every temp "
              "→ nonlinear temporal NOT established (floor-height / cross-temp artifact) → "
              "§25 = 'weak LINEAR lag memory only'.")


if __name__ == "__main__":
    main()
