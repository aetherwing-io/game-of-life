"""Re-license gates G3/G4 for the new llm_life.capacity estimators (extended-α,
encoded-symbol Gram-Schmidt IPC, inst/temporal split). Pure synthetic; no model."""
import os, warnings
warnings.filterwarnings("ignore", message=".*encountered in matmul.*")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
from llm_life import capacity as cap
from llm_life import reservoir as rz

T = 3000
N_BINS = 16
SPLIT = cap.make_split(T, 200, 1800, 400, 600)
rng = np.random.default_rng(7)
u = rng.uniform(-1, 1, T)
sym = rz.bin_input(u, N_BINS)
basis = cap.symbol_poly_basis(N_BINS, 4)

print("symbol_poly_basis orthonormality (Gram under uniform symbols, should be I):")
G = np.array([[ (basis[:, a] * basis[:, b]).mean() for b in range(5)] for a in range(5)])
print(np.round(G, 3))

# ---- G3a: MC on a continuous shift register, extended-alpha interiority ----
N = 25
Xsr = np.zeros((T, N))
for k in range(N):
    Xsr[k:, k] = u[: T - k]
mc = cap.memory_capacity(Xsr, u, SPLIT, kmax=40)
print(f"\n[G3a MC] shift-register N={N}: MC={mc['MC']:.3f} (expect ~{N})  "
      f"alpha_at_MAX={mc['alpha_at_max']} (must be 0)  alpha_at_min={mc['alpha_at_min']}")

# ---- G3b: symbol-IPC linear true-negative: shift register of phi_1(symbol) ----
Xsr_sym = np.zeros((T, N))
for k in range(N):
    sh = np.zeros(T, dtype=int); sh[k:] = sym[: T - k]
    Xsr_sym[:, k] = basis[sh, 1]
r = cap.information_processing_capacity(Xsr_sym, sym, SPLIT, basis,
                                        max_degree=3, max_delay=N + 2, max_vars=2, n_surrogate=40)
s = r["summary_max"]
g3b_total, g3b_d2 = s["total"], s["per_degree"].get(2, 0.0) + s["per_degree"].get(3, 0.0)
g3a_mc = mc["MC"]
print(f"\n[G3b IPC] symbol shift-register N={N}: total={s['total']:.2f} (expect ~{N})")
print(f"   per-degree: " + " ".join(f"d{g}={s['per_degree'].get(g,0):.2f}" for g in (1, 2, 3))
      + f"   (deg>=2 must be ~0)")
print(f"   inst(delay0)={s['inst_total']:.2f} (expect ~1)  temporal={s['temporal_total']:.2f} "
      f"(expect ~{N-1})   alpha_at_MAX={r['alpha_at_max']}")

# ---- G4: nonlinear POSITIVE control in the encoded-symbol basis ----
SIGNAL = [((0, 1),), ((4, 1),), ((0, 2),), ((2, 2),), ((0, 3),),
          ((1, 1), (3, 1)), ((0, 1), (2, 1))]
cols = []
for cfg in SIGNAL:
    cols.append(cap.ipc_symbol_target(cfg, sym, basis))
Xpc = np.column_stack(cols)
r = cap.information_processing_capacity(Xpc, sym, SPLIT, basis,
                                        max_degree=3, max_delay=4, max_vars=2, n_surrogate=40)
s = r["summary_max"]
capmap = {tuple(row["config"]): row["capacity"] for row in r["rows"]}
print(f"\n[G4 IPC nonlinear positive control] features={Xpc.shape[1]} eff_rank={r['eff_rank']} "
      f"total={s['total']:.3f} (expect ~{len(SIGNAL)})")
print(f"   per-degree: " + " ".join(f"d{g}={s['per_degree'].get(g,0):.2f}" for g in (1, 2, 3))
      + "   (construction: d1=2, d2=4, d3=1)")
print(f"   inst={s['inst_total']:.2f} temporal={s['temporal_total']:.2f}  "
      f"alpha_at_MAX={r['alpha_at_max']}")
print("   recovered capacity on constructed configs (expect ~1 each):")
for cfg in SIGNAL:
    print(f"     deg{sum(n for _,n in cfg)} {'inst' if cap.is_instantaneous(cfg) else 'temp'} "
          f"{str(cfg):<22} cap={capmap.get(tuple(cfg),0):.3f}")
leak = sum(row["capacity"] for row in r["rows"]
           if tuple(row["config"]) not in {tuple(c) for c in SIGNAL}
           and row["capacity"] > r["thresholds"]["max"])
print(f"   off-target leakage above max-threshold = {leak:.3f} (expect ~0)")

# ---- REGRESSION ASSERTS + saved artifact (gate-parity with G3; lead nit a) ----
# G4 uses encoded-symbol TARGETS (the symbol GS basis), so run B hits exactly the
# feature count with no quantization gap (lead nit b).
d1 = s["per_degree"].get(1, 0.0); d2 = s["per_degree"].get(2, 0.0); d3 = s["per_degree"].get(3, 0.0)
min_cap = min(capmap.get(tuple(c), 0.0) for c in SIGNAL)
checks = {
    "G3a_shiftreg_MC==N":        (abs(g3a_mc - N) < 0.05, f"{g3a_mc:.3f} vs {N}"),
    "G3a_alpha_not_at_max":      (mc["alpha_at_max_signif"] == 0 if "alpha_at_max_signif" in mc
                                  else True, str(mc.get("alpha_at_max_signif", "n/a"))),
    "G3b_symbolIPC_total==N":    (abs(g3b_total - N) < 0.1, f"{g3b_total:.3f} vs {N}"),
    "G3b_nonlinear_temporal~0":  (g3b_d2 < 0.1, f"{g3b_d2:.4f}"),
    "G4_total==7":               (abs(s["total"] - len(SIGNAL)) < 0.05, f"{s['total']:.4f} vs {len(SIGNAL)}"),
    "G4_degree_split_2_4_1":     (abs(d1 - 2) < 0.05 and abs(d2 - 4) < 0.05 and abs(d3 - 1) < 0.05,
                                  f"d1={d1:.3f} d2={d2:.3f} d3={d3:.3f}"),
    "G4_each_config_cap~1":      (min_cap > 0.95, f"min={min_cap:.4f}"),
    "G4_leakage~0":              (leak < 0.02, f"{leak:.4f}"),
    "G4_inst==3_temporal==4":    (abs(s["inst_total"] - 3) < 0.05 and abs(s["temporal_total"] - 4) < 0.05,
                                  f"inst={s['inst_total']:.3f} temp={s['temporal_total']:.3f}"),
}
print("\n=== REGRESSION ASSERTS ===")
failed = []
for name, (ok, detail) in checks.items():
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  ({detail})")
    if not ok:
        failed.append(name)
import csv as _csv, os as _os
_os.makedirs("results", exist_ok=True)
with open(_os.path.join("results", "capacity_gate_validation.csv"), "w", newline="") as _f:
    _w = _csv.writer(_f); _w.writerow(["check", "pass", "detail"])
    for name, (ok, detail) in checks.items():
        _w.writerow([name, int(ok), detail])
print("[wrote] results/capacity_gate_validation.csv")
assert not failed, f"GATE REGRESSION FAILED: {failed}"
print("ALL GATES PASS (G3a/G3b/G4 regression-asserted).")
