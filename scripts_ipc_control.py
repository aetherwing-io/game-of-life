"""GATE G4 — nonlinear POSITIVE control for the Dambre IPC estimator.

G3 proved the estimator's linear recovery (shift register -> MC=N) and its
nonlinear TRUE NEGATIVE (degree>=2 == 0 on a purely linear reservoir). G4 proves
the complementary claim the degree>=2 IPC headline depends on: the estimator
correctly MEASURES degree>=2 capacity WHEN IT IS PRESENT.

We build a synthetic reservoir whose feature columns ARE a known set of
encoded-symbol Legendre-product basis functions, including
  * degree-2 single-variable   Phat_2(b_{t-k})
  * degree-2 two-variable       Phat_1(b_{t-k1}) * Phat_1(b_{t-k2})
  * degree-3 single-variable    Phat_3(b_{t-k})
and feed them through the SAME rz.information_processing_capacity used for the
LLM. It must recover capacity ~ 1 on exactly the constructed configs, ~0
elsewhere, total ~ number of independent features (== eff_rank), with the degree
split matching the construction.

Critically the control is built in the ENCODED-SYMBOL basis: the continuous input
u(t) is binned to a symbol exactly as the LLM reservoir bins it, and the features
are Legendre products of the *bin-center value* of that symbol -- so the test
exercises the real binned pipeline (targets are still continuous-u Legendre, as
in the LLM run). The small gap from 1.0 is the documented 16-bin quantization
ceiling, which we also measure; it confirms the LLM's far-lower capacity is not a
binning artifact.

Run:  python scripts_ipc_control.py
"""
import os, warnings
warnings.filterwarnings("ignore", message=".*encountered in matmul.*")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
from llm_life import reservoir as rz

T = 3000
N_BINS = 16
SPLIT = rz.make_split(T, 200, 1800, 400, 600)
# the ground-truth signal configs (each becomes one feature column)
SIGNAL = [
    ((0, 1),),            # degree 1 (linear, delay 0)
    ((4, 1),),            # degree 1 (linear, delay 4)
    ((0, 2),),            # degree 2 single-var, delay 0
    ((2, 2),),            # degree 2 single-var, delay 2
    ((0, 3),),            # degree 3 single-var, delay 0
    ((1, 1), (3, 1)),     # degree 2 two-var (delays 1,3)
    ((0, 1), (2, 1)),     # degree 2 two-var (delays 0,2)
]


def legendre_product(config, stream):
    y = np.ones(len(stream))
    kmax = max(k for k, _ in config)
    for k, n in config:
        sh = np.zeros(len(stream))
        sh[k:] = stream[: len(stream) - k]
        y = y * rz.norm_legendre(n, sh)
    y[:kmax] = 0.0
    return y


def run_control(label, stream, u_targets):
    """stream = the encoding the FEATURES are built from; u_targets = the
    continuous input the IPC TARGETS are built from (the real pipeline path)."""
    X = np.column_stack([legendre_product(c, stream) for c in SIGNAL])
    erank = int(np.linalg.matrix_rank(X[SPLIT.train] - X[SPLIT.train].mean(0)))
    Xz, = rz.standardize(X[SPLIT.train], X)
    r = rz.information_processing_capacity(
        Xz, u_targets, SPLIT, max_degree=3, max_delay=4, max_vars=2, n_surrogate=48)
    # map recovered capacity back onto configs
    cap = {tuple(row["config"]): row["capacity_thr"] for row in r["rows"]}
    print(f"\n== {label} ==  features={X.shape[1]} eff_rank={erank}  "
          f"IPC_total={r['total']:.3f}  threshold={r['threshold']:.4f}")
    deg = r["per_degree"]
    print("  per-degree:", " ".join(f"d{g}={deg.get(g,0.0):.3f}" for g in (1, 2, 3)))
    print("  recovered capacity on the CONSTRUCTED configs (expect ~ceiling):")
    constructed_total = 0.0
    for c in SIGNAL:
        v = cap.get(tuple(c), 0.0)
        constructed_total += v
        print(f"    deg{sum(n for _,n in c)} {str(c):<22} cap={v:.3f}")
    # leakage: capacity on configs NOT constructed
    leak = sum(row["capacity_thr"] for row in r["rows"]
               if tuple(row["config"]) not in {tuple(c) for c in SIGNAL})
    print(f"  constructed-config total={constructed_total:.3f}  "
          f"off-target leakage={leak:.3f}")
    return r


def main():
    rng = np.random.default_rng(7)
    u = rng.uniform(-1.0, 1.0, size=T)

    # encode exactly as the LLM reservoir does: bin -> bin-center value
    b = rz.bin_input(u, N_BINS)
    centers = -1.0 + (np.arange(N_BINS) + 0.5) * 2.0 / N_BINS
    uc = centers[b]

    print("GATE G4: IPC nonlinear positive control")
    print(f"  SIGNAL configs (degrees): "
          f"{[sum(n for _,n in c) for c in SIGNAL]}  "
          f"-> deg1={sum(1 for c in SIGNAL if sum(n for _,n in c)==1)}, "
          f"deg2={sum(1 for c in SIGNAL if sum(n for _,n in c)==2)}, "
          f"deg3={sum(1 for c in SIGNAL if sum(n for _,n in c)==3)}")

    # (A) continuous-Legendre reference: features AND targets from continuous u.
    #     The estimator should be EXACT here (total == #features).
    run_control("A: continuous-Legendre reference (exactness)", u, u)

    # (B) encoded-symbol control (the real pipeline): features from the binned
    #     symbol's bin-center value; targets from continuous u.
    run_control("B: ENCODED-SYMBOL control (real binned pipeline)", uc, u)

    # (C) measure the pure 16-bin quantization ceiling: one feature = Phat_1(uc),
    #     target = Phat_1(u); best linear R^2 is the per-term ceiling.
    f = rz.norm_legendre(1, uc)
    Xz, = rz.standardize(f[SPLIT.train, None], f[:, None])
    ceil = rz._target_capacity(Xz, rz.norm_legendre(1, u), SPLIT,
                               alphas=(1e-4, 1e-3, 1e-2, 1e-1, 1.0))
    print(f"\n[quantization ceiling] degree-1 single-term R^2 with {N_BINS} bins = {ceil:.4f}")
    print("  -> encoded-symbol recovery should sit at ~ this ceiling per term, and the")
    print("     LLM reservoir's measured IPC (~0.66) is far below it, so binning is NOT")
    print("     the limiter on the LLM result.")


if __name__ == "__main__":
    main()
