"""Licensed capacity estimators for the §25 reservoir study.

This module supersedes the quick MC/IPC in ``reservoir.py`` with the disciplined
versions the methodology review (task #41) requires, so the magnitudes quoted in
FINDINGS §25 are trustworthy:

1.  A WIDE ridge-α grid with an INTERIORITY guarantee. The first pass pinned the
    validation-selected α at the grid maximum (100) for every lag -- the optimum
    was outside the grid, so those magnitudes were not licensed. Here the grid
    spans 1e-6..1e8 and the caller asserts the selected α is interior (never the
    min/max) or hard-fails.
2.  IPC over an ENCODED-SYMBOL Gram-Schmidt basis: orthonormal polynomials of the
    *binned input symbol* (what the reservoir actually sees), not continuous
    Legendre. Exactly orthonormal under the empirical (uniform) symbol measure, so
    capacities sum cleanly and the total is properly bounded by readout rank.
3.  An INSTANTANEOUS (all delays = 0) vs TEMPORAL (any delay >= 1) split, per
    degree. Instantaneous = a static nonlinear map of the *current* input;
    temporal = genuine computation over time / memory. This 2x2 (degree x
    inst/temporal) is the load-bearing decomposition for §25.
4.  A SHUFFLED-INPUT surrogate floor (rebuild the same basis targets from a
    time-permuted symbol stream) reported at the {99, 99.9, max} percentiles --
    the by-chance capacity a finite sample grants, subtracted before summing.

Ridge is solved once via an eigendecomposition of the centered Gram matrix and
reused across every α and every target (centering absorbs the intercept), so the
thousands of target fits an IPC sweep needs stay cheap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .reservoir import Split, enumerate_ipc_configs, make_split  # noqa: F401  (re-exported)

# Wide grid with headroom on both ends; ~2 points per decade.
ALPHA_GRID = tuple(float(a) for a in np.logspace(-6, 8, 29))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


class RidgePath:
    """Ridge readout for a FIXED design matrix, amortized across α and targets.

    Centering the training features and target removes the intercept, so ridge
    reduces to ``w(α) = V diag(1/(λ+α)) Vᵀ Xcᵀ yc`` with a single eigendecomposition
    ``Xcᵀ Xc = V diag(λ) Vᵀ`` reused for every α and every target.
    """

    def __init__(self, X: np.ndarray, split: Split, alphas=ALPHA_GRID):
        self.alphas = np.asarray(alphas, dtype=float)
        self.tr = np.arange(split.train.start, split.train.stop)
        self.va = np.arange(split.val.start, split.val.stop)
        self.te = np.arange(split.test.start, split.test.stop)
        Xtr = X[self.tr]
        self.mu = Xtr.mean(axis=0, keepdims=True)
        Xc = Xtr - self.mu
        A = Xc.T @ Xc
        lam, V = np.linalg.eigh(A)
        self.lam = np.clip(lam, 0.0, None)
        self.V = V
        self.Xc_tr = Xc
        # precompute centered val/test designs projected into eigenbasis
        self.Pva = (X[self.va] - self.mu) @ V          # (n_va, d)
        self.Pte = (X[self.te] - self.mu) @ V          # (n_te, d)

    def capacity(self, y: np.ndarray) -> dict:
        """Validation-selected ridge test R² for target ``y`` (capacity, clipped
        at 0), plus the selected α and whether it sits on the grid boundary."""
        ytr = y[self.tr]
        ybar = float(ytr.mean())
        g = self.V.T @ (self.Xc_tr.T @ (ytr - ybar))    # (d,)
        yva, yte = y[self.va], y[self.te]
        best = (-np.inf, -1, float("nan"))
        for i, a in enumerate(self.alphas):
            coef = g / (self.lam + a)                   # (d,)
            pred_va = self.Pva @ coef + ybar
            r = r2_score(yva, pred_va)
            if r > best[0]:
                best = (r, i, a)
        _, bi, ba = best
        coef = g / (self.lam + ba)
        pred_te = self.Pte @ coef + ybar
        cap = max(0.0, r2_score(yte, pred_te))
        return {"capacity": cap, "alpha": ba, "alpha_idx": bi,
                # only the MAX boundary signals an under-explored grid (the bug we
                # are fixing); the MIN boundary just means no regularization helps
                # (optimal alpha -> 0), which is benign for clean targets.
                "at_max": bi == len(self.alphas) - 1,
                "at_min": bi == 0,
                "val_r2": best[0]}


def effective_rank(X: np.ndarray) -> int:
    return int(np.linalg.matrix_rank(X - X.mean(axis=0, keepdims=True)))


# --------------------------------------------------------------------------- #
# Memory capacity (continuous u(t-k) targets) -- the classic Jaeger number
# --------------------------------------------------------------------------- #
def memory_capacity(X: np.ndarray, u: np.ndarray, split: Split, kmax: int,
                    alphas=ALPHA_GRID) -> dict:
    """MC_k = test R² reconstructing the input variable ``u`` at lag k, ``u(t-k)``;
    MC = Σ_k MC_k. ``u`` is whatever input variable the caller passes -- pass the
    ENCODED bin index b(t) (what the reservoir receives) to make degree-1 IPC == MC
    exactly; passing continuous u differs only ~0.5% (quantization).
    Requires ``washout > kmax`` so every train/val/test row has a valid u(t-k)."""
    assert split.washout > kmax, "washout must exceed kmax so u(t-k) is defined"
    rp = RidgePath(X, split, alphas)
    mc_k, alpha_k, at_max, at_min, at_max_signif = [], [], 0, 0, 0
    T = len(u)
    for k in range(kmax + 1):
        y = np.zeros(T)
        if k == 0:
            y = u.copy()
        else:
            y[k:] = u[: T - k]
        r = rp.capacity(y)
        mc_k.append(r["capacity"]); alpha_k.append(r["alpha"])
        at_max += int(r["at_max"]); at_min += int(r["at_min"])
        # a MAX-boundary hit only matters where capacity is non-negligible: for a
        # genuinely zero-capacity lag, alpha->inf (shrink to mean) is CORRECT.
        at_max_signif += int(r["at_max"] and r["capacity"] > 0.01)
    mc_k = np.array(mc_k)
    return {"mc_k": mc_k, "MC": float(mc_k.sum()), "alpha_k": np.array(alpha_k),
            "kmax": kmax, "readout_dim": int(X.shape[1]),
            "alpha_at_max": at_max, "alpha_at_min": at_min,
            "alpha_at_max_signif": at_max_signif, "n_fits": kmax + 1}


# --------------------------------------------------------------------------- #
# Encoded-symbol Gram-Schmidt basis
# --------------------------------------------------------------------------- #
def symbol_poly_basis(n_bins: int, max_degree: int) -> np.ndarray:
    """(n_bins, max_degree+1) orthonormal polynomial basis on the symbol index,
    Gram-Schmidt'd under the uniform measure over the ``n_bins`` symbols. Column
    ``n`` is φ_n; φ_0≡1, E[φ_n]=0 (n≥1), E[φ_n φ_m]=δ_nm for i.i.d. uniform
    symbols. The discrete analogue of normalized Legendre, matched to the bins
    the reservoir actually sees."""
    x = np.linspace(-1.0, 1.0, n_bins)
    M = np.vander(x, max_degree + 1, increasing=True).astype(float)  # [1, x, x², ...]
    Q = np.zeros_like(M)
    for j in range(max_degree + 1):
        v = M[:, j].copy()
        for i in range(j):
            v -= (Q[:, i] @ v / n_bins) * Q[:, i]
        v /= np.sqrt((v @ v) / n_bins)
        Q[:, j] = v
    return Q


def ipc_symbol_target(config, symbols: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Target for an IPC config = product of φ_{n_i}(symbol(t-k_i))."""
    T = len(symbols)
    y = np.ones(T)
    kmax = max(k for k, _ in config)
    for k, n in config:
        sh = np.zeros(T, dtype=np.int64)
        sh[k:] = symbols[: T - k]
        y = y * basis[sh, n]
    y[:kmax] = 0.0
    return y


def is_instantaneous(config) -> bool:
    """A target is instantaneous iff every factor reads delay 0 (a function of the
    CURRENT input only). With distinct delays per config this means a single
    delay-0 factor."""
    return max(k for k, _ in config) == 0


# --------------------------------------------------------------------------- #
# IPC over the symbol basis, with inst/temporal x degree split + shuffled floor
# --------------------------------------------------------------------------- #
def information_processing_capacity(
    X: np.ndarray,
    symbols: np.ndarray,
    split: Split,
    basis: np.ndarray,
    max_degree: int = 4,
    max_delay: int = 12,
    max_vars: int = 2,
    alphas=ALPHA_GRID,
    n_surrogate: int = 48,
    rng_seed: int = 0,
) -> dict:
    """Dambre IPC on the encoded-symbol Gram-Schmidt basis.

    Returns capacity decomposed by degree and by instantaneous/temporal, the
    selected-α boundary-hit count, and a shuffled-INPUT surrogate floor reported
    at the {99, 99.9, max} percentiles. The headline totals threshold each config
    capacity at the surrogate MAX (the most conservative choice)."""
    rp = RidgePath(X, split, alphas)
    configs = list(enumerate_ipc_configs(max_degree, max_delay, max_vars))

    rows, at_max, at_min = [], 0, 0
    for total, config in configs:
        y = ipc_symbol_target(config, symbols, basis)
        r = rp.capacity(y)
        at_max += int(r["at_max"]); at_min += int(r["at_min"])
        rows.append({"degree": total, "config": config,
                     "inst": is_instantaneous(config), "capacity": r["capacity"],
                     "at_max": r["at_max"]})

    # shuffled-INPUT surrogate: rebuild the SAME basis targets from a time-permuted
    # symbol stream (destroys the reservoir<->target temporal relation) -> the
    # by-chance capacity floor. The floor is DEGREE-STRATIFIED: higher-degree
    # Legendre/symbol targets have heavier tails and a finite sample fits them
    # spuriously better, so a single global floor under-catches high-degree
    # structured bias (which is *seed-reproducible*, so cross-seed stability does
    # not rule it out). We therefore recompute the null PER DEGREE and threshold
    # each config against its own degree-matched floor (gate #47). All configs are
    # used in the null (no subsampling) so every degree's tail is well populated.
    rng = np.random.default_rng(rng_seed)
    null_by_deg = {d: [] for d in range(1, max_degree + 1)}
    for _ in range(n_surrogate):
        sym_sh = rng.permutation(symbols)
        for total, config in configs:
            y = ipc_symbol_target(config, sym_sh, basis)
            null_by_deg[total].append(rp.capacity(y)["capacity"])
    thr_deg = {}
    for d, vals in null_by_deg.items():
        v = np.array(vals) if vals else np.array([0.0])
        thr_deg[d] = {"p99": float(np.quantile(v, 0.99)),
                      "p999": float(np.quantile(v, 0.999)), "max": float(v.max())}
    all_null = np.concatenate([np.array(v) if v else np.array([0.0])
                               for v in null_by_deg.values()])
    thr = {"p99": float(np.quantile(all_null, 0.99)),       # degree-agnostic (legacy)
           "p999": float(np.quantile(all_null, 0.999)), "max": float(all_null.max())}

    def summarize(level):
        """level in {max,p999,p99}: threshold each config against its DEGREE's
        floor at that level (degree-stratified)."""
        per_deg, per_cell, total = {}, {}, 0.0
        for row in rows:
            d = row["degree"]
            floor = thr_deg[d][level]
            c = row["capacity"] if row["capacity"] > floor else 0.0
            cell = "inst" if row["inst"] else "temporal"
            per_deg[d] = per_deg.get(d, 0.0) + c
            per_cell[(d, cell)] = per_cell.get((d, cell), 0.0) + c
            total += c
        inst_total = sum(v for (d, cell), v in per_cell.items() if cell == "inst")
        temp_total = sum(v for (d, cell), v in per_cell.items() if cell == "temporal")
        return {"total": total, "per_degree": per_deg, "per_cell": per_cell,
                "inst_total": inst_total, "temporal_total": temp_total}

    # significance vs the degree-matched MAX floor
    at_max_signif = sum(int(row["at_max"] and row["capacity"] > thr_deg[row["degree"]]["max"])
                        for row in rows)
    out = {"rows": rows, "thresholds": thr, "thresholds_by_degree": thr_deg,
           "alpha_at_max": at_max, "alpha_at_min": at_min,
           "alpha_at_max_signif": at_max_signif,
           "readout_dim": int(X.shape[1]), "eff_rank": effective_rank(X[rp.tr]),
           "n_configs": len(configs),
           "caps": {"max_degree": max_degree, "max_delay": max_delay, "max_vars": max_vars}}
    for name in ("max", "p999", "p99"):
        out[f"summary_{name}"] = summarize(name)
    return out
