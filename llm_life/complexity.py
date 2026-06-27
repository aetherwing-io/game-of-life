"""Information-theoretic complexity diagnostics that ACTUALLY discriminate
Wolfram class — the ones the repo needed but never had.

Background (FINDINGS sec 1, sec 20): the repo's two structural metrics,
``integrated_autocorr_time`` (tau_int) and ``spatial_corr_length`` (xi), were
shown NOT to separate the Wolfram classes — under a fair matched-init protocol
rule-110 (Class 4) reads tau_int=0.47, *below* chaotic rule-30's 1.39, and xi=1
for 110/30/90 alike. tau_int is a temporal autocorrelation of *global activity*
(spatially blind) and xi a spatial autocorrelation of the *change-indicator*
field (configuration-blind), so neither sees rule-110's gliders. The whole
"no edge of chaos" verdict therefore rested on eyeballing space-time PNGs.

This module adds the canonical complexity measures from the CA / computational-
mechanics literature, which are designed to peak at Class 4:

  * ``block_entropies`` / ``entropy_rate``           -- H(n), h_mu (randomness)
  * ``excess_entropy``                                -- E, the predictive
        information stored in spatial configuration (Crutchfield-Feldman). High
        for Class 4, ~0 for Class 3 (no memory) and Class 1/2 (trivial).
  * ``spatial_mutual_information``                    -- I(X_i ; X_{i+d}); a
        low-bias 2-point structure probe. Exactly 0 for additive/PRNG Class-3
        rules (30, 90), nonzero for Class-4 (110, 54).
  * ``local_transfer_entropy``                        -- Lizier's spatially
        resolved information-transfer field; lights up gliders as coherent
        filaments. The first *spatially resolved* diagnostic in the repo.

The (h_mu, E) plane is Langton's complexity-vs-entropy picture: Class 3 lives at
high h_mu / zero E; Class 4 at moderate h_mu / high E; Class 1/2 at low h_mu.

Estimator note. Block/excess entropy from a plug-in estimator is biased when the
alphabet is large relative to the sample count (LLM token fields). Every excess-
entropy routine here therefore also returns a **time-shuffle-null baseline**
(each column permuted in time -> destroys spatial/temporal structure, preserves
per-site marginals and the *identical* finite-sample bias). ``E_excess = E_raw -
E_shuffle`` is the bias-controlled number; it is ~0 for a structureless field of
any alphabet size and only positive when real spatial structure is present.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _relabel(field: np.ndarray) -> np.ndarray:
    """Map arbitrary token ids to a dense 0..K-1 alphabet (keeps block keys small)."""
    _, inv = np.unique(field, return_inverse=True)
    return inv.reshape(field.shape).astype(np.int64)


def binarize(field: np.ndarray, dead_token: int) -> np.ndarray:
    """Coarse-grain a token field to live/dead {0,1}.

    Block/excess entropy is only well-sampled on a SMALL alphabet; a large token
    alphabet saturates H(n) at the sample-count ceiling (for signal AND null
    alike), so structure cancels. Binarizing to live/dead makes an LLM field
    directly commensurate with the binary reference CAs, keeps block entropy
    well-sampled, and is exactly the right coarse-graining for the
    glider/activity-geometry question (gliders are live/dead patterns)."""
    return (field != dead_token).astype(np.int64)


def _entropy_bits(counts: np.ndarray) -> float:
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _iid_marginal_surrogate(field: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A field of the same shape whose cells are drawn IID from the pooled
    single-site token distribution. Destroys ALL spatial structure (even for a
    *frozen* field, where a marginal-preserving time-shuffle would be degenerate
    and reproduce the input), while matching the alphabet and pooled frequencies
    -- so its excess entropy is the pure finite-sample bias floor."""
    flat = field.reshape(-1)
    draw = rng.choice(flat, size=flat.size, replace=True)
    return draw.reshape(field.shape)


# --------------------------------------------------------------------------- #
# block / excess entropy
# --------------------------------------------------------------------------- #
def block_entropies(field: np.ndarray, nmax: int = 8) -> np.ndarray:
    """H(n) in bits for spatial windows of length n=0..nmax, pooled over all rows
    and ring positions. H[0] = 0."""
    field = _relabel(field)
    T, L = field.shape
    base = int(field.max()) + 1
    H = np.zeros(nmax + 1)
    for n in range(1, nmax + 1):
        key = np.zeros((T, L), dtype=np.int64)
        for j in range(n):
            key = key * base + np.roll(field, -j, axis=1)
        _, counts = np.unique(key.reshape(-1), return_counts=True)
        H[n] = _entropy_bits(counts)
    return H


def entropy_rate(field: np.ndarray, nmax: int = 8, tail: int = 3) -> float:
    """h_mu (bits/site): mean of the last `tail` block-entropy increments H(n)-H(n-1)."""
    H = block_entropies(field, nmax)
    return float(np.diff(H)[-tail:].mean())


def excess_entropy(field: np.ndarray, nmax: int = 8, tail: int = 3) -> tuple[float, float]:
    """(E, h_mu): E = H(nmax) - h_mu*nmax is the intercept of the asymptotic
    block-entropy line H(n) ~ h_mu*n + E -- the spatial predictive information."""
    H = block_entropies(field, nmax)
    h_mu = float(np.diff(H)[-tail:].mean())
    E = float(H[nmax] - h_mu * nmax)
    return max(E, 0.0), h_mu


def excess_entropy_corrected(field: np.ndarray, nmax: int = 8, tail: int = 3,
                             seed: int = 0, n_surr: int = 2) -> dict:
    """Bias-controlled excess entropy. Returns E_raw, E_shuffle (finite-sample
    bias floor from an IID-marginal surrogate), E_excess = E_raw - E_shuffle
    (clamped >=0), and h_mu."""
    E_raw, h_mu = excess_entropy(field, nmax, tail)
    rng = np.random.default_rng(seed)
    E_sh = float(np.mean([excess_entropy(_iid_marginal_surrogate(field, rng), nmax, tail)[0]
                          for _ in range(n_surr)]))
    return {"E_raw": E_raw, "E_shuffle": E_sh,
            "E_excess": max(E_raw - E_sh, 0.0), "h_mu": h_mu}


# --------------------------------------------------------------------------- #
# spatial mutual information
# --------------------------------------------------------------------------- #
def _mi_pairs(a: np.ndarray, b: np.ndarray) -> float:
    a = _relabel(a.reshape(1, -1))[0]
    b = _relabel(b.reshape(1, -1))[0]
    na, nb = int(a.max()) + 1, int(b.max()) + 1
    J = np.zeros((na, nb))
    np.add.at(J, (a, b), 1.0)
    J /= J.sum()
    pa = J.sum(1, keepdims=True)
    pb = J.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = J * (np.log2(J) - np.log2(pa) - np.log2(pb))
    return float(np.nansum(terms))


def spatial_mutual_information(field: np.ndarray, d: int = 1) -> float:
    """I(X_i ; X_{i+d}) in bits, pooled over all i (ring) and all rows."""
    a = field.reshape(-1)
    b = np.roll(field, -d, axis=1).reshape(-1)
    return _mi_pairs(a, b)


# --------------------------------------------------------------------------- #
# local transfer entropy  (Lizier's glider filter)
# --------------------------------------------------------------------------- #
def local_transfer_entropy(field: np.ndarray, src: str = "right", k: int = 1
                           ) -> tuple[np.ndarray, float]:
    """Local apparent transfer entropy from a neighbour to the cell.

    t(i,t) = log2 [ p(x_{t+1} | x_t^{(k)}, y_t) / p(x_{t+1} | x_t^{(k)}) ] where
    y is the left/right neighbour. Probabilities are plug-in over all (cell,time)
    samples. Returns (field of shape (T-1-k, L), mean). Coherent positive
    filaments = travelling structures transporting information (gliders);
    a structureless near-zero field = chaos or frozen.
    """
    field = _relabel(field)
    T, L = field.shape
    shift = 1 if src == "left" else -1
    Y = np.roll(field, shift, axis=1)
    base = int(field.max()) + 1
    futs, hists, srcs = [], [], []
    for t in range(k, T - 1):
        futs.append(field[t + 1])
        srcs.append(Y[t])
        key = np.zeros(L, dtype=np.int64)
        for j in range(k):
            key = key * base + field[t - k + 1 + j]
        hists.append(key)
    fut = np.concatenate(futs)
    src_ = np.concatenate(srcs)
    hist = np.concatenate(hists)
    from collections import defaultdict
    n_xh, n_xhy, n_h, n_hy = (defaultdict(float) for _ in range(4))
    for f, h, y in zip(fut, hist, src_):
        n_xh[(h, f)] += 1; n_xhy[(h, y, f)] += 1
        n_h[h] += 1; n_hy[(h, y)] += 1
    loc = np.empty(fut.size)
    for i, (f, h, y) in enumerate(zip(fut, hist, src_)):
        p_x_h = n_xh[(h, f)] / n_h[h]
        p_x_hy = n_xhy[(h, y, f)] / n_hy[(h, y)]
        loc[i] = np.log2(p_x_hy / p_x_h) if (p_x_h > 0 and p_x_hy > 0) else 0.0
    return loc.reshape(T - 1 - k, L), float(loc.mean())


# --------------------------------------------------------------------------- #
# one-call summary
# --------------------------------------------------------------------------- #
def spatial_mi_excess(field: np.ndarray, d: int = 1, seed: int = 0, n_surr: int = 2) -> float:
    """Bias-corrected spatial MI: I(d) minus the IID-marginal surrogate floor
    (the finite-sample positive bias of the plug-in MI estimator)."""
    rng = np.random.default_rng(seed)
    raw = spatial_mutual_information(field, d)
    floor = float(np.mean([
        spatial_mutual_information(_iid_marginal_surrogate(field, rng), d)
        for _ in range(n_surr)]))
    return max(raw - floor, 0.0)


def complexity_summary(field: np.ndarray, nmax: int = 8, seed: int = 0,
                       dead_token: int | None = None) -> dict:
    """All scalar diagnostics for a (T+1, L) integer state field, post-burn.

    The PRIMARY (h_mu, E) plane is computed on the live/dead binarized field
    (well-sampled, comparable to the binary reference CAs) when ``dead_token`` is
    given. Full-token bias-corrected spatial MI is reported as a secondary,
    alphabet-aware structure probe."""
    bin_field = binarize(field, dead_token) if dead_token is not None else field
    ec = excess_entropy_corrected(bin_field, nmax=nmax, seed=seed)
    te_r = local_transfer_entropy(bin_field, "right", 1)[1]
    return {
        "h_mu": ec["h_mu"],
        "E_raw": ec["E_raw"],
        "E_shuffle": ec["E_shuffle"],
        "E_excess": ec["E_excess"],
        "MI1_bin": spatial_mi_excess(bin_field, 1, seed),
        "MI1_tok": spatial_mi_excess(field, 1, seed),
        "MI2_tok": spatial_mi_excess(field, 2, seed),
        "TE_right_bin": te_r,
        "live_frac": float((bin_field == 1).mean()),
        "site_entropy_tok": _entropy_bits(np.unique(field, return_counts=True)[1]),
    }
