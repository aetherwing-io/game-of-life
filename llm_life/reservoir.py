"""Driven-reservoir apparatus over the causal LLM cellular-automaton map.

This builds directly on the §23 *consistency* result: the causal full-attention
map, iterated under a SHARED Gumbel-noise realization, is a consistent map -- two
replicas started from different states synchronize. Consistency is exactly the
**echo-state property** (ESP): the reservoir state becomes a deterministic
function of the input history, with the influence of the initial condition
washing out. ESP is the precondition for reservoir computing, which is why the
causal map is our reservoir.

Apparatus
---------
The lattice is ``L`` token sites. A small set of *input sites* is clamped each
step to a token drawn from a fixed codebook indexed by binning the scalar input
``u(t) in [-1, 1]``. The remaining ``L - n_in`` sites are the reservoir.

* **Geometry.** Input sites sit at the far LEFT (lowest indices). Causal
  attention is leftward, so a token at position ``i`` is in the left-context of
  every reservoir site ``j > i``; placing the drive at the left maximizes its
  reach into the reservoir.
* **Determinism.** A FIXED shared-Gumbel noise sequence (seeded once, replayed
  identically for every run/replica) makes the reservoir a deterministic
  function of ``(input stream, initial lattice)`` -- this is the §23
  coupled-noise machinery, re-used verbatim via ``BaseAutomaton.step(noise=...)``.
* **Recursion.** With ``s`` carrying the reservoir state ``x(t-1)``::

      s[input_sites] = codebook[bin(u(t))]      # inject u(t)
      s = auto.step(s, T, noise=noise[t])        # model reads u(t) + x(t-1)
      x(t) = readout(s[reservoir_sites])         # x(t) = f(x(t-1), u(t))

  so the recorded state at time ``t`` is the canonical driven-reservoir state, a
  function of inputs up to and including ``u(t)``.
* **Readout.** Each reservoir site's token id is mapped to its top-K PCA
  projection of the model input-embedding matrix (a K-dim generalization of
  ``viz.embedding_rgb_table``'s top-3 RGB) and concatenated over reservoir sites
  -> a real vector of dimension ``(L - n_in) * K``. The reservoir stays the
  TOKEN lattice, not hidden activations -- faithful to the project.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .automaton import StepNoise
from .sampler import gumbel_like


# --------------------------------------------------------------------------- #
# Readout: top-K embedding-PCA projection table
# --------------------------------------------------------------------------- #
def pca_readout_table(embedding_matrix: np.ndarray, k: int) -> np.ndarray:
    """(V, d) embeddings -> (V, K) top-K PCA projection, standardized per
    component (zero mean, unit variance across the vocabulary).

    This is the K-dim generalization of ``viz.embedding_rgb_table`` (which uses
    top-3 + a [0,1] colour rescale). For a regression readout we instead
    standardize each component so the feature blocks are comparably scaled and
    the ridge problem is well-conditioned.
    """
    X = np.nan_to_num(embedding_matrix.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    X = X - X.mean(axis=0, keepdims=True)
    if X.shape[0] > 60000:
        rng = np.random.default_rng(0)
        sample = X[rng.choice(X.shape[0], 60000, replace=False)]
        _, _, Vt = np.linalg.svd(sample, full_matrices=False)
    else:
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
    proj = X @ Vt[:k].T                       # (V, K)
    proj = np.nan_to_num(proj)
    std = proj.std(axis=0, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return proj / std


# --------------------------------------------------------------------------- #
# Input codebook: scalar u in [-1,1] -> token id
# --------------------------------------------------------------------------- #
def build_codebook(
    n_bins: int,
    vocab: int,
    embedding_matrix: np.ndarray | None = None,
    mode: str = "pc1",
    seed: int = 0,
    exclude: tuple[int, ...] = (),
) -> np.ndarray:
    """Return an (n_bins,) array of distinct token ids, one per input bin.

    mode:
      * ``"pc1"``   tokens whose PC1 embedding projection is at evenly spaced
        quantiles -- a monotone, maximally-spread-in-embedding-space map from
        ``u`` to input token (requires ``embedding_matrix``). Injective and
        clean: larger ``u`` -> embedding further along the dominant axis.
      * ``"random"`` distinct random token ids (a fixed seed makes it
        reproducible). No geometric relation between adjacent bins.
      * ``"evenly"`` evenly spaced token ids across the vocabulary.
    """
    excl = set(int(e) for e in exclude)
    if mode == "pc1" and embedding_matrix is not None:
        X = np.nan_to_num(embedding_matrix.astype(np.float64))
        X = X - X.mean(axis=0, keepdims=True)
        # PC1 direction via a cheap power-free SVD on a row sample if huge
        if X.shape[0] > 60000:
            rng = np.random.default_rng(0)
            Xs = X[rng.choice(X.shape[0], 60000, replace=False)]
            _, _, Vt = np.linalg.svd(Xs, full_matrices=False)
        else:
            _, _, Vt = np.linalg.svd(X, full_matrices=False)
        pc1 = X @ Vt[0]                       # (V,)
        order = np.argsort(pc1)
        order = np.array([i for i in order if i not in excl])
        # evenly spaced quantiles along the sorted PC1 axis
        idx = np.linspace(0, len(order) - 1, n_bins).round().astype(int)
        return order[idx]
    if mode == "evenly":
        ids = np.linspace(0, vocab - 1, n_bins + 2).round().astype(int)[1:-1]
        return ids
    # random
    rng = np.random.default_rng(seed)
    pool = [i for i in range(vocab) if i not in excl]
    return rng.choice(pool, size=n_bins, replace=False)


def bin_input(u: np.ndarray, n_bins: int) -> np.ndarray:
    """Map u in [-1,1] -> bin index in [0, n_bins-1]."""
    b = np.floor((np.clip(u, -1.0, 1.0) + 1.0) / 2.0 * n_bins).astype(int)
    return np.clip(b, 0, n_bins - 1)


# --------------------------------------------------------------------------- #
# Fixed shared-noise sequence (the §23 consistency machinery)
# --------------------------------------------------------------------------- #
def make_noise(steps: int, L: int, vocab: int, device: str, seed: int) -> list[StepNoise]:
    """A reproducible list of ``steps`` Gumbel tensors of shape (L, V). Replaying
    this identical list across runs/replicas is what makes the driven map a
    deterministic, consistent reservoir."""
    gen_dev = "cpu" if device == "mps" else device
    g = torch.Generator(device=gen_dev)
    g.manual_seed(seed)
    template = torch.empty(L, vocab, dtype=torch.float32, device=device)
    return [StepNoise(gumbel=gumbel_like(template, generator=g)) for _ in range(steps)]


# --------------------------------------------------------------------------- #
# The driven reservoir
# --------------------------------------------------------------------------- #
@dataclass
class DrivenReservoir:
    """Drive the causal LLM-CA map with a scalar input stream under fixed noise.

    Parameters carried:
      * ``auto``        an LLMAutomaton (causal map).
      * ``L``           lattice length.
      * ``vocab``       vocabulary size.
      * ``codebook``    (n_bins,) token ids, one per input bin.
      * ``input_sites`` indices clamped to the input token each step.
      * ``temp``        sampling temperature (default 0.7).
      * ``absorbing``   absorbing vacuum rule (default False; the §23 causal
                        consistency was established in the soft regime).
    """

    auto: object
    L: int
    vocab: int
    codebook: np.ndarray
    input_sites: np.ndarray
    temp: float = 0.7
    absorbing: bool = False
    device: str = "cpu"
    reservoir_sites: np.ndarray = field(init=False)

    def __post_init__(self):
        self.input_sites = np.asarray(self.input_sites, dtype=np.int64)
        mask = np.ones(self.L, dtype=bool)
        mask[self.input_sites] = False
        self.reservoir_sites = np.nonzero(mask)[0]

    @property
    def n_bins(self) -> int:
        return len(self.codebook)

    def random_init(self, seed: int) -> torch.Tensor:
        gen_dev = "cpu" if self.device == "mps" else self.device
        g = torch.Generator(device=gen_dev)
        g.manual_seed(seed)
        return torch.randint(0, self.vocab, (self.L,), generator=g).to(self.device)

    @torch.no_grad()
    def run(self, u: np.ndarray, init: torch.Tensor, noises: list[StepNoise]) -> np.ndarray:
        """Drive the reservoir with input stream ``u`` (shape (T,)) from ``init``
        using the fixed noise list ``noises`` (len >= T). Returns the post-step
        state trajectory ``states`` of shape (T, L) as an int64 numpy array,
        where ``states[t]`` is x(t) = f(x(t-1), u(t)) on the reservoir sites
        (input sites hold whatever was sampled there and are ignored downstream).
        """
        T = len(u)
        assert len(noises) >= T, "need at least one noise tensor per step"
        bins = bin_input(np.asarray(u, dtype=np.float64), self.n_bins)
        in_tokens = torch.as_tensor(self.codebook[bins], dtype=torch.long, device=self.device)
        in_sites = torch.as_tensor(self.input_sites, dtype=torch.long, device=self.device)
        s = init.clone()
        states = np.empty((T, self.L), dtype=np.int64)
        for t in range(T):
            s = s.clone()
            s[in_sites] = in_tokens[t]                      # inject u(t)
            s, _ = self.auto.step(s, self.temp, self.absorbing, noise=noises[t])
            states[t] = s.detach().to("cpu").numpy()
        return states

    def features(self, states: np.ndarray, table: np.ndarray,
                 sites: np.ndarray | str | None = None) -> np.ndarray:
        """(T, L) ids + (V, K) readout table -> (T, len(sites)*K) feature matrix.

        ``sites`` selects which lattice columns the readout reads:
          * ``None`` (default) -> the reservoir sites (input sites EXCLUDED; this
            is the headline readout -- capacity must come from the reservoir, not
            from reading the clamped input back off itself);
          * ``"input"`` -> only the clamped input sites (the leak control: how
            much "capacity" is just the input copied off its own sites);
          * ``"all"`` -> every site;
          * an explicit array of indices.
        """
        if sites is None:
            cols = self.reservoir_sites
        elif isinstance(sites, str):
            cols = {"input": self.input_sites, "all": np.arange(self.L),
                    "reservoir": self.reservoir_sites}[sites]
        else:
            cols = np.asarray(sites, dtype=np.int64)
        feats = table[states[:, cols]]                     # (T, len(cols), K)
        return feats.reshape(feats.shape[0], -1)


# --------------------------------------------------------------------------- #
# Ridge readout + Memory Capacity
# --------------------------------------------------------------------------- #
def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Closed-form ridge with an explicit intercept column. Returns weights
    ``w`` of shape (D+1,) for the design ``[X, 1]``. The intercept (last column)
    is NOT regularized."""
    n, d = X.shape
    Xa = np.hstack([X, np.ones((n, 1))])
    A = Xa.T @ Xa
    reg = alpha * np.eye(d + 1)
    reg[-1, -1] = 0.0                                      # don't penalize intercept
    w = np.linalg.solve(A + reg, Xa.T @ y)
    return w


def ridge_predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.hstack([X, np.ones((X.shape[0], 1))]) @ w


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


@dataclass
class Split:
    """Index ranges for a washout / train / validation / test partition of a
    single driven run."""

    washout: int
    train: slice
    val: slice
    test: slice


def make_split(T: int, washout: int, train: int, val: int, test: int) -> Split:
    a = washout
    return Split(
        washout=washout,
        train=slice(a, a + train),
        val=slice(a + train, a + train + val),
        test=slice(a + train + val, a + train + val + test),
    )


def memory_capacity(
    X: np.ndarray,
    u: np.ndarray,
    split: Split,
    kmax: int,
    alphas=(1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0),
) -> dict:
    """SUPERSEDED — α-railed (narrow grid, max 100). The canonical estimator is
    ``llm_life.capacity.memory_capacity`` (wide-α 1e-6…1e8 + interiority assert,
    encoded-symbol Gram-Schmidt basis); see FINDINGS §25. Kept for provenance only —
    its grid pinned at the maximum, under-counting capacity (MC 0.23→0.41 once widened).

    Linear Memory Capacity (Jaeger 2002).

    For each lag ``k`` in ``0..kmax`` fit a ridge readout from the reservoir
    feature ``X(t)`` to the past input ``u(t-k)``; ``MC_k`` is the **test** R^2.
    The ridge ``alpha`` is selected per lag on the **validation** split (never
    the test split). ``MC = sum_k MC_k`` (clipped at 0 per lag, the standard
    convention). Returns per-lag arrays and totals.
    """
    T = X.shape[0]
    mc_k, alpha_k, mc_k_val = [], [], []
    for k in range(kmax + 1):
        # target: u shifted by k; valid where t-k >= 0
        def slc(s: slice):
            idx = np.arange(s.start, s.stop)
            idx = idx[idx - k >= 0]
            return idx
        tr, va, te = slc(split.train), slc(split.val), slc(split.test)
        Xtr, ytr = X[tr], u[tr - k]
        Xva, yva = X[va], u[va - k]
        Xte, yte = X[te], u[te - k]
        best_a, best_val = alphas[0], -np.inf
        for a in alphas:
            w = ridge_fit(Xtr, ytr, a)
            r = r2_score(yva, ridge_predict(Xva, w))
            if r > best_val:
                best_val, best_a = r, a
        w = ridge_fit(Xtr, ytr, best_a)
        r_test = r2_score(yte, ridge_predict(Xte, w))
        mc_k.append(max(0.0, r_test))
        mc_k_val.append(best_val)
        alpha_k.append(best_a)
    mc_k = np.array(mc_k)
    return {
        "mc_k": mc_k,
        "mc_k_val": np.array(mc_k_val),
        "alpha_k": np.array(alpha_k),
        "MC": float(mc_k.sum()),
        "kmax": kmax,
        "readout_dim": int(X.shape[1]),
    }


def standardize(Xtrain_ref: np.ndarray, *mats: np.ndarray):
    """Z-score features using statistics from ``Xtrain_ref`` only (no test
    leakage). Returns the standardized versions of every matrix in ``mats``."""
    mu = Xtrain_ref.mean(axis=0, keepdims=True)
    sd = Xtrain_ref.std(axis=0, keepdims=True)
    sd = np.where(sd == 0, 1.0, sd)
    return tuple((M - mu) / sd for M in mats)


# --------------------------------------------------------------------------- #
# Information Processing Capacity (Dambre et al. 2012)
# --------------------------------------------------------------------------- #
# IPC generalizes Memory Capacity to a complete orthonormal basis of functions
# of the input history. For i.i.d. ``u ~ Uniform[-1,1]`` the right basis is
# products of *normalized Legendre* polynomials of delayed inputs:
#
#     y_{config}(t) = prod_i  Phat_{n_i}( u(t - k_i) ),   Phat_n(x) = sqrt(2n+1) P_n(x)
#
# with distinct delays k_i and degrees n_i >= 1. These are orthonormal:
# E[y]=0, E[y^2]=1, E[y_a y_b]=delta_ab (i.i.d. input). The capacity of a target
# is the linear-readout test R^2; the TOTAL capacity summed over the basis is
# bounded by the number of linearly independent reservoir states (readout rank).
# Grouping configs by total degree (sum_i n_i) separates LINEAR memory
# (degree 1, == MC_k) from NONLINEAR processing (degree >= 2).
from itertools import combinations


def norm_legendre(n: int, x: np.ndarray) -> np.ndarray:
    """Normalized Legendre Phat_n(x) = sqrt(2n+1) P_n(x); orthonormal under
    Uniform[-1,1]."""
    from scipy.special import eval_legendre
    return np.sqrt(2 * n + 1) * eval_legendre(n, x)


def enumerate_ipc_configs(max_degree: int, max_delay: int, max_vars: int):
    """All IPC target configurations up to ``max_degree`` total degree, delays in
    ``0..max_delay``, and at most ``max_vars`` distinct delays per target.

    A config is a tuple of (delay, degree) pairs with distinct delays and
    degree >= 1. Yields (total_degree, config).
    """
    delays = list(range(max_delay + 1))
    seen = set()
    for nvars in range(1, max_vars + 1):
        for delay_combo in combinations(delays, nvars):
            # distribute degrees: each chosen delay gets degree >= 1, total <= max_degree
            for degs in _degree_assignments(nvars, max_degree):
                total = sum(degs)
                if total < nvars:  # each var needs >=1
                    continue
                config = tuple(sorted(zip(delay_combo, degs)))
                if config in seen:
                    continue
                seen.add(config)
                yield total, config


def _degree_assignments(nvars: int, max_degree: int):
    """All tuples of ``nvars`` positive integers summing to <= max_degree."""
    if nvars == 1:
        for d in range(1, max_degree + 1):
            yield (d,)
        return
    for first in range(1, max_degree + 1):
        for rest in _degree_assignments(nvars - 1, max_degree - first):
            yield (first,) + rest


def ipc_target(config, u: np.ndarray) -> np.ndarray:
    """Build the (T,) target signal for an IPC config (product of normalized
    Legendre polynomials of delayed inputs). Entries with t-k < 0 are left 0."""
    T = len(u)
    y = np.ones(T)
    kmax = max(k for k, _ in config)
    for k, n in config:
        shifted = np.zeros(T)
        shifted[k:] = u[: T - k]
        y = y * norm_legendre(n, shifted)
    y[:kmax] = 0.0
    return y


def information_processing_capacity(
    X: np.ndarray,
    u: np.ndarray,
    split: Split,
    max_degree: int = 4,
    max_delay: int = 12,
    max_vars: int = 2,
    alphas=(1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0),
    threshold: float | None = None,
    n_surrogate: int = 64,
    surrogate_quantile: float = 0.999,
    rng_seed: int = 0,
) -> dict:
    """SUPERSEDED — α-railed + Legendre (continuous-u) basis. The canonical estimator
    is ``llm_life.capacity.information_processing_capacity`` (wide-α + encoded-symbol
    Gram-Schmidt basis + degree-stratified floor); see FINDINGS §25. Provenance only.

    Dambre IPC with a surrogate-calibrated significance threshold.

    For every config (up to the degree/delay/vars caps) compute the linear
    readout test R^2; keep only capacities above ``threshold`` and sum. The
    threshold defaults to a high quantile of a SHUFFLED-target null
    distribution -- the finite-sample R^2 bias floor -- so spurious capacity is
    not counted (Dambre's false-positive control). Returns per-degree totals and
    the full per-config table.
    """
    # bias floor from shuffled targets (destroys any real input relation, leaves
    # only the finite-sample fitting bias for a unit-variance target)
    if threshold is None:
        rng = np.random.default_rng(rng_seed)
        null_caps = []
        for _ in range(n_surrogate):
            yk = rng.standard_normal(X.shape[0])
            null_caps.append(_target_capacity(X, yk, split, alphas))
        threshold = float(np.quantile(null_caps, surrogate_quantile))

    per_degree = {}
    rows = []
    for total, config in enumerate_ipc_configs(max_degree, max_delay, max_vars):
        y = ipc_target(config, u)
        c = _target_capacity(X, y, split, alphas)
        c_thr = c if c > threshold else 0.0
        per_degree.setdefault(total, 0.0)
        per_degree[total] += c_thr
        rows.append({"degree": total, "config": config, "capacity": c,
                     "capacity_thr": c_thr})
    total_capacity = float(sum(per_degree.values()))
    return {
        "per_degree": per_degree,
        "total": total_capacity,
        "threshold": threshold,
        "readout_dim": int(X.shape[1]),
        "rows": rows,
        "caps": {"max_degree": max_degree, "max_delay": max_delay, "max_vars": max_vars},
    }


def _target_capacity(X: np.ndarray, y: np.ndarray, split: Split, alphas) -> float:
    """Validation-selected ridge readout test R^2 for a single target. Capacity
    is clipped at 0 (a negative R^2 means no capacity)."""
    tr = np.arange(split.train.start, split.train.stop)
    va = np.arange(split.val.start, split.val.stop)
    te = np.arange(split.test.start, split.test.stop)
    Xtr, ytr = X[tr], y[tr]
    best_a, best_val = alphas[0], -np.inf
    for a in alphas:
        w = ridge_fit(Xtr, ytr, a)
        r = r2_score(y[va], ridge_predict(X[va], w))
        if r > best_val:
            best_val, best_a = r, a
    w = ridge_fit(Xtr, ytr, best_a)
    return max(0.0, r2_score(y[te], ridge_predict(X[te], w)))
