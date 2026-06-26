"""Order parameters and dynamical-systems diagnostics.

Everything here operates on a trajectory ``states`` of shape (T+1, L) of token
ids (numpy int array). The metrics are the ones that actually distinguish
Wolfram classes / locate an edge-of-chaos transition:

  * activity rho(t)          -- fraction of sites that changed since last gen
  * live_density(t)          -- fraction of sites not equal to the dead token
  * token_entropy(t)         -- Shannon entropy (bits) of the per-gen token hist
  * integrated_autocorr_time -- temporal correlation time of rho(t)  [tau_int]
  * spatial_corr_length      -- decay length of the change-indicator field
  * lyapunov_estimate        -- early-time slope of log Hamming(t) for a
                                coupled-noise damage-spreading pair

The damage / Lyapunov measurement is the sharpest edge detector: bounded
separation => ordered, exponential => chaotic, marginal/power-law => critical.
"""

from __future__ import annotations

import numpy as np


def activity(states: np.ndarray) -> np.ndarray:
    """rho(t) for t=1..T: fraction of sites that changed since the previous gen."""
    return (states[1:] != states[:-1]).mean(axis=1)


def live_density(states: np.ndarray, dead_token: int) -> np.ndarray:
    """Fraction of live (non-dead) sites per generation, length T+1."""
    return (states != dead_token).mean(axis=1)


def token_entropy(states: np.ndarray, base: float = 2.0) -> np.ndarray:
    """Per-generation Shannon entropy of the empirical token distribution."""
    T1, L = states.shape
    out = np.empty(T1)
    for t in range(T1):
        _, counts = np.unique(states[t], return_counts=True)
        p = counts / L
        out[t] = -(p * (np.log(p) / np.log(base))).sum()
    return out


def integrated_autocorr_time(series: np.ndarray, c: float = 5.0) -> float:
    """Integrated autocorrelation time tau_int of a 1D series.

    tau_int = 1 + 2 * sum_{k>=1} rho(k), truncated self-consistently at the
    first window M with M >= c * tau (the standard automatic windowing of
    Sokal). Diverges (relative to series length) near criticality -> critical
    slowing down.
    """
    x = np.asarray(series, dtype=float)
    x = x - x.mean()
    n = x.size
    if n < 4 or np.allclose(x, 0.0):
        return 0.0
    var = (x * x).mean()
    if var == 0.0:
        return 0.0
    # autocovariance via FFT
    f = np.fft.rfft(x, n=2 * n)
    acov = np.fft.irfft(f * np.conjugate(f))[:n] / (n * var)
    tau = 1.0
    for M in range(1, n):
        tau = 1.0 + 2.0 * acov[1:M + 1].sum()
        if M >= c * tau:
            break
    return float(max(tau, 0.0))


def spatial_corr_length(states: np.ndarray) -> float:
    """Decay length (in sites) of the time-averaged change-indicator field.

    We build a_i(t) = 1[site i changed at step t], compute its spatial
    autocorrelation averaged over time, and return the lag at which it first
    falls below 1/e. Larger => longer-range spatial order.
    """
    changed = (states[1:] != states[:-1]).astype(float)  # (T, L)
    T, L = changed.shape
    if T == 0:
        return 0.0
    a = changed - changed.mean(axis=1, keepdims=True)
    denom = (a * a).mean()
    if denom == 0.0:
        return 0.0
    corr = np.zeros(L)
    for r in range(L):
        if r == 0:
            corr[0] = 1.0
        else:
            corr[r] = (a[:, :L - r] * a[:, r:]).mean() / denom
    thr = 1.0 / np.e
    below = np.where(corr < thr)[0]
    return float(below[0]) if below.size else float(L)


def hamming_curve(states_a: np.ndarray, states_b: np.ndarray) -> np.ndarray:
    """Per-generation Hamming distance between two coupled replicas, length T+1."""
    return (states_a != states_b).sum(axis=1).astype(float)


def lyapunov_estimate(hamming: np.ndarray, short_frac: float = 0.1) -> dict:
    """Characterize a coupled-noise damage-spreading curve.

    These curves are often NON-monotonic: a perturbation amplifies at short
    times (local instability) and then, under shared noise, may heal back to
    zero (common-noise-induced synchronization). A single exponential slope
    misrepresents that, so we report the curve's actual shape:

      * short_time_rate -- mean log-growth per step from t=0 to the peak
                           (positive => initial amplification / local instability)
      * peak_hamming    -- maximum separation reached
      * time_to_peak    -- generation index of the peak
      * final_hamming   -- separation at the end
      * synchronized    -- True if the replicas re-converged (final ~ 0)
      * lyapunov        -- log-slope over the rising segment [0, peak] (the
                           short-time conditional exponent)
    """
    h = np.asarray(hamming, dtype=float)
    peak_idx = int(np.argmax(h))
    peak = float(h[peak_idx])
    final = float(h[-1])
    if peak_idx >= 1 and h[0] > 0 and peak > 0:
        lam = float((np.log(peak) - np.log(h[0])) / peak_idx)
    else:
        lam = 0.0
    n_short = max(1, int(h.size * short_frac))
    short_rate = lam if peak_idx <= n_short else float(
        (np.log(max(h[n_short], 1e-9)) - np.log(max(h[0], 1e-9))) / n_short
    )
    return {
        "lyapunov": lam,
        "short_time_rate": short_rate,
        "peak_hamming": peak,
        "time_to_peak": peak_idx,
        "final_hamming": final,
        "synchronized": bool(final <= max(1.0, 0.01 * h.max())),
    }
