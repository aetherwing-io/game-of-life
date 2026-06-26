"""Space-time visualization.

The money plot: position on x, generation (time) on y (top = gen 0), colour =
token. To make structure *visible* we don't hash token ids randomly -- we
project the model's input-embedding matrix to its top-3 principal components and
map each token id to an RGB colour. Semantically/embedding-similar tokens get
similar colours, so coherent regions, oscillations, and travelling fronts show
up as contiguous colour structures and diagonal streaks rather than noise.
"""

from __future__ import annotations

import numpy as np


def embedding_rgb_table(embedding_matrix: np.ndarray) -> np.ndarray:
    """(V, d) embeddings -> (V, 3) RGB in [0,1] via top-3 PCA components."""
    # Some checkpoints have non-finite or fp16-overflowing embedding rows (e.g.
    # untrained/padding rows, or fp16 weights); scrub them so the SVD/matmul
    # don't propagate NaN/Inf into the colour table (which blanks the image).
    X = np.nan_to_num(embedding_matrix.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    X = X - X.mean(axis=0, keepdims=True)
    # top-3 right singular vectors
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    proj = np.nan_to_num(X @ Vt[:3].T)  # (V, 3)
    lo = proj.min(axis=0, keepdims=True)
    hi = proj.max(axis=0, keepdims=True)
    rng = np.where(hi - lo == 0, 1.0, hi - lo)
    return (proj - lo) / rng


def spacetime_image(states: np.ndarray, rgb_table: np.ndarray) -> np.ndarray:
    """(T+1, L) ids + (V,3) table -> (T+1, L, 3) image array in [0,1]."""
    return rgb_table[states]


def save_spacetime(states: np.ndarray, rgb_table: np.ndarray, path: str, title: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = spacetime_image(states, rgb_table)
    T1, L = states.shape
    fig_h = max(3.0, min(12.0, T1 / 40.0))
    fig_w = max(3.0, min(12.0, L / 20.0))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(img, interpolation="nearest", aspect="auto")
    ax.set_xlabel("position")
    ax.set_ylabel("generation")
    if title:
        ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_animation(
    states: np.ndarray,
    rgb_table: np.ndarray,
    path: str,
    title: str = "",
    window: int = 80,
    fps: int = 20,
) -> None:
    """Animated GIF: the space-time diagram filling in generation by generation,
    scrolling once it exceeds ``window`` rows. You watch fronts propagate in
    real time instead of seeing the whole history at once.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    img = spacetime_image(states, rgb_table)  # (T+1, L, 3)
    T1, L = states.shape
    fig_w = max(3.0, min(12.0, L / 16.0))
    fig, ax = plt.subplots(figsize=(fig_w, 4.5))
    ax.set_xlabel("position")
    ax.set_ylabel("generation")
    if title:
        ax.set_title(title, fontsize=9)
    blank = np.ones((window, L, 3))
    im = ax.imshow(blank, interpolation="nearest", aspect="auto")

    def update(frame):
        lo = max(0, frame - window + 1)
        chunk = img[lo:frame + 1]
        if chunk.shape[0] < window:
            pad = np.ones((window - chunk.shape[0], L, 3))
            chunk = np.vstack([chunk, pad])
        im.set_data(chunk)
        ax.set_ylabel(f"generation {lo}–{frame}")
        return (im,)

    anim = FuncAnimation(fig, update, frames=T1, interval=1000 // fps, blit=False)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"[wrote] {path}")
