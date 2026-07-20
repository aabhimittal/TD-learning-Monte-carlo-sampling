"""Small utilities for smoothing and visualising learning curves.

``moving_average`` and ``ascii_sparkline`` are pure NumPy and have no extra
dependencies. ``plot_learning_curves`` renders a proper figure and imports
matplotlib lazily, so the core library stays install-light -- you only need
matplotlib if you actually call it.
"""
from __future__ import annotations

import numpy as np

_BLOCKS = " ▁▂▃▄▅▆▇█"


def moving_average(values, window: int = 10) -> np.ndarray:
    """Trailing simple moving average.

    Returns an array the same length as ``values``; the first ``window-1``
    entries average over the (shorter) available prefix so the curve starts
    at episode 1 rather than ``window``.
    """
    values = np.asarray(values, dtype=np.float64)
    if window <= 1 or values.size == 0:
        return values
    window = min(window, values.size)
    cumsum = np.cumsum(np.insert(values, 0, 0.0))
    out = np.empty_like(values)
    for i in range(values.size):
        lo = max(0, i - window + 1)
        out[i] = (cumsum[i + 1] - cumsum[lo]) / (i + 1 - lo)
    return out


def ascii_sparkline(values, width: int = 50) -> str:
    """Render ``values`` as a one-line Unicode sparkline (no dependencies)."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return ""
    if values.size > width:  # downsample by averaging buckets
        idx = np.linspace(0, values.size, width + 1).astype(int)
        values = np.array([values[idx[i]:idx[i + 1]].mean() for i in range(width)])
    lo, hi = values.min(), values.max()
    if hi - lo < 1e-12:
        return _BLOCKS[1] * len(values)
    scaled = (values - lo) / (hi - lo) * (len(_BLOCKS) - 2) + 1
    return "".join(_BLOCKS[int(round(s))] for s in scaled)


def plot_learning_curves(curves: dict, window: int = 25, title: str = "Learning curves",
                         xlabel: str = "Episode", ylabel: str = "Return", path=None):
    """Plot one smoothed curve per entry in ``curves`` ({name: returns}).

    Saves to ``path`` if given (and returns it), otherwise shows the figure.
    Requires matplotlib (imported lazily).
    """
    try:
        import matplotlib
        if path is not None:
            matplotlib.use("Agg")  # headless-safe when only saving
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "plot_learning_curves requires matplotlib. Install it with "
            "`pip install matplotlib` (or `pip install -e '.[plot]'`)."
        ) from exc

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, returns in curves.items():
        smoothed = moving_average(returns, window)
        ax.plot(np.arange(1, len(smoothed) + 1), smoothed, label=name, linewidth=1.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if path is not None:
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path
    plt.show()
    return None
