"""Session-based windowing for Stage 2 (LSTM-AE).

Groups latent vectors by session_id, sorts within each session by timestamp,
and creates non-overlapping windows of size W. Analyzes session lengths
to decide between session-based windowing and per-flow fallback.
"""

from __future__ import annotations

import logging
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def analyze_session_lengths(
    meta: pd.DataFrame,
    window_cfg: dict | None = None,
) -> dict:
    """Analyze session length distribution for windowing decisions.

    Args:
        meta: DataFrame with 'session_id' and 'timestamp' columns
        window_cfg: optional config; if session_mode=='index_fallback' then no raise on collapse

    Returns:
        dict with statistics about session lengths
    """
    if "session_id" not in meta.columns or (len(meta) > 0 and meta["session_id"].iloc[0] == "none"):
        stats = {
            "has_sessions": False,
            "total_sessions": 0,
            "total_flows": len(meta),
            "median_length": 0,
            "mean_length": 0,
            "recommended_mode": "per_flow",
        }
        index_fallback = (window_cfg or {}).get("session_mode") == "index_fallback"
        if index_fallback:
            logger.warning(
                "No valid session IDs — session analysis skipped (session_mode=index_fallback)."
            )
            return stats
        raise RuntimeError(
            "SESSION COLLAPSE DETECTED: No valid sessions found. "
            "Check column name normalization for src_ip/dst_ip/protocol/timestamp. "
            "Inspect raw column names with: pd.read_csv(path, nrows=0).columns.tolist()"
        )

    session_lengths = meta.groupby("session_id").size()
    total_sessions = len(session_lengths)
    total_flows = len(meta)

    if total_sessions == 1:
        index_fallback = (window_cfg or {}).get("session_mode") == "index_fallback"
        if index_fallback:
            logger.warning(
                "All %d flows mapped to 1 session (session_mode=index_fallback).",
                total_flows,
            )
            return {
                "has_sessions": False,
                "total_sessions": 0,
                "total_flows": total_flows,
                "median_length": 0,
                "mean_length": 0,
                "recommended_mode": "per_flow",
            }
        raise RuntimeError(
            f"SESSION COLLAPSE DETECTED: All {total_flows} flows mapped to 1 session. "
            "Session key (src_ip, dst_ip, protocol) is constant — likely column detection failure."
        )

    stats = {
        "has_sessions": True,
        "total_sessions": total_sessions,
        "total_flows": total_flows,
        "median_length": float(session_lengths.median()),
        "mean_length": float(session_lengths.mean()),
        "min_length": int(session_lengths.min()),
        "max_length": int(session_lengths.max()),
        "p25_length": float(session_lengths.quantile(0.25)),
        "p75_length": float(session_lengths.quantile(0.75)),
        "p90_length": float(session_lengths.quantile(0.90)),
        "pct_sessions_ge3": float((session_lengths >= 3).mean() * 100),
        "pct_sessions_ge5": float((session_lengths >= 5).mean() * 100),
    }

    flows_per_session = total_flows / total_sessions
    if flows_per_session > 10000:
        logger.warning(
            "Suspiciously large sessions: %.0f flows/session avg. Possible partial collapse.",
            flows_per_session,
        )

    if stats["median_length"] >= 3:
        stats["recommended_mode"] = "session"
        stats["recommended_window"] = min(int(stats["median_length"]), 15)
    else:
        stats["recommended_mode"] = "per_flow"
        stats["recommended_window"] = 1

    logger.info(
        "Session analysis: %d sessions, median_len=%.1f, mean_len=%.1f, "
        "recommended=%s (W=%s)",
        stats["total_sessions"], stats["median_length"],
        stats["mean_length"], stats["recommended_mode"],
        stats.get("recommended_window", 1),
    )
    return stats


def plot_session_lengths(
    meta: pd.DataFrame,
    save_path: str | None = None,
) -> None:
    """Plot histogram of session lengths."""
    if "session_id" not in meta.columns or meta["session_id"].iloc[0] == "none":
        return

    lengths = meta.groupby("session_id").size()
    fig, ax = plt.subplots(figsize=(8, 4))
    lengths_clipped = lengths.clip(upper=50)
    ax.hist(lengths_clipped, bins=50, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Session Length (flows)")
    ax.set_ylabel("Count")
    ax.set_title("Session Length Distribution")
    ax.axvline(x=lengths.median(), color="red", linestyle="--",
               label=f"Median={lengths.median():.0f}")
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def create_windows_session_based(
    latent_vectors: np.ndarray,
    meta: pd.DataFrame,
    window_size: int = 5,
    labels: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Create non-overlapping windows from latent vectors grouped by session.

    Args:
        latent_vectors: (N, latent_dim) from Stage 1 encoder
        meta: DataFrame with session_id and timestamp
        window_size: W
        labels: optional (N,) binary labels for evaluation

    Returns:
        (windows, window_labels)
        - windows: (n_windows, W, latent_dim)
        - window_labels: (n_windows,) majority label per window, or None
    """
    assert len(latent_vectors) == len(meta), "Mismatch between vectors and metadata"

    sessions = meta["session_id"].values
    timestamps = meta["timestamp"].values
    unique_sessions = np.unique(sessions)

    all_windows = []
    all_labels = []

    for sid in unique_sessions:
        mask = sessions == sid
        idx = np.where(mask)[0]

        ts = timestamps[idx]
        sort_order = np.argsort(ts)
        idx_sorted = idx[sort_order]

        vecs = latent_vectors[idx_sorted]
        n = len(vecs)

        n_full_windows = n // window_size
        for w in range(n_full_windows):
            start = w * window_size
            end = start + window_size
            all_windows.append(vecs[start:end])

            if labels is not None:
                window_labs = labels[idx_sorted[start:end]]
                majority = 1 if np.sum(window_labs) > 0 else 0
                all_labels.append(majority)

    if not all_windows:
        logger.warning("No windows created — all sessions shorter than W=%d", window_size)
        return np.empty((0, window_size, latent_vectors.shape[1])), None

    windows = np.array(all_windows)
    window_labels = np.array(all_labels) if labels is not None else None

    logger.info(
        "Session windowing: %d windows (W=%d) from %d sessions, "
        "coverage=%.1f%% of %d flows",
        len(windows), window_size, len(unique_sessions),
        len(windows) * window_size / len(latent_vectors) * 100,
        len(latent_vectors),
    )
    return windows, window_labels


def create_windows_index_based(
    latent_vectors: np.ndarray,
    window_size: int = 5,
    labels: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Create non-overlapping windows by consecutive index (no session grouping).

    Fallback when session-based windowing is not feasible.
    """
    n = len(latent_vectors)
    n_windows = n // window_size

    windows = latent_vectors[:n_windows * window_size].reshape(
        n_windows, window_size, -1
    )

    window_labels = None
    if labels is not None:
        lab_trunc = labels[:n_windows * window_size].reshape(n_windows, window_size)
        window_labels = (lab_trunc.sum(axis=1) > 0).astype(int)

    logger.info(
        "Index windowing: %d windows (W=%d), coverage=%.1f%%",
        n_windows, window_size, n_windows * window_size / n * 100,
    )
    return windows, window_labels


def create_windows_per_flow(
    latent_vectors: np.ndarray,
    labels: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Wrap each flow's latent vector as (1, latent_dim) for Dense-AE."""
    windows = latent_vectors[:, np.newaxis, :]
    logger.info("Per-flow mode: %d samples, shape=%s", len(windows), windows.shape)
    return windows, labels


def create_windows(
    latent_vectors: np.ndarray,
    meta: pd.DataFrame,
    session_stats: dict,
    window_cfg: dict,
    labels: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, int]:
    """Main windowing dispatcher based on config and session analysis.

    Returns: (windows, window_labels, effective_window_size)
    """
    mode = window_cfg.get("mode", "auto")
    W = window_cfg.get("window_size", 5)
    min_sess = window_cfg.get("min_session_length", 3)
    fallback = window_cfg.get("fallback_mode", "per_flow")

    if mode == "auto":
        if session_stats.get("has_sessions", False) and session_stats.get("median_length", 0) >= min_sess:
            mode = "session"
            W = session_stats.get("recommended_window", W)
        else:
            mode = fallback
            logger.info(
                "Auto mode: median session length (%.1f) < %d, using fallback='%s'",
                session_stats.get("median_length", 0), min_sess, fallback,
            )

    if mode == "session":
        windows, wl = create_windows_session_based(
            latent_vectors, meta, window_size=W, labels=labels,
        )
        if len(windows) == 0:
            logger.warning("Session windowing produced 0 windows, falling back to per_flow")
            windows, wl = create_windows_per_flow(latent_vectors, labels)
            W = 1
        return windows, wl, W

    elif mode == "index":
        windows, wl = create_windows_index_based(
            latent_vectors, window_size=W, labels=labels,
        )
        return windows, wl, W

    else:
        windows, wl = create_windows_per_flow(latent_vectors, labels)
        return windows, wl, 1
