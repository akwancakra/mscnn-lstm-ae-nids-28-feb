"""Domain shift analysis between CIC-IDS-2017 and CSE-CIC-IDS-2018.

Uses Kolmogorov-Smirnov test to quantify per-feature distributional
differences. Fit/sample ONLY from CIC benign training data.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def ks_test_per_feature(
    X_source: np.ndarray,
    X_target: np.ndarray,
    feature_names: list[str],
    max_samples: int = 10000,
) -> pd.DataFrame:
    """Compute KS-statistic per feature between source and target distributions.

    Both arrays should be already scaled using the same scaler (fit on source).
    """
    rng = np.random.RandomState(42)

    n_src = min(len(X_source), max_samples)
    n_tgt = min(len(X_target), max_samples)
    idx_src = rng.choice(len(X_source), n_src, replace=False)
    idx_tgt = rng.choice(len(X_target), n_tgt, replace=False)

    results = []
    for i, feat in enumerate(feature_names):
        s = X_source[idx_src, i]
        t = X_target[idx_tgt, i]
        ks_stat, p_val = stats.ks_2samp(s, t)
        results.append({
            "feature": feat,
            "ks_statistic": ks_stat,
            "p_value": p_val,
            "mean_source": np.mean(s),
            "mean_target": np.mean(t),
            "std_source": np.std(s),
            "std_target": np.std(t),
        })

    df = pd.DataFrame(results).sort_values("ks_statistic", ascending=False)
    df = df.reset_index(drop=True)

    n_significant = (df["p_value"] < 0.01).sum()
    high_shift = (df["ks_statistic"] > 0.3).sum()
    logger.info(
        "Domain shift: %d/%d features significantly different (p<0.01), "
        "%d with KS>0.3",
        n_significant, len(feature_names), high_shift,
    )
    return df


def plot_domain_shift(
    ks_df: pd.DataFrame,
    top_n: int = 20,
    save_path: str | None = None,
) -> None:
    """Bar chart of top-N features by KS statistic."""
    top = ks_df.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#e74c3c" if ks > 0.3 else "#f39c12" if ks > 0.15 else "#2ecc71"
              for ks in top["ks_statistic"]]
    ax.barh(range(len(top)), top["ks_statistic"].values, color=colors)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"].values, fontsize=8)
    ax.set_xlabel("KS Statistic")
    ax.set_title("Domain Shift: CIC-2017 vs CSE-2018 (Top Features)")
    ax.axvline(x=0.3, color="red", linestyle="--", alpha=0.5, label="High shift (>0.3)")
    ax.axvline(x=0.15, color="orange", linestyle="--", alpha=0.5, label="Moderate shift (>0.15)")
    ax.legend(fontsize=8)
    ax.invert_yaxis()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Domain shift plot saved to %s", save_path)
    plt.close(fig)


def summarize_shift(ks_df: pd.DataFrame) -> dict:
    """Return summary statistics of domain shift analysis."""
    return {
        "n_features": len(ks_df),
        "n_significant_p001": int((ks_df["p_value"] < 0.01).sum()),
        "n_high_shift_ks03": int((ks_df["ks_statistic"] > 0.3).sum()),
        "n_moderate_shift_ks015": int(
            ((ks_df["ks_statistic"] > 0.15) & (ks_df["ks_statistic"] <= 0.3)).sum()
        ),
        "mean_ks": float(ks_df["ks_statistic"].mean()),
        "median_ks": float(ks_df["ks_statistic"].median()),
        "max_ks": float(ks_df["ks_statistic"].max()),
        "top5_shifted": ks_df.head(5)[["feature", "ks_statistic"]].to_dict("records"),
    }
