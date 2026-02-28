"""Threshold determination and combined anomaly scoring.

All thresholds computed ONLY from benign validation reconstruction errors.
No test data is used in any threshold computation.

Normalizer is fitted on benign validation errors so test data is normalized
with the same stats (no data leakage from test min/max).

Four threshold strategies:
  1. Z-Score: mean + k*std
  2. Percentile: p-th percentile
  3. Gaussian Fit: fit Gaussian, use mean + k*std of fitted distribution
  4. IQR: Q3 + k*IQR
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


@dataclass
class ErrorNormalizer:
    """Fitted normalizer: stores mean/std from benign validation errors."""
    mean: float = 0.0
    std: float = 1.0

    def fit(self, errors: np.ndarray) -> "ErrorNormalizer":
        self.mean = float(np.mean(errors))
        self.std = float(np.std(errors))
        if self.std < 1e-12:
            self.std = 1.0
        return self

    def transform(self, errors: np.ndarray) -> np.ndarray:
        """Z-score normalize using fitted mean/std, then clip to [0, ~]."""
        z = (errors - self.mean) / self.std
        return np.clip(z, 0, None)

    def to_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_dict(cls, d: dict) -> "ErrorNormalizer":
        return cls(mean=d["mean"], std=d["std"])


def fit_normalizers(
    errors_s1: np.ndarray,
    errors_s2: np.ndarray | None,
) -> tuple[ErrorNormalizer, ErrorNormalizer | None]:
    """Fit normalizers on benign validation errors."""
    norm1 = ErrorNormalizer().fit(errors_s1)
    logger.info("S1 normalizer: mean=%.6f, std=%.6f", norm1.mean, norm1.std)
    norm2 = None
    if errors_s2 is not None and len(errors_s2) > 0:
        norm2 = ErrorNormalizer().fit(errors_s2)
        logger.info("S2 normalizer: mean=%.6f, std=%.6f", norm2.mean, norm2.std)
    return norm1, norm2


def combine_scores(
    errors_stage1: np.ndarray,
    errors_stage2: np.ndarray | None,
    norm1: ErrorNormalizer,
    norm2: ErrorNormalizer | None,
    alpha: float = 0.5,
) -> np.ndarray:
    """Combine Stage 1 and Stage 2 errors into a single anomaly score.

    Uses fitted normalizers (z-score based) so scores are comparable
    across train/val/test without data leakage.
    """
    e1_norm = norm1.transform(errors_stage1)

    if errors_stage2 is None or len(errors_stage2) == 0 or norm2 is None:
        logger.info("Stage 2 errors unavailable — using Stage 1 only (alpha=1.0)")
        return e1_norm

    e2_norm = norm2.transform(errors_stage2)

    if e2_norm.std() < 1e-10:
        logger.warning(
            "Stage 2 errors degenerate (std=%.2e), using alpha=%.1f for Stage 1",
            e2_norm.std(), 0.7,
        )
        alpha = 0.7

    combined = alpha * e1_norm + (1 - alpha) * e2_norm
    logger.info(
        "Combined scores: alpha=%.2f, mean=%.4f, std=%.4f",
        alpha, combined.mean(), combined.std(),
    )
    return combined


# ============================================================
#  Threshold strategies (all fit on benign validation errors)
# ============================================================

def threshold_zscore(
    benign_errors: np.ndarray, k_values: list[float]
) -> dict[str, float]:
    """Z-Score thresholds: mean + k*std."""
    mu = benign_errors.mean()
    sigma = benign_errors.std()
    results = {}
    for k in k_values:
        t = mu + k * sigma
        fpr = (benign_errors > t).mean()
        results[f"zscore_k{k}"] = {
            "threshold": float(t),
            "fpr_on_benign_val": float(fpr),
            "k": k,
        }
    return results


def threshold_percentile(
    benign_errors: np.ndarray, percentiles: list[float]
) -> dict[str, float]:
    """Percentile-based thresholds."""
    results = {}
    for p in percentiles:
        t = np.percentile(benign_errors, p)
        fpr = (benign_errors > t).mean()
        results[f"percentile_p{p}"] = {
            "threshold": float(t),
            "fpr_on_benign_val": float(fpr),
            "percentile": p,
        }
    return results


def threshold_gaussian_fit(
    benign_errors: np.ndarray, k_values: list[float]
) -> dict[str, float]:
    """Gaussian fit thresholds: fit N(mu, sigma) then use mu + k*sigma."""
    mu, sigma = sp_stats.norm.fit(benign_errors)
    results = {}
    for k in k_values:
        t = mu + k * sigma
        fpr = (benign_errors > t).mean()
        results[f"gaussian_k{k}"] = {
            "threshold": float(t),
            "fpr_on_benign_val": float(fpr),
            "k": k,
            "fitted_mu": float(mu),
            "fitted_sigma": float(sigma),
        }
    return results


def threshold_iqr(
    benign_errors: np.ndarray, k_values: list[float]
) -> dict[str, float]:
    """IQR-based thresholds: Q3 + k*IQR."""
    q1 = np.percentile(benign_errors, 25)
    q3 = np.percentile(benign_errors, 75)
    iqr = q3 - q1
    results = {}
    for k in k_values:
        t = q3 + k * iqr
        fpr = (benign_errors > t).mean()
        results[f"iqr_k{k}"] = {
            "threshold": float(t),
            "fpr_on_benign_val": float(fpr),
            "k": k,
            "q1": float(q1),
            "q3": float(q3),
            "iqr": float(iqr),
        }
    return results


def compute_all_thresholds(
    benign_errors: np.ndarray,
    cfg: dict,
) -> dict:
    """Compute all threshold strategies and select the best one.

    Selection criterion: lowest threshold that achieves FPR < target on benign val.
    """
    t_cfg = cfg.get("threshold", {})
    target_fpr = t_cfg.get("target_fpr", 0.10)

    all_results = {}

    all_results.update(
        threshold_zscore(benign_errors, t_cfg.get("zscore_k", [1.5, 2.0, 2.5, 3.0]))
    )
    all_results.update(
        threshold_percentile(benign_errors, t_cfg.get("percentiles", [90, 93, 95, 97]))
    )
    all_results.update(
        threshold_gaussian_fit(benign_errors, t_cfg.get("zscore_k", [2.0, 2.5, 3.0]))
    )
    all_results.update(
        threshold_iqr(benign_errors, t_cfg.get("iqr_k", [1.0, 1.5, 2.0, 3.0]))
    )

    # Select best: lowest threshold with FPR <= target
    valid = {
        k: v for k, v in all_results.items()
        if v["fpr_on_benign_val"] <= target_fpr
    }

    if valid:
        best_key = min(valid, key=lambda k: valid[k]["threshold"])
        best_info = valid[best_key]
    else:
        best_key = min(
            all_results,
            key=lambda k: abs(all_results[k]["fpr_on_benign_val"] - target_fpr),
        )
        best_info = all_results[best_key]
        logger.warning(
            "No threshold achieves FPR <= %.3f on benign val. "
            "Selected '%s' with FPR=%.4f (closest).",
            target_fpr, best_key, best_info["fpr_on_benign_val"],
        )

    logger.info(
        "Selected threshold: %s = %.6f (FPR=%.4f on benign val)",
        best_key, best_info["threshold"], best_info["fpr_on_benign_val"],
    )

    return {
        "all_thresholds": all_results,
        "selected": best_key,
        "selected_threshold": best_info["threshold"],
        "selected_fpr": best_info["fpr_on_benign_val"],
        "target_fpr": target_fpr,
    }
