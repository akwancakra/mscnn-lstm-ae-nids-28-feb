"""Preprocessing pipeline: clean, impute, feature-select, scale, reshape.

All transformations are fit ONLY on benign CIC-IDS-2017 training data.
No information from test/evaluation sets leaks into preprocessing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from src.data.loader import (
    detect_label_column,
    extract_session_metadata,
    list_csv_files,
    read_csv_chunk,
)

logger = logging.getLogger(__name__)


def replace_inf(df: pd.DataFrame) -> pd.DataFrame:
    """Replace +/-inf with NaN for subsequent imputation."""
    return df.replace([np.inf, -np.inf], np.nan)


def fit_median_imputer(X: pd.DataFrame) -> pd.Series:
    """Compute per-column medians on benign training data for imputation."""
    medians = X.median()
    n_missing = X.isna().any().sum()
    logger.info("Imputer fit: %d/%d features have NaN values", n_missing, len(X.columns))
    return medians


def apply_imputer(X: pd.DataFrame, medians: pd.Series) -> pd.DataFrame:
    """Fill NaN using pre-fitted medians. Columns not in medians get 0."""
    for col in X.columns:
        if col in medians.index:
            X[col] = X[col].fillna(medians[col])
        else:
            X[col] = X[col].fillna(0.0)
    return X


def select_features_nzv(
    X: pd.DataFrame, threshold: float = 1e-5
) -> list[str]:
    """Identify and remove features with near-zero variance."""
    variances = X.var()
    nzv_mask = variances < threshold
    dropped = list(variances[nzv_mask].index)
    if dropped:
        logger.info("NZV filter: dropping %d features (var < %g)", len(dropped), threshold)
        logger.debug("NZV dropped: %s", dropped)
    return [c for c in X.columns if c not in dropped]


def select_features_corr(
    X: pd.DataFrame, threshold: float = 0.98
) -> list[str]:
    """Remove one of each pair of highly correlated features."""
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = set()
    for col in upper.columns:
        highly_corr = upper.index[upper[col] > threshold].tolist()
        if highly_corr:
            to_drop.add(col)
    if to_drop:
        logger.info("Correlation filter: dropping %d features (corr > %g)", len(to_drop), threshold)
        logger.debug("Correlation dropped: %s", sorted(to_drop))
    return [c for c in X.columns if c not in to_drop]


def check_label_leakage(
    X_benign: pd.DataFrame, feature_names: list[str]
) -> list[str]:
    """Detect features that might leak label info.

    A feature leaks if it has zero variance on benign data (always the same
    value), which could mean it's an implicit binary indicator of attack type.
    """
    suspicious = []
    for col in feature_names:
        if col not in X_benign.columns:
            continue
        vals = X_benign[col]
        unique_count = vals.nunique()
        if unique_count <= 1:
            suspicious.append(col)
    if suspicious:
        logger.warning(
            "Potential label leakage — %d features have <=1 unique value "
            "on benign data (may be constant-zero indicators): %s",
            len(suspicious), suspicious,
        )
    return suspicious


def fit_scaler(X: pd.DataFrame) -> RobustScaler:
    """Fit RobustScaler on benign training data."""
    scaler = RobustScaler()
    scaler.fit(X.values)
    logger.info("RobustScaler fit on %d samples, %d features", len(X), X.shape[1])
    return scaler


def apply_scaler(
    X: np.ndarray, scaler: RobustScaler, clip_value: float = 5.0
) -> np.ndarray:
    """Transform and clip scaled values to [-clip, clip]."""
    X_scaled = scaler.transform(X)
    X_clipped = np.clip(X_scaled, -clip_value, clip_value)
    return X_clipped


def reshape_for_conv1d(X: np.ndarray) -> np.ndarray:
    """Reshape (N, n_features) -> (N, n_features, 1) for Conv1D input."""
    return X.reshape(-1, X.shape[1], 1)


def compute_latent_dim(n_features: int) -> int:
    """Compute latent dimension: max(12, n_features // 3)."""
    return max(12, n_features // 3)


class PreprocessingPipeline:
    """Encapsulates all fitted preprocessing state for reproducible transforms."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.pp_cfg = cfg.get("preprocessing", {})
        self.medians: Optional[pd.Series] = None
        self.scaler: Optional[RobustScaler] = None
        self.feature_names: list[str] = []
        self.n_features_original: int = 0
        self.n_features_final: int = 0
        self.latent_dim: int = 0
        self._is_fitted = False

    def fit(
        self,
        X_benign: pd.DataFrame,
        all_shared_features: list[str],
    ) -> "PreprocessingPipeline":
        """Fit all preprocessing on benign CIC-2017 train split.

        Steps: replace inf -> impute -> NZV filter -> corr filter ->
        leakage check -> scale.
        """
        filter_cfg = self.pp_cfg.get("feature_filter", {})
        nzv_thresh = filter_cfg.get("nzv_threshold", 1e-5)
        corr_thresh = filter_cfg.get("corr_threshold", 0.98)
        clip_val = self.pp_cfg.get("post_scale_clip", 5.0)

        available = [c for c in all_shared_features if c in X_benign.columns]
        X = X_benign[available].copy()
        self.n_features_original = len(available)

        X = replace_inf(X)
        X = X.apply(pd.to_numeric, errors="coerce")

        self.medians = fit_median_imputer(X)
        X = apply_imputer(X, self.medians)

        kept = select_features_nzv(X, threshold=nzv_thresh)
        X = X[kept]

        kept = select_features_corr(X, threshold=corr_thresh)
        X = X[kept]

        leaky = check_label_leakage(X, list(X.columns))
        if leaky:
            X = X.drop(columns=leaky)
            logger.info("Dropped %d leaky features", len(leaky))

        self.feature_names = list(X.columns)
        self.n_features_final = len(self.feature_names)
        logger.info(
            "Features: %d original -> %d after filtering",
            self.n_features_original, self.n_features_final,
        )

        self.scaler = fit_scaler(X)
        self.latent_dim = compute_latent_dim(self.n_features_final)

        logger.info(
            "Features: %d final, latent_dim=%d (compression=%.1fx)",
            self.n_features_final, self.latent_dim,
            self.n_features_final / self.latent_dim,
        )

        self._is_fitted = True
        return self

    def transform(
        self, df: pd.DataFrame, reshape_1d: bool = True
    ) -> np.ndarray:
        """Apply fitted preprocessing to a DataFrame.

        Returns numpy array of shape (N, n_features, 1) if reshape_1d else (N, n_features).
        """
        assert self._is_fitted, "Pipeline must be fit before transform"
        clip_val = self.pp_cfg.get("post_scale_clip", 5.0)

        available = [c for c in self.feature_names if c in df.columns]
        missing = set(self.feature_names) - set(available)
        X = df[available].copy()

        if missing:
            for col in missing:
                X[col] = 0.0
        X = X[self.feature_names]

        X = replace_inf(X)
        X = X.apply(pd.to_numeric, errors="coerce")
        X = apply_imputer(X, self.medians)

        X_scaled = apply_scaler(X.values, self.scaler, clip_value=clip_val)

        if reshape_1d:
            return reshape_for_conv1d(X_scaled)
        return X_scaled

    def save(self, path: str | Path) -> None:
        """Persist fitted pipeline to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "medians": self.medians,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "n_features_original": self.n_features_original,
            "n_features_final": self.n_features_final,
            "latent_dim": self.latent_dim,
            "pp_cfg": self.pp_cfg,
        }
        joblib.dump(state, path)
        logger.info("Pipeline saved to %s", path)

    def load(self, path: str | Path) -> "PreprocessingPipeline":
        """Restore fitted pipeline from disk."""
        state = joblib.load(path)
        self.medians = state["medians"]
        self.scaler = state["scaler"]
        self.feature_names = state["feature_names"]
        self.n_features_original = state["n_features_original"]
        self.n_features_final = state["n_features_final"]
        self.latent_dim = state.get("latent_dim", compute_latent_dim(self.n_features_final))
        self.pp_cfg = state.get("pp_cfg", self.pp_cfg)
        self._is_fitted = True
        logger.info("Pipeline loaded from %s", path)
        return self


def load_and_prepare_benign_train(
    cic_files: list[Path],
    shared_features: list[str],
    label_col: str,
    benign_label: str,
    session_cfg: dict,
    chunksize: int = 50000,
    val_size: float = 0.2,
    split_by_file: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load CIC-IDS-2017 benign data, split into train/val.

    Returns: (X_train, X_val, meta_train, meta_val)
    - X_train/X_val: raw numeric DataFrames (pre-scaling)
    - meta_train/meta_val: session metadata DataFrames
    """
    if split_by_file and len(cic_files) > 1:
        n_val = max(1, int(len(cic_files) * val_size))
        val_files = cic_files[-n_val:]
        train_files = cic_files[:-n_val]
        logger.info("File-level split: %d train files, %d val files", len(train_files), len(val_files))
    else:
        train_files = cic_files
        val_files = []

    def _load_benign(files: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
        dfs = []
        metas = []
        needed_cols = list(set(
            shared_features + [label_col]
            + _flatten_aliases(session_cfg)
        ))
        for f in files:
            for chunk in read_csv_chunk(f, chunksize=chunksize):
                if label_col in chunk.columns:
                    benign_mask = chunk[label_col].astype(str).str.strip().str.upper() == benign_label.upper()
                    chunk = chunk[benign_mask]
                if len(chunk) == 0:
                    continue
                meta = extract_session_metadata(chunk, session_cfg, dataset_name="CIC")
                avail_feats = [c for c in shared_features if c in chunk.columns]
                dfs.append(chunk[avail_feats])
                metas.append(meta)
        if not dfs:
            return pd.DataFrame(), pd.DataFrame()
        return pd.concat(dfs, ignore_index=True), pd.concat(metas, ignore_index=True)

    X_train, meta_train = _load_benign(train_files)
    logger.info("Benign train: %d samples", len(X_train))

    if val_files:
        X_val, meta_val = _load_benign(val_files)
    else:
        n = len(X_train)
        split_idx = int(n * (1 - val_size))
        idx = np.random.permutation(n)
        train_idx, val_idx = idx[:split_idx], idx[split_idx:]
        X_val = X_train.iloc[val_idx].reset_index(drop=True)
        meta_val = meta_train.iloc[val_idx].reset_index(drop=True)
        X_train = X_train.iloc[train_idx].reset_index(drop=True)
        meta_train = meta_train.iloc[train_idx].reset_index(drop=True)

    logger.info("Benign val: %d samples", len(X_val))
    return X_train, X_val, meta_train, meta_val


def load_all_labeled(
    csv_files: list[Path],
    features: list[str],
    label_col: str,
    session_cfg: dict,
    column_mapper: Optional[dict[str, str]] = None,
    chunksize: int = 50000,
    dataset_name: str = "dataset",
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load ALL data (benign + attack) from files, returning features, labels, metadata."""
    dfs = []
    labels_list = []
    metas = []
    for f in csv_files:
        for chunk in read_csv_chunk(f, chunksize=chunksize, column_mapper=column_mapper):
            if label_col in chunk.columns:
                lab = chunk[label_col].astype(str).str.strip()
                labels_list.append(lab)
            else:
                labels_list.append(pd.Series(["UNKNOWN"] * len(chunk)))
            meta = extract_session_metadata(chunk, session_cfg, dataset_name=dataset_name)
            avail = [c for c in features if c in chunk.columns]
            dfs.append(chunk[avail])
            metas.append(meta)

    X = pd.concat(dfs, ignore_index=True)
    y = pd.concat(labels_list, ignore_index=True)
    meta = pd.concat(metas, ignore_index=True)
    logger.info("Loaded %d samples (%d features) from %d files", len(X), X.shape[1], len(csv_files))
    return X, y, meta


def _flatten_aliases(session_cfg: dict) -> list[str]:
    """Extract all alias names from session config."""
    names = []
    for key in ["src_ip", "dst_ip", "protocol", "timestamp"]:
        aliases = session_cfg.get(key, [])
        if isinstance(aliases, list):
            names.extend(aliases)
        else:
            names.append(str(aliases))
    return names
