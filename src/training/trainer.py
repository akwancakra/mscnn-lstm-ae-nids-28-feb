"""Training logic for Stage 1 (MSCNN-AE) and Stage 2 (BiLSTM-AE).

Both stages trained ONLY on benign CIC-IDS-2017 data.
Stage 2 operates on latent vectors extracted from a trained Stage 1 encoder.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.model.mscnn_ae import build_mscnn_ae
from src.model.lstm_ae import build_bilstm_ae, compute_temporal_latent_dim

logger = logging.getLogger(__name__)


def get_callbacks(
    checkpoint_path: str,
    patience_es: int = 10,
    patience_lr: int = 5,
    lr_factor: float = 0.5,
    min_lr: float = 1e-6,
) -> list[keras.callbacks.Callback]:
    """Standard callbacks: EarlyStopping, ReduceLR, ModelCheckpoint."""
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience_es,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=lr_factor,
            patience=patience_lr,
            min_lr=min_lr,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
    ]


def plot_training_curves(
    history: keras.callbacks.History,
    title: str = "Training",
    save_path: str | None = None,
) -> None:
    """Plot training and validation loss curves."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history.history["loss"], label="Train Loss")
    ax.plot(history.history["val_loss"], label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(f"{title} — Training Curves")
    ax.legend()
    ax.set_yscale("log")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Training curves saved to %s", save_path)
    plt.close(fig)


# =========================================================
#  Stage 1: MSCNN-AE (Conv1D)
# =========================================================

def train_stage1(
    X_train: np.ndarray,
    X_val: np.ndarray,
    cfg: dict,
    models_dir: str,
    results_dir: str,
) -> tuple[keras.Model, keras.Model, keras.callbacks.History]:
    """Train the MSCNN Autoencoder (Stage 1).

    Args:
        X_train: (N_train, n_features, 1) benign training data
        X_val: (N_val, n_features, 1) benign validation data
        cfg: full config dict
        models_dir: directory to save model checkpoints
        results_dir: directory to save plots

    Returns:
        (full_model, encoder, history)
    """
    s1 = cfg.get("stage1", {})
    n_features = X_train.shape[1]

    latent_dim = s1.get("latent_dim", "auto")
    if latent_dim == "auto":
        from src.data.preprocessing import compute_latent_dim
        latent_dim = compute_latent_dim(n_features)

    model, encoder = build_mscnn_ae(
        n_features=n_features,
        latent_dim=latent_dim,
        conv_filters=s1.get("conv_filters", [64, 64, 64]),
        conv_kernels=s1.get("conv_kernels", [1, 3, 5]),
        reduction_filters=s1.get("reduction_filters", 128),
    )

    optimizer = keras.optimizers.Adam(
        learning_rate=s1.get("learning_rate", 1e-3),
        clipnorm=s1.get("clipnorm", 1.0),
    )
    model.compile(optimizer=optimizer, loss="mse")

    ckpt_path = str(Path(models_dir) / "stage1_mscnn_ae.keras")
    callbacks = get_callbacks(
        checkpoint_path=ckpt_path,
        patience_es=s1.get("early_stopping_patience", 10),
        patience_lr=s1.get("reduce_lr_patience", 5),
        lr_factor=s1.get("reduce_lr_factor", 0.5),
        min_lr=s1.get("min_lr", 1e-6),
    )

    logger.info(
        "Stage 1 training: %d train, %d val, batch=%d, max_epochs=%d",
        len(X_train), len(X_val),
        s1.get("batch_size", 256), s1.get("epochs", 100),
    )

    history = model.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        batch_size=s1.get("batch_size", 256),
        epochs=s1.get("epochs", 100),
        callbacks=callbacks,
        verbose=1,
    )

    plot_training_curves(
        history, title="Stage 1 — MSCNN-AE",
        save_path=str(Path(results_dir) / "stage1_training_curves.png"),
    )

    return model, encoder, history


def extract_latent_vectors(
    encoder: keras.Model,
    X: np.ndarray,
    batch_size: int = 512,
) -> np.ndarray:
    """Extract latent vectors from Stage 1 encoder."""
    latent = encoder.predict(X, batch_size=batch_size, verbose=0)
    logger.info("Extracted latent vectors: shape=%s", latent.shape)
    return latent


def compute_stage1_errors(
    model: keras.Model,
    X: np.ndarray,
    batch_size: int = 512,
) -> np.ndarray:
    """Compute per-sample MSE reconstruction error for Stage 1.

    Input shape: (N, n_features, 1) -> mean over axes (1, 2)
    """
    X_recon = model.predict(X, batch_size=batch_size, verbose=0)
    errors = np.mean((X - X_recon) ** 2, axis=tuple(range(1, X.ndim)))
    logger.info(
        "Stage 1 errors: mean=%.6f, std=%.6f, max=%.6f",
        errors.mean(), errors.std(), errors.max(),
    )
    return errors


# =========================================================
#  Stage 2: BiLSTM-AE
# =========================================================

def train_stage2(
    windows_train: np.ndarray,
    windows_val: np.ndarray,
    latent_dim: int,
    window_size: int,
    cfg: dict,
    models_dir: str,
    results_dir: str,
) -> tuple[keras.Model, keras.Model, keras.callbacks.History]:
    """Train the BiLSTM/Dense Autoencoder (Stage 2).

    Args:
        windows_train: (N_train, W, latent_dim)
        windows_val: (N_val, W, latent_dim)
        latent_dim: dimension of latent vectors from Stage 1
        window_size: W
        cfg: full config dict
        models_dir: directory to save checkpoints
        results_dir: directory to save plots

    Returns:
        (full_model, encoder, history)
    """
    s2 = cfg.get("stage2", {})

    temporal_latent_dim = s2.get("temporal_latent_dim", "auto")
    if temporal_latent_dim == "auto":
        temporal_latent_dim = compute_temporal_latent_dim(latent_dim)

    model, encoder = build_bilstm_ae(
        window_size=window_size,
        latent_dim=latent_dim,
        temporal_latent_dim=temporal_latent_dim,
        lstm_units=s2.get("lstm_units", 32),
        dropout=s2.get("dropout", 0.3),
    )

    optimizer = keras.optimizers.Adam(
        learning_rate=s2.get("learning_rate", 1e-3),
        clipnorm=s2.get("clipnorm", 1.0),
    )
    model.compile(optimizer=optimizer, loss="mse")

    ckpt_path = str(Path(models_dir) / "stage2_bilstm_ae.keras")
    callbacks = get_callbacks(
        checkpoint_path=ckpt_path,
        patience_es=s2.get("early_stopping_patience", 10),
        patience_lr=s2.get("reduce_lr_patience", 5),
        lr_factor=s2.get("reduce_lr_factor", 0.5),
        min_lr=s2.get("min_lr", 1e-6),
    )

    logger.info(
        "Stage 2 training: %d train, %d val, W=%d, batch=%d",
        len(windows_train), len(windows_val), window_size,
        s2.get("batch_size", 256),
    )

    history = model.fit(
        windows_train, windows_train,
        validation_data=(windows_val, windows_val),
        batch_size=s2.get("batch_size", 256),
        epochs=s2.get("epochs", 100),
        callbacks=callbacks,
        verbose=1,
    )

    stage_name = "BiLSTM-AE" if window_size > 1 else "Dense-AE"
    plot_training_curves(
        history, title=f"Stage 2 — {stage_name}",
        save_path=str(Path(results_dir) / "stage2_training_curves.png"),
    )

    return model, encoder, history


def compute_stage2_errors(
    model: keras.Model,
    windows: np.ndarray,
    batch_size: int = 512,
) -> np.ndarray:
    """Compute per-window MSE reconstruction error for Stage 2."""
    W_recon = model.predict(windows, batch_size=batch_size, verbose=0)
    errors = np.mean((windows - W_recon) ** 2, axis=tuple(range(1, windows.ndim)))
    logger.info(
        "Stage 2 errors: mean=%.6f, std=%.6f, max=%.6f",
        errors.mean(), errors.std(), errors.max(),
    )
    return errors
