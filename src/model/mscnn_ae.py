"""Stage 1: Multi-Scale CNN Autoencoder (MSCNN-AE).

Per-flow autoencoder operating on 1D feature vectors.
Uses 3 parallel Conv1D branches with different kernel sizes (1, 3, 5)
to capture multi-scale relationships between features.

Architecture:
  Encoder: 3x Conv1D branches -> Concat -> Conv1D(reduction) -> GAP1D -> Dense(latent)
  Decoder: Dense -> Reshape -> Conv1DTranspose layers -> output (linear activation)

Conv1D avoids artificial 2D reshaping and eliminates the need for bilinear resize.
"""

from __future__ import annotations

import logging

from tensorflow import keras
from tensorflow.keras import layers

logger = logging.getLogger(__name__)


def build_mscnn_ae(
    n_features: int,
    latent_dim: int,
    conv_filters: list[int] | None = None,
    conv_kernels: list[int] | None = None,
    reduction_filters: int = 128,
) -> tuple[keras.Model, keras.Model]:
    """Build the MSCNN Autoencoder with Conv1D.

    Args:
        n_features: number of input features
        latent_dim: bottleneck dimension
        conv_filters: filters per branch (default [64, 64, 64])
        conv_kernels: kernel sizes per branch (default [1, 3, 5])
        reduction_filters: filters for the reduction conv layer

    Returns:
        (full_model, encoder_model)
    """
    if conv_filters is None:
        conv_filters = [64, 64, 64]
    if conv_kernels is None:
        conv_kernels = [1, 3, 5]

    inp = layers.Input(shape=(n_features, 1), name="input")

    # --- Encoder: Multi-Scale Conv1D branches ---
    branches = []
    for i, (n_filt, k_size) in enumerate(zip(conv_filters, conv_kernels)):
        x = layers.Conv1D(
            n_filt, k_size, padding="same",
            activation="relu", name=f"enc_branch{i+1}_conv1",
        )(inp)
        x = layers.BatchNormalization(name=f"enc_branch{i+1}_bn1")(x)
        x = layers.Conv1D(
            n_filt, k_size, padding="same",
            activation="relu", name=f"enc_branch{i+1}_conv2",
        )(x)
        x = layers.BatchNormalization(name=f"enc_branch{i+1}_bn2")(x)
        branches.append(x)

    merged = layers.Concatenate(name="enc_merge")(branches)

    x = layers.Conv1D(
        reduction_filters, 3, padding="same",
        activation="relu", name="enc_reduce_conv",
    )(merged)
    x = layers.BatchNormalization(name="enc_reduce_bn")(x)

    x = layers.GlobalAveragePooling1D(name="enc_gap")(x)
    x = layers.Dropout(0.2, name="enc_dropout")(x)

    latent = layers.Dense(latent_dim, activation="linear", name="latent")(x)

    # --- Decoder ---
    dec_init_len = max(4, n_features // 4)
    dec_init_filters = reduction_filters // 2

    x = layers.Dense(
        dec_init_len * dec_init_filters,
        activation="relu", name="dec_dense",
    )(latent)
    x = layers.Reshape((dec_init_len, dec_init_filters), name="dec_reshape")(x)

    x = layers.Conv1DTranspose(
        dec_init_filters, 3, padding="same",
        activation="relu", name="dec_conv_t1",
    )(x)
    x = layers.BatchNormalization(name="dec_bn1")(x)

    x = layers.Conv1DTranspose(
        dec_init_filters // 2, 3, padding="same",
        activation="relu", name="dec_conv_t2",
    )(x)
    x = layers.BatchNormalization(name="dec_bn2")(x)

    # Dense layer to map to exact n_features length
    x = layers.Flatten(name="dec_flatten")(x)
    x = layers.Dense(n_features, activation="linear", name="dec_dense_out")(x)
    output = layers.Reshape((n_features, 1), name="output")(x)

    model = keras.Model(inp, output, name="MSCNN_AE")
    encoder = keras.Model(inp, latent, name="MSCNN_Encoder")

    _log_model_summary(model, latent_dim, n_features)
    return model, encoder


def _log_model_summary(model: keras.Model, latent_dim: int, n_features: int) -> None:
    total_params = model.count_params()
    compression_ratio = n_features / latent_dim if latent_dim > 0 else 0
    logger.info(
        "MSCNN-AE (Conv1D): input=(%d,1), latent=%d, "
        "compression=%.1fx, params=%d",
        n_features, latent_dim, compression_ratio, total_params,
    )
    if compression_ratio < 3:
        logger.warning(
            "Low compression ratio (%.1fx) — bottleneck may be too large.",
            compression_ratio,
        )
