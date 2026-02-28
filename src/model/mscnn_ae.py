"""Stage 1: Multi-Scale CNN Autoencoder (MSCNN-AE).

Per-flow autoencoder operating on 2D-reshaped feature vectors.
Uses 3 parallel Conv2D branches with different kernel sizes (1x1, 3x3, 5x5)
to capture multi-scale spatial relationships between features.

Architecture:
  Encoder: 3x Conv2D branches -> Concat -> Conv2D(64) -> GlobalAveragePooling2D -> Dense(latent)
  Decoder: Dense -> Reshape -> Conv2DTranspose layers -> output (linear activation)
"""

from __future__ import annotations

import logging
import math

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

logger = logging.getLogger(__name__)


def build_mscnn_ae(
    nx: int,
    ny: int,
    latent_dim: int,
    conv_filters: list[int] | None = None,
    conv_kernels: list[int] | None = None,
    reduction_filters: int = 64,
) -> tuple[keras.Model, keras.Model]:
    """Build the MSCNN Autoencoder.

    Args:
        nx, ny: spatial dimensions after 2D reshape
        latent_dim: bottleneck dimension
        conv_filters: number of filters per branch (default [32, 32, 32])
        conv_kernels: kernel sizes per branch (default [1, 3, 5])
        reduction_filters: filters for the reduction conv layer

    Returns:
        (full_model, encoder_model)
        - full_model: input -> reconstructed output
        - encoder_model: input -> latent vector
    """
    if conv_filters is None:
        conv_filters = [32, 32, 32]
    if conv_kernels is None:
        conv_kernels = [1, 3, 5]

    inp = layers.Input(shape=(nx, ny, 1), name="input")

    # --- Encoder: Multi-Scale Conv2D branches ---
    branches = []
    for i, (n_filt, k_size) in enumerate(zip(conv_filters, conv_kernels)):
        pad = "same"
        branch = layers.Conv2D(
            n_filt, (k_size, k_size), padding=pad,
            activation="relu", name=f"enc_branch{i+1}_conv"
        )(inp)
        branch = layers.BatchNormalization(name=f"enc_branch{i+1}_bn")(branch)
        branches.append(branch)

    merged = layers.Concatenate(name="enc_merge")(branches)
    x = layers.Conv2D(
        reduction_filters, (3, 3), padding="same",
        activation="relu", name="enc_reduce_conv"
    )(merged)
    x = layers.BatchNormalization(name="enc_reduce_bn")(x)

    # GlobalAveragePooling2D — avoids Flatten which destroys spatial structure
    x = layers.GlobalAveragePooling2D(name="enc_gap")(x)

    latent = layers.Dense(latent_dim, activation="linear", name="latent")(x)

    # --- Decoder ---
    min_spatial = max(2, min(nx, ny) // 2)
    dec_init_filters = reduction_filters

    x = layers.Dense(
        min_spatial * min_spatial * dec_init_filters,
        activation="relu", name="dec_dense"
    )(latent)
    x = layers.Reshape((min_spatial, min_spatial, dec_init_filters), name="dec_reshape")(x)

    x = layers.Conv2DTranspose(
        reduction_filters, (3, 3), padding="same",
        activation="relu", name="dec_conv_t1"
    )(x)
    x = layers.BatchNormalization(name="dec_bn1")(x)

    # Upsample to reach (or slightly exceed) target spatial dims before cropping.
    # Use ceil so that after upsampling we have >= nx, ny and dec_crop can trim.
    if min_spatial < max(nx, ny):
        scale_x = max(1, math.ceil(nx / min_spatial))
        scale_y = max(1, math.ceil(ny / min_spatial))
        x = layers.UpSampling2D(size=(scale_x, scale_y), name="dec_upsample")(x)

    x = layers.Conv2DTranspose(
        32, (3, 3), padding="same",
        activation="relu", name="dec_conv_t2"
    )(x)
    x = layers.BatchNormalization(name="dec_bn2")(x)

    # Force exact spatial shape (nx, ny) so output always matches input
    def to_target_size(t, target_nx=nx, target_ny=ny):
        return tf.image.resize(t, [target_nx, target_ny], method="bilinear")
    x = layers.Lambda(to_target_size, name="dec_resize_to_target")(x)

    output = layers.Conv2DTranspose(
        1, (1, 1), padding="same",
        activation="linear", name="output"
    )(x)

    model = keras.Model(inp, output, name="MSCNN_AE")
    encoder = keras.Model(inp, latent, name="MSCNN_Encoder")

    _log_model_summary(model, latent_dim, nx, ny)
    return model, encoder


def _log_model_summary(model: keras.Model, latent_dim: int, nx: int, ny: int) -> None:
    """Log key architecture stats."""
    total_params = model.count_params()
    input_dim = nx * ny
    compression_ratio = input_dim / latent_dim if latent_dim > 0 else 0
    logger.info(
        "MSCNN-AE: input=(%d,%d,1), latent=%d, "
        "compression=%.1fx, params=%d",
        nx, ny, latent_dim, compression_ratio, total_params,
    )
    if compression_ratio < 5:
        logger.warning(
            "Low compression ratio (%.1fx) — bottleneck may be too large. "
            "Consider reducing latent_dim.", compression_ratio,
        )
