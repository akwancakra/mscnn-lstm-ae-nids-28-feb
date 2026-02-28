"""Stage 2: LSTM Autoencoder for temporal pattern learning.

Operates on sequences of latent vectors from Stage 1.
Input shape: (W, latent_dim) where W = window size.

When W=1 (per-flow fallback), uses a Dense Autoencoder instead.

Architecture (W > 1):
  Encoder: LSTM(32) -> Dropout(0.3) -> Dense(temporal_latent_dim, linear)
  Decoder: RepeatVector(W) -> LSTM(32) -> Dropout(0.3) ->
           TimeDistributed(Dense(64) -> BN -> ReLU) ->
           TimeDistributed(Dense(latent_dim, linear))

Architecture (W = 1, Dense fallback):
  Encoder: Dense(64) -> BN -> ReLU -> Dense(temporal_latent_dim, linear)
  Decoder: Dense(64) -> BN -> ReLU -> Dense(latent_dim, linear)
"""

from __future__ import annotations

import logging

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

logger = logging.getLogger(__name__)


def compute_temporal_latent_dim(latent_dim: int) -> int:
    """Compute temporal bottleneck: max(2, latent_dim // 4)."""
    return max(2, latent_dim // 4)


def build_lstm_ae(
    window_size: int,
    latent_dim: int,
    temporal_latent_dim: int | None = None,
    lstm_units: int = 32,
    dropout: float = 0.3,
) -> tuple[keras.Model, keras.Model]:
    """Build LSTM-AE (or Dense-AE for W=1).

    Args:
        window_size: sequence length W
        latent_dim: dimension of each latent vector from Stage 1
        temporal_latent_dim: bottleneck for temporal compression
        lstm_units: number of LSTM units
        dropout: dropout rate

    Returns:
        (full_model, encoder_model)
    """
    if temporal_latent_dim is None:
        temporal_latent_dim = compute_temporal_latent_dim(latent_dim)

    if window_size <= 1:
        return _build_dense_ae(latent_dim, temporal_latent_dim)

    inp = layers.Input(shape=(window_size, latent_dim), name="seq_input")

    # --- Encoder ---
    x = layers.LSTM(lstm_units, return_sequences=False, name="enc_lstm")(inp)
    x = layers.Dropout(dropout, name="enc_dropout")(x)
    temporal_latent = layers.Dense(
        temporal_latent_dim, activation="linear", name="temporal_latent"
    )(x)

    # --- Decoder ---
    x = layers.RepeatVector(window_size, name="dec_repeat")(temporal_latent)
    x = layers.LSTM(lstm_units, return_sequences=True, name="dec_lstm")(x)
    x = layers.Dropout(dropout, name="dec_dropout")(x)
    x = layers.TimeDistributed(
        layers.Dense(64, activation="relu"), name="dec_td_dense1"
    )(x)
    x = layers.TimeDistributed(
        layers.BatchNormalization(), name="dec_td_bn"
    )(x)
    output = layers.TimeDistributed(
        layers.Dense(latent_dim, activation="linear"), name="dec_output"
    )(x)

    model = keras.Model(inp, output, name="LSTM_AE")
    encoder = keras.Model(inp, temporal_latent, name="LSTM_Encoder")

    total_params = model.count_params()
    compression = (window_size * latent_dim) / temporal_latent_dim
    logger.info(
        "LSTM-AE: input=(%d,%d), temporal_latent=%d, "
        "compression=%.1fx, params=%d",
        window_size, latent_dim, temporal_latent_dim,
        compression, total_params,
    )
    return model, encoder


def _build_dense_ae(
    latent_dim: int, temporal_latent_dim: int
) -> tuple[keras.Model, keras.Model]:
    """Dense AE fallback for per-flow mode (W=1)."""
    inp = layers.Input(shape=(1, latent_dim), name="seq_input")
    x = layers.Flatten(name="flatten")(inp)

    # Encoder
    x = layers.Dense(64, activation="relu", name="enc_dense1")(x)
    x = layers.BatchNormalization(name="enc_bn")(x)
    temporal_latent = layers.Dense(
        temporal_latent_dim, activation="linear", name="temporal_latent"
    )(x)

    # Decoder
    x = layers.Dense(64, activation="relu", name="dec_dense1")(temporal_latent)
    x = layers.BatchNormalization(name="dec_bn")(x)
    x = layers.Dense(latent_dim, activation="linear", name="dec_dense_out")(x)
    output = layers.Reshape((1, latent_dim), name="dec_reshape")(x)

    model = keras.Model(inp, output, name="Dense_AE")
    encoder = keras.Model(inp, temporal_latent, name="Dense_Encoder")

    logger.info(
        "Dense-AE (W=1 fallback): latent_dim=%d, temporal_latent=%d, params=%d",
        latent_dim, temporal_latent_dim, model.count_params(),
    )
    return model, encoder
