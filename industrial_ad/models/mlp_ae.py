"""MLP autoencoder architecture for fixed-window reconstruction."""

from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import nn
from .utils import as_shape, build_activation, validate_window_shapes


class MLPAutoencoder(nn.Module):
    """Mirror-symmetric MLP for window reconstruction or forecasting."""

    def __init__(
        self,
        input_shape: Iterable[int],
        output_shape: Iterable[int],
        hidden_dims: list[int],
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_shape = as_shape(input_shape)
        self.output_shape = as_shape(output_shape)
        validate_window_shapes("MLPAutoencoder", self.input_shape, self.output_shape)
        if not hidden_dims:
            raise ValueError("MLPAutoencoder requires at least one hidden dimension.")

        self.activation_name = activation
        self.dropout = float(dropout)
        input_dim = math.prod(self.input_shape)
        output_dim = math.prod(self.output_shape)
        self.network = self._build_network(input_dim=input_dim, output_dim=output_dim, hidden_dims=hidden_dims)

    def _build_network(self, *, input_dim: int, output_dim: int, hidden_dims: list[int]) -> nn.Sequential:
        """Build the same mirrored encoder/decoder pattern as the original baseline."""
        encoder_dims = [input_dim, *hidden_dims]
        decoder_dims = [*hidden_dims[::-1], output_dim]
        layers: list[nn.Module] = []

        # The first encoder layer stays linear-only to match the old baseline exactly.
        for index, (in_features, out_features) in enumerate(zip(encoder_dims[:-1], encoder_dims[1:])):
            if index > 0:
                layers.append(build_activation(self.activation_name))
                if self.dropout > 0:
                    layers.append(nn.Dropout(self.dropout))
            layers.append(nn.Linear(in_features, out_features))

        # Decoder depth and widths always mirror the encoder; there is no separate decoder config.
        for index, (in_features, out_features) in enumerate(zip(decoder_dims[:-1], decoder_dims[1:])):
            if index > 0:
                layers.append(build_activation(self.activation_name))
                if self.dropout > 0:
                    layers.append(nn.Dropout(self.dropout))
            layers.append(nn.Linear(in_features, out_features))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Flatten a `(time, channels)` window, run the MLP and restore the window shape."""
        batch_size = x.shape[0]
        # The legacy implementation flattened in channel-major order, so we keep the same layout unconditionally.
        flattened = x.swapaxes(-1, -2).reshape(batch_size, -1)
        prediction = self.network(flattened)
        time_steps, channels = self.output_shape
        return prediction.reshape(batch_size, channels, time_steps).swapaxes(-1, -2)
