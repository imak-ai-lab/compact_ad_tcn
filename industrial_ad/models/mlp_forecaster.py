"""Direct MLP forecaster for fixed-window multi-step prediction."""

from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import nn
from .utils import as_shape, build_activation, validate_3d_input, validate_window_shapes


class MLPForecaster(nn.Module):
    """Direct MLP forecaster for `(time, channels)` windows."""

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
        validate_window_shapes("MLPForecaster", self.input_shape, self.output_shape, same_channels=True)
        if not hidden_dims:
            raise ValueError("MLPForecaster requires at least one hidden dimension.")

        input_dim = math.prod(self.input_shape)
        output_dim = math.prod(self.output_shape)
        self.network = self._build_network(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=[int(value) for value in hidden_dims],
            activation=activation,
            dropout=float(dropout),
        )

    @staticmethod
    def _build_network(
        *,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int],
        activation: str,
        dropout: float,
    ) -> nn.Sequential:
        dims = [input_dim, *hidden_dims, output_dim]
        layers: list[nn.Module] = []
        for index, (in_features, out_features) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(in_features, out_features))
            is_last = index == len(dims) - 2
            if not is_last:
                layers.append(build_activation(activation))
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        validate_3d_input("MLPForecaster", x, self.input_shape)
        batch_size = x.shape[0]
        flattened = x.swapaxes(-1, -2).reshape(batch_size, -1)
        prediction = self.network(flattened)
        forecast_steps, channels = self.output_shape
        return prediction.reshape(batch_size, channels, forecast_steps).swapaxes(-1, -2)
