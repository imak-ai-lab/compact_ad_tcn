"""Lightweight TCN forecaster with a shared temporal mixer head."""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
from torch import nn
from .utils import as_shape, build_activation, make_dilations, validate_3d_input, validate_window_shapes
from .tcn_forecaster import TemporalResidualBlock


class SharedTemporalHiddenMixerHead(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        output_channels: int,
        final_steps: int,
        horizon: int,
        temporal_bases: int = 1,
        mixer_channels: int = 64,
        activation: str = "relu",
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.final_steps = final_steps
        self.horizon = horizon
        self.temporal_bases = temporal_bases

        self.temporal = nn.Linear(final_steps, temporal_bases * horizon)

        act = build_activation(activation)

        self.mixer = nn.Sequential(
            nn.Conv1d(hidden_channels * temporal_bases, mixer_channels, kernel_size=1, bias=True),
            act,
            nn.Conv1d(mixer_channels, output_channels, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, hidden_channels, final_steps)
        b, d, _ = x.shape

        y = self.temporal(x)
        # y: (B, hidden_channels, temporal_bases * horizon)

        if self.temporal_bases > 1:
            y = y.view(b, d, self.temporal_bases, self.horizon)
            y = y.reshape(b, d * self.temporal_bases, self.horizon)
        else:
            y = y.view(b, d, self.horizon)

        y = self.mixer(y)
        # y: (B, output_channels, horizon)

        return y
    

class TCNLightForecaster(nn.Module):
    """Direct TCN forecaster for `(time, channels)` windows."""

    def __init__(
        self,
        input_shape: Iterable[int],
        output_shape: Iterable[int],
        hidden_channels: int = 32,
        num_blocks: int = 5,
        kernel_size: int = 3,
        activation: str = "relu",
        dropout: float = 0.0,
        separable: bool = True,
        norm: str = "layer",
        dilations: Sequence[int] | None = None,
        final_steps: int | None = None,
        head_temporal_bases: int = 1,
        head_mixer_channels: int = 64,
    ) -> None:
        super().__init__()
        self.input_shape = as_shape(input_shape)
        self.output_shape = as_shape(output_shape)
        validate_window_shapes("TCNLightForecaster", self.input_shape, self.output_shape, same_channels=True)
        if hidden_channels <= 0 or num_blocks <= 0:
            raise ValueError("hidden_channels and num_blocks must be positive.")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd.")

        self.history_steps, self.channels = self.input_shape
        self.forecast_steps = int(self.output_shape[0])
        self.hidden_channels = int(hidden_channels)
        self.kernel_size = int(kernel_size)
        self.num_blocks = int(num_blocks)
        self.final_steps = int(final_steps) if final_steps is not None else self.forecast_steps
        self.head_temporal_bases = int(head_temporal_bases)
        self.head_mixer_channels = int(head_mixer_channels)

        self.dilations = make_dilations(self.num_blocks, dilations)

        self.input_projection = nn.Conv1d(self.channels, self.hidden_channels, kernel_size=1, bias=False)
        self.blocks = nn.Sequential(
            *[
                TemporalResidualBlock(
                    self.hidden_channels,
                    self.hidden_channels,
                    kernel_size=self.kernel_size,
                    dilation=dilation,
                    activation=activation,
                    dropout=float(dropout),
                    separable=bool(separable),
                    norm=norm,
                )
                for dilation in self.dilations
            ]
        )
        # self.output_projection = nn.Conv1d(self.hidden_channels, self.channels, kernel_size=1, bias=False)
        self.forecast_head = SharedTemporalHiddenMixerHead(
            self.hidden_channels,
            self.channels,
            self.final_steps,
            self.forecast_steps,
            temporal_bases=self.head_temporal_bases,
            mixer_channels=self.head_mixer_channels,
            activation=activation
        )
    
    @property
    def receptive_field(self) -> int:
        return 1 + 2 * (self.kernel_size - 1) * sum(self.dilations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        validate_3d_input("TCNLightForecaster", x, self.input_shape)
        y = x.transpose(1, 2)
        y = self.input_projection(y)
        y = self.blocks(y)
        context = y[:, :, -self.final_steps :]
        prediction = self.forecast_head(context)
        return prediction.transpose(1, 2)
