"""Temporal convolutional autoencoder with residual dilated blocks."""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
from torch import nn
from .utils import as_shape, build_activation, make_dilations, validate_3d_input, validate_window_shapes


class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise-separable 1D convolution to reduce parameter count."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve sequence length.")
        padding = dilation * (kernel_size - 1) // 2
        self.depthwise = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class TemporalResidualBlock(nn.Module):
    """Residual TCN block with optional depthwise-separable convolutions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        dilation: int,
        activation: str = "relu",
        dropout: float = 0.0,
        separable: bool = True,
        norm: str = "batch",
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("TemporalResidualBlock requires an odd kernel_size.")

        conv = DepthwiseSeparableConv1d if separable else nn.Conv1d
        padding = dilation * (kernel_size - 1) // 2

        if separable:
            self.conv1 = conv(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation, bias=False)
            self.conv2 = conv(out_channels, out_channels, kernel_size=kernel_size, dilation=dilation, bias=False)
        else:
            self.conv1 = conv(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            )
            self.conv2 = conv(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            )

        self.norm1 = self._build_norm(norm, out_channels)
        self.norm2 = self._build_norm(norm, out_channels)
        self.activation1 = build_activation(activation)
        self.activation2 = build_activation(activation)
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False) if in_channels != out_channels else nn.Identity()

    @staticmethod
    def _build_norm(norm: str, channels: int) -> nn.Module:
        kind = norm.lower()
        if kind == "batch":
            return nn.BatchNorm1d(channels)
        if kind == "layer":
            return nn.GroupNorm(num_groups=1, num_channels=channels)
        if kind == "none":
            return nn.Identity()
        raise ValueError(f"Unsupported norm: {norm}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        y = self.conv1(x)
        y = self.norm1(y)
        y = self.activation1(y)
        y = self.dropout1(y)

        y = self.conv2(y)
        y = self.norm2(y)
        y = self.activation2(y)
        y = self.dropout2(y)
        return y + residual


class TCNAutoencoder(nn.Module):
    """TCN autoencoder for window reconstruction.

    Expects input with shape ``(batch, time_steps, channels)`` and returns the same shape.
    The encoder and decoder are mirror-symmetric in dilation schedule.
    """

    def __init__(
        self,
        input_shape: Iterable[int],
        output_shape: Iterable[int],
        hidden_channels: int = 32,
        latent_channels: int = 8,
        num_blocks: int = 5,
        kernel_size: int = 3,
        activation: str = "relu",
        dropout: float = 0.0,
        separable: bool = True,
        norm: str = "batch",
        dilations: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        self.input_shape = as_shape(input_shape)
        self.output_shape = as_shape(output_shape)
        validate_window_shapes("TCNAutoencoder", self.input_shape, self.output_shape, same_shape=True)
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd.")
        if hidden_channels <= 0 or latent_channels <= 0:
            raise ValueError("hidden_channels and latent_channels must be positive.")
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive.")

        time_steps, in_channels = self.input_shape
        self.time_steps = time_steps
        self.in_channels = in_channels
        self.hidden_channels = int(hidden_channels)
        self.latent_channels = int(latent_channels)
        self.num_blocks = int(num_blocks)
        self.kernel_size = int(kernel_size)

        self.dilations = make_dilations(self.num_blocks, dilations)

        self.input_projection = nn.Conv1d(in_channels, hidden_channels, kernel_size=1, bias=False)

        self.encoder = nn.Sequential(
            *[
                TemporalResidualBlock(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    activation=activation,
                    dropout=dropout,
                    separable=separable,
                    norm=norm,
                )
                for dilation in self.dilations
            ]
        )

        self.to_latent = nn.Conv1d(hidden_channels, latent_channels, kernel_size=1, bias=False)
        self.latent_norm = TemporalResidualBlock._build_norm(norm, latent_channels)
        self.latent_activation = build_activation(activation)

        self.from_latent = nn.Conv1d(latent_channels, hidden_channels, kernel_size=1, bias=False)
        self.decoder = nn.Sequential(
            *[
                TemporalResidualBlock(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    activation=activation,
                    dropout=dropout,
                    separable=separable,
                    norm=norm,
                )
                for dilation in reversed(self.dilations)
            ]
        )
        self.output_projection = nn.Conv1d(hidden_channels, self.output_shape[1], kernel_size=1, bias=True)

    @property
    def receptive_field(self) -> int:
        # There are two convolutions per block in the encoder and two more in the mirrored decoder.
        return 1 + 4 * (self.kernel_size - 1) * sum(self.dilations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        validate_3d_input("TCNAutoencoder", x, self.input_shape)
        # Conv1d uses (batch, channels, time).
        y = x.transpose(1, 2)
        y = self.input_projection(y)
        y = self.encoder(y)
        y = self.to_latent(y)
        y = self.latent_norm(y)
        y = self.latent_activation(y)
        y = self.from_latent(y)
        y = self.decoder(y)
        y = self.output_projection(y)
        return y.transpose(1, 2)
