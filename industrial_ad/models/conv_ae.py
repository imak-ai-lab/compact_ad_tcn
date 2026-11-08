"""Convolutional autoencoder architecture for window reconstruction."""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn
from .utils import as_shape, build_activation, validate_3d_input, validate_window_shapes


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, activation: str) -> None:
        super().__init__(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            build_activation(activation),
        )


class ConvAutoencoder(nn.Module):
    """Classical 1D convolutional autoencoder for window reconstruction.

    Architecture:
        Conv -> Act -> Pool -> ... -> Conv bottleneck -> Act
        -> Upsample -> Conv -> Act -> ... -> Conv output

    Input and output tensors use shape (batch, time_steps, channels).
    If the time length is not divisible by the total pooling factor, the model
    pads the time axis as evenly as possible on both sides and crops back to the
    original length after decoding.
    """

    def __init__(
        self,
        input_shape: Iterable[int],
        output_shape: Iterable[int],
        hidden_channels: Iterable[int] = (32, 64),
        latent_channels: int | None = None,
        kernel_size: int = 5,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.input_shape = as_shape(input_shape)
        self.output_shape = as_shape(output_shape)
        validate_window_shapes("ConvAutoencoder", self.input_shape, self.output_shape)
        if self.input_shape[0] != self.output_shape[0]:
            raise ValueError("ConvAutoencoder is intended for reconstruction, so input and output time_steps must match.")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve the temporal length with symmetric padding.")

        hidden_channels = [int(value) for value in hidden_channels]
        if not hidden_channels:
            raise ValueError("ConvAutoencoder requires at least one hidden channel width.")

        time_steps, input_channels = self.input_shape
        _, output_channels = self.output_shape
        depth = len(hidden_channels)
        reduction = 2**depth

        padded_time_steps = ((time_steps + reduction - 1) // reduction) * reduction
        total_padding = padded_time_steps - time_steps
        self.pad_left = total_padding // 2
        self.pad_right = total_padding - self.pad_left
        self.original_time_steps = time_steps

        latent_time_steps = padded_time_steps // reduction
        if latent_channels is None:
            latent_channels = min(input_channels, max(8, hidden_channels[-1] // 4))
        latent_channels = int(latent_channels)

        input_elements = time_steps * input_channels
        latent_elements = latent_time_steps * latent_channels
        if latent_elements >= input_elements:
            raise ValueError(
                "The bottleneck must be compressive. Choose fewer latent_channels or more pooling stages."
            )

        encoder_layers: list[nn.Module] = []
        in_channels = input_channels
        for out_channels_stage in hidden_channels:
            encoder_layers.append(ConvBlock(in_channels, out_channels_stage, kernel_size, activation))
            encoder_layers.append(nn.MaxPool1d(kernel_size=2))
            in_channels = out_channels_stage
        encoder_layers.append(ConvBlock(in_channels, latent_channels, kernel_size, activation))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: list[nn.Module] = []
        in_channels = latent_channels
        for out_channels_stage in reversed(hidden_channels):
            decoder_layers.append(nn.Upsample(scale_factor=2, mode="nearest"))
            decoder_layers.append(ConvBlock(in_channels, out_channels_stage, kernel_size, activation))
            in_channels = out_channels_stage
        decoder_layers.append(nn.Conv1d(in_channels, output_channels, kernel_size, padding=kernel_size // 2))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        validate_3d_input("ConvAutoencoder", x, self.input_shape)

        x = x.transpose(1, 2)
        if self.pad_left or self.pad_right:
            x = F.pad(x, (self.pad_left, self.pad_right))
        x = self.encoder(x)
        x = self.decoder(x)
        if self.pad_left or self.pad_right:
            x = x[..., self.pad_left : self.pad_left + self.original_time_steps]
        return x.transpose(1, 2)
