"""GRU autoencoder that repeats a latent vector through the decoder."""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn

from .utils import as_shape, recurrent_dropout, validate_3d_input, validate_window_shapes


class GRURepeatedAutoencoder(nn.Module):
    """GRU encoder-decoder for window reconstruction.

    Expects input with shape ``(batch, time_steps, channels)`` and returns the same shape.
    The encoder compresses the whole window into a single latent vector. The decoder
    reconstructs the complete sequence from that latent representation.
    """

    def __init__(
        self,
        input_shape: Iterable[int],
        output_shape: Iterable[int],
        hidden_size: int = 64,
        latent_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
        decoder_input: str = "latent",
    ) -> None:
        super().__init__()
        self.input_shape = as_shape(input_shape)
        self.output_shape = as_shape(output_shape)
        validate_window_shapes("GRURepeatedAutoencoder", self.input_shape, self.output_shape, same_shape=True)
        if hidden_size <= 0 or latent_size <= 0:
            raise ValueError("hidden_size and latent_size must be positive.")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if decoder_input not in {"latent", "zeros"}:
            raise ValueError("decoder_input must be either 'latent' or 'zeros'.")

        time_steps, channels = self.input_shape
        self.time_steps = time_steps
        self.channels = channels
        self.hidden_size = int(hidden_size)
        self.latent_size = int(latent_size)
        self.num_layers = int(num_layers)
        self.decoder_input_mode = decoder_input

        recurrent_dropout_p = recurrent_dropout(dropout, self.num_layers)
        self.encoder = nn.GRU(
            input_size=self.channels,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=recurrent_dropout_p,
            batch_first=True,
        )
        self.to_latent = nn.Linear(self.hidden_size, self.latent_size)
        self.latent_activation = nn.Tanh()

        self.decoder_input_projection = nn.Linear(self.latent_size, self.hidden_size)
        self.decoder_hidden_projection = nn.Linear(self.latent_size, self.num_layers * self.hidden_size)
        self.decoder = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=recurrent_dropout_p,
            batch_first=True,
        )
        self.output_projection = nn.Linear(self.hidden_size, self.output_shape[1])

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.encoder(x)
        top_hidden = hidden[-1]
        latent = self.latent_activation(self.to_latent(top_hidden))
        return latent

    def _build_decoder_inputs(self, latent: torch.Tensor) -> torch.Tensor:
        batch_size = latent.shape[0]
        if self.decoder_input_mode == "zeros":
            return torch.zeros(
                batch_size,
                self.time_steps,
                self.hidden_size,
                device=latent.device,
                dtype=latent.dtype,
            )

        repeated = self.decoder_input_projection(latent).unsqueeze(1)
        return repeated.repeat(1, self.time_steps, 1)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        decoder_inputs = self._build_decoder_inputs(latent)
        hidden = (
            self.decoder_hidden_projection(latent)
            .view(latent.shape[0], self.num_layers, self.hidden_size)
            .transpose(0, 1)
            .contiguous()
        )
        decoded, _ = self.decoder(decoder_inputs, hidden)
        return self.output_projection(decoded)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        validate_3d_input("GRURepeatedAutoencoder", x, self.input_shape)
        latent = self.encode(x)
        reconstruction = self.decode(latent)
        return reconstruction
