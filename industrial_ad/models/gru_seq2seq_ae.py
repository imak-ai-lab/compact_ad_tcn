"""Seq2seq GRU autoencoder with optional teacher forcing."""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn

from .utils import as_shape, recurrent_dropout, validate_3d_input, validate_window_shapes


class GRUSeq2seqAutoencoder(nn.Module):
    """Canonical GRU encoder-decoder for fixed-window reconstruction.

    Input and output shapes are ``(batch, time_steps, channels)``.

    Main differences from the previous hybrid GRU implementation:
    1. The decoder is conditioned through its initial hidden state, as in classical
       seq2seq encoder-decoder models for reconstruction.
    2. The decoder consumes a shifted sequence of previous values during training
       (teacher forcing / scheduled sampling) and free-runs during evaluation.
    3. Multi-layer hidden states are kept in the correct ``(layers, batch, hidden)``
       layout with no batch mixing.
    """

    def __init__(
        self,
        input_shape: Iterable[int],
        output_shape: Iterable[int],
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        teacher_forcing_ratio: float = 0.0,
        reverse_target: bool = False,
    ) -> None:
        super().__init__()
        self.input_shape = as_shape(input_shape)
        self.output_shape = as_shape(output_shape)
        validate_window_shapes("GRUSeq2seqAutoencoder", self.input_shape, self.output_shape, same_shape=True)
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if not 0.0 <= float(teacher_forcing_ratio) <= 1.0:
            raise ValueError("teacher_forcing_ratio must be in [0, 1].")

        self.time_steps, self.channels = self.input_shape
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.teacher_forcing_ratio = float(teacher_forcing_ratio)
        self.reverse_target = bool(reverse_target)

        recurrent_dropout_p = recurrent_dropout(dropout, self.num_layers)

        self.encoder = nn.GRU(
            input_size=self.channels,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=recurrent_dropout_p,
            batch_first=True,
        )

        self.decoder = nn.GRU(
            input_size=self.channels,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=recurrent_dropout_p,
            batch_first=True,
        )
        self.output_projection = nn.Linear(self.hidden_size, self.channels)
        self.register_buffer("start_token", torch.zeros(1, 1, self.channels), persistent=False)

    def _prepare_target(self, x: torch.Tensor) -> torch.Tensor:
        return x.flip(dims=(1,)) if self.reverse_target else x

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.encoder(x)
        return hidden

    def decode(self, target: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        batch_size = target.shape[0]
        device = target.device
        dtype = target.dtype

        prev = self.start_token.expand(batch_size, 1, self.channels).to(device=device, dtype=dtype)
        predictions = []

        for t in range(self.time_steps):
            out, hidden = self.decoder(prev, hidden)
            pred = self.output_projection(out)
            predictions.append(pred)

            if t + 1 == self.time_steps:
                continue

            if self.training and self.teacher_forcing_ratio > 0.0:
                teacher_mask = (torch.rand(batch_size, 1, 1, device=device) < self.teacher_forcing_ratio)
                prev = torch.where(teacher_mask, target[:, t : t + 1, :], pred)
            else:
                prev = pred

        return torch.cat(predictions, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        validate_3d_input("GRUSeq2seqAutoencoder", x, self.input_shape)
        target = self._prepare_target(x)
        hidden = self.encode(x)
        reconstruction = self.decode(target=target, hidden=hidden)
        if self.reverse_target:
            reconstruction = reconstruction.flip(dims=(1,))
        return reconstruction
