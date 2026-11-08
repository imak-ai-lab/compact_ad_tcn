"""Transformer autoencoder with per-timestep latent bottleneck."""

from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import nn
from .utils import as_shape, build_activation, validate_3d_input, validate_window_shapes


class SinusoidalPositionalEncoding(nn.Module):
    """Parameter-free sinusoidal positional encoding from the original Transformer."""

    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive.")
        if max_len <= 0:
            raise ValueError("max_len must be positive.")

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # (1, T, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)].to(dtype=x.dtype, device=x.device)


class TransformerAutoencoder(nn.Module):
    """Compact Transformer autoencoder for window reconstruction.

    Design choice:
    - sequence-valued latent representation (one latent token per timestep),
    - Transformer encoder stack for the encoder,
    - narrow per-token bottleneck,
    - second Transformer encoder stack used as the decoder.

    This keeps the interface identical in spirit to the existing WindowMLP baseline:
    input and output are both shaped as (batch, time_steps, channels).
    """

    def __init__(
        self,
        input_shape: Iterable[int],
        output_shape: Iterable[int],
        d_model: int = 48,
        nhead: int = 4,
        num_layers: int = 2,
        latent_dim: int | None = None,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
        activation: str = "gelu",
        norm_first: bool = True,
    ) -> None:
        super().__init__()
        self.input_shape = as_shape(input_shape)
        self.output_shape = as_shape(output_shape)
        validate_window_shapes("TransformerAutoencoder", self.input_shape, self.output_shape, same_shape=True)
        if d_model <= 0 or nhead <= 0 or num_layers <= 0:
            raise ValueError("d_model, nhead and num_layers must be positive.")
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead.")

        self.time_steps, self.channels = self.input_shape
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.num_layers = int(num_layers)
        self.latent_dim = int(latent_dim) if latent_dim is not None else max(8, d_model // 4)
        self.dim_feedforward = int(dim_feedforward) if dim_feedforward is not None else 2 * self.d_model
        self.dropout_p = float(dropout)
        self.activation_name = activation.lower()
        if self.activation_name not in {"relu", "gelu"}:
            raise ValueError("TransformerEncoderLayer supports activation='relu' or 'gelu' in this implementation.")

        self.input_projection = nn.Linear(self.channels, self.d_model)
        self.positional_encoding = SinusoidalPositionalEncoding(d_model=self.d_model, max_len=self.time_steps)
        self.input_norm = nn.LayerNorm(self.d_model)
        self.input_dropout = nn.Dropout(self.dropout_p) if self.dropout_p > 0 else nn.Identity()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout_p,
            activation=self.activation_name,
            batch_first=True,
            norm_first=bool(norm_first),
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers,
            norm=nn.LayerNorm(self.d_model),
            enable_nested_tensor=False,
        )

        self.to_latent_norm = nn.LayerNorm(self.d_model)
        self.to_latent = nn.Linear(self.d_model, self.latent_dim)
        self.latent_activation = build_activation("gelu" if self.activation_name == "gelu" else "relu")
        self.latent_dropout = nn.Dropout(self.dropout_p) if self.dropout_p > 0 else nn.Identity()
        self.from_latent = nn.Linear(self.latent_dim, self.d_model)
        self.decoder_input_norm = nn.LayerNorm(self.d_model)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout_p,
            activation=self.activation_name,
            batch_first=True,
            norm_first=bool(norm_first),
        )
        self.decoder = nn.TransformerEncoder(
            decoder_layer,
            num_layers=self.num_layers,
            norm=nn.LayerNorm(self.d_model),
            enable_nested_tensor=False,
        )

        self.output_projection = nn.Linear(self.d_model, self.channels)
        self.reset_parameters()

    @staticmethod
    def _reset_layer_norm(layer_norm: nn.LayerNorm) -> None:
        if layer_norm.weight is not None:
            nn.init.ones_(layer_norm.weight)
        if layer_norm.bias is not None:
            nn.init.zeros_(layer_norm.bias)

    def _reinit_transformer_stack(self, stack: nn.TransformerEncoder) -> None:
        for layer in stack.layers:
            nn.init.xavier_uniform_(layer.self_attn.in_proj_weight)
            if layer.self_attn.in_proj_bias is not None:
                nn.init.zeros_(layer.self_attn.in_proj_bias)
            nn.init.xavier_uniform_(layer.self_attn.out_proj.weight)
            if layer.self_attn.out_proj.bias is not None:
                nn.init.zeros_(layer.self_attn.out_proj.bias)

            nn.init.xavier_uniform_(layer.linear1.weight)
            nn.init.zeros_(layer.linear1.bias)
            nn.init.xavier_uniform_(layer.linear2.weight)
            nn.init.zeros_(layer.linear2.bias)

            self._reset_layer_norm(layer.norm1)
            self._reset_layer_norm(layer.norm2)
        if stack.norm is not None:
            self._reset_layer_norm(stack.norm)

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)
        self._reset_layer_norm(self.input_norm)

        self._reinit_transformer_stack(self.encoder)

        self._reset_layer_norm(self.to_latent_norm)
        nn.init.xavier_uniform_(self.to_latent.weight)
        nn.init.zeros_(self.to_latent.bias)
        nn.init.xavier_uniform_(self.from_latent.weight)
        nn.init.zeros_(self.from_latent.bias)
        self._reset_layer_norm(self.decoder_input_norm)

        self._reinit_transformer_stack(self.decoder)

        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        y = self.input_projection(x)
        y = self.positional_encoding(y)
        y = self.input_norm(y)
        y = self.input_dropout(y)
        y = self.encoder(y)
        y = self.to_latent_norm(y)
        y = self.to_latent(y)
        y = self.latent_activation(y)
        y = self.latent_dropout(y)
        return y  # (batch, time_steps, latent_dim)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        y = self.from_latent(z)
        y = self.decoder_input_norm(y)
        y = self.decoder(y)
        return self.output_projection(y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        validate_3d_input("TransformerAutoencoder", x, self.input_shape)
        z = self.encode(x)
        return self.decode(z)
