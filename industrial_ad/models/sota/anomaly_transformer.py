"""Anomaly Transformer (Xu et al., ICLR 2022) — port of the official implementation.

Source: https://github.com/thuml/Anomaly-Transformer (MIT, Copyright (c) 2022 THUML @ Tsinghua University), `model/{embed,attn,
AnomalyTransformer}.py` and `solver.py` (minimax objective + anomaly criterion).
Interface adaptations are marked `ADAPTED:`.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch import nn

from industrial_ad.models.utils import as_shape, validate_3d_input, validate_window_shapes


def kl_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """`my_kl_loss` from solver.py."""
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, : x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in: int, d_model: int) -> None:
        super().__init__()
        self.tokenConv = nn.Conv1d(
            in_channels=c_in, out_channels=d_model, kernel_size=3,
            padding=1, padding_mode="circular", bias=False,
        )
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)


class DataEmbedding(nn.Module):
    def __init__(self, c_in: int, d_model: int, dropout: float = 0.0, max_len: int = 5000) -> None:
        super().__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        # ADAPTED: the original hard-codes max_len=5000 rows. Those rows are a
        # persistent buffer, so they land in the state dict and inflate the
        # reported model size by 10 MB (1x5000x512 floats) while only `win_size`
        # of them can ever be indexed. Sized to the window instead — this makes
        # the size comparison fair to the baseline, not to us.
        self.position_embedding = PositionalEmbedding(d_model=d_model, max_len=max_len)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.value_embedding(x) + self.position_embedding(x))


class AnomalyAttention(nn.Module):
    def __init__(self, win_size: int, scale: float | None = None, attention_dropout: float = 0.0) -> None:
        super().__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)
        # ADAPTED: the original allocates this on `.cuda()`; a registered buffer
        # follows the module device and keeps CPU inference/benchmarking working.
        distances = torch.zeros((win_size, win_size))
        for i in range(win_size):
            for j in range(win_size):
                distances[i][j] = abs(i - j)
        self.register_buffer("distances", distances, persistent=False)

    def forward(self, queries, keys, values, sigma):
        B, L, H, E = queries.shape
        scale = self.scale or 1.0 / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        attn = scale * scores

        sigma = sigma.transpose(1, 2)  # B L H -> B H L
        window_size = attn.shape[-1]
        sigma = torch.sigmoid(sigma * 5) + 1e-5
        sigma = torch.pow(3, sigma) - 1
        sigma = sigma.unsqueeze(-1).repeat(1, 1, 1, window_size)  # B H L L
        prior = self.distances.unsqueeze(0).unsqueeze(0).repeat(sigma.shape[0], sigma.shape[1], 1, 1)
        prior = 1.0 / (math.sqrt(2 * math.pi) * sigma) * torch.exp(-(prior**2) / 2 / (sigma**2))

        series = self.dropout(torch.softmax(attn, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", series, values)
        return V.contiguous(), series, prior, sigma


class AttentionLayer(nn.Module):
    def __init__(self, attention: nn.Module, d_model: int, n_heads: int) -> None:
        super().__init__()
        d_keys = d_model // n_heads
        d_values = d_model // n_heads
        self.norm = nn.LayerNorm(d_model)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.sigma_projection = nn.Linear(d_model, n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        x = queries
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        sigma = self.sigma_projection(x).view(B, L, H)

        out, series, prior, sigma = self.inner_attention(queries, keys, values, sigma)
        out = out.view(B, L, -1)
        return self.out_projection(out), series, prior, sigma


class EncoderLayer(nn.Module):
    def __init__(self, attention: nn.Module, d_model: int, d_ff: int | None = None,
                 dropout: float = 0.1, activation: str = "relu") -> None:
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x):
        new_x, attn, mask, sigma = self.attention(x, x, x)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y), attn, mask, sigma


class AnomalyTransformerDetector(nn.Module):
    """Anomaly Transformer with the association-discrepancy minimax objective."""

    def __init__(
        self,
        input_shape: Iterable[int],
        output_shape: Iterable[int],
        d_model: int = 512,
        n_heads: int = 8,
        e_layers: int = 3,
        d_ff: int = 512,
        dropout: float = 0.0,
        activation: str = "gelu",
        k: float = 3.0,
        temperature: float = 50.0,
    ) -> None:
        super().__init__()
        self.input_shape = as_shape(input_shape)
        self.output_shape = as_shape(output_shape)
        validate_window_shapes("AnomalyTransformer", self.input_shape, self.output_shape, same_shape=True)

        self.win_size, self.channels = self.input_shape
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        self.k = float(k)
        self.temperature = float(temperature)

        self.embedding = DataEmbedding(self.channels, d_model, dropout, max_len=self.win_size)
        self.attn_layers = nn.ModuleList(
            [
                EncoderLayer(
                    AttentionLayer(AnomalyAttention(self.win_size, attention_dropout=dropout), d_model, n_heads),
                    d_model, d_ff, dropout=dropout, activation=activation,
                )
                for _ in range(int(e_layers))
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.projection = nn.Linear(d_model, self.channels, bias=True)

    def forward_with_association(self, x: torch.Tensor):
        validate_3d_input("AnomalyTransformer", x, self.input_shape)
        out = self.embedding(x)
        series_list, prior_list = [], []
        for layer in self.attn_layers:
            out, series, prior, _ = layer(out)
            series_list.append(series)
            prior_list.append(prior)
        out = self.norm(out)
        return self.projection(out), series_list, prior_list

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruction only, shaped like the input."""
        reconstruction, _, _ = self.forward_with_association(x)
        return reconstruction

    @staticmethod
    def _normalized_prior(prior: torch.Tensor, win_size: int) -> torch.Tensor:
        return prior / torch.unsqueeze(torch.sum(prior, dim=-1), dim=-1).repeat(1, 1, 1, win_size)

    def loss_phases(self, x: torch.Tensor, y: torch.Tensor, criterion, ctx: dict[str, Any]):
        """Minimax objective from solver.py.

        The original calls `loss1.backward(retain_graph=True); loss2.backward();
        optimizer.step()` — a single step on the summed gradients, which is what
        the summed loss below produces.
        """
        reconstruction, series_list, prior_list = self.forward_with_association(x)

        series_loss = 0.0
        prior_loss = 0.0
        for series, prior in zip(series_list, prior_list):
            normalized = self._normalized_prior(prior, self.win_size)
            series_loss = series_loss + (
                torch.mean(kl_divergence(series, normalized.detach()))
                + torch.mean(kl_divergence(normalized.detach(), series))
            )
            prior_loss = prior_loss + (
                torch.mean(kl_divergence(normalized, series.detach()))
                + torch.mean(kl_divergence(series.detach(), normalized))
            )
        series_loss = series_loss / len(prior_list)
        prior_loss = prior_loss / len(prior_list)

        rec_loss = criterion(reconstruction, y)
        loss1 = rec_loss - self.k * series_loss  # minimize phase
        loss2 = rec_loss + self.k * prior_loss  # maximize phase
        return [(loss1 + loss2, None)]

    @torch.no_grad()
    def anomaly_error(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Author criterion: softmax(-AssDis) * reconstruction error, per timestep."""
        reconstruction, series_list, prior_list = self.forward_with_association(x)
        errors = torch.mean((y - reconstruction) ** 2, dim=-1)  # (batch, win_size)

        series_loss = 0.0
        prior_loss = 0.0
        for series, prior in zip(series_list, prior_list):
            normalized = self._normalized_prior(prior, self.win_size)
            series_loss = series_loss + kl_divergence(series, normalized.detach()) * self.temperature
            prior_loss = prior_loss + kl_divergence(normalized, series.detach()) * self.temperature

        metric = torch.softmax(-series_loss - prior_loss, dim=-1)
        return metric * errors
