"""TranAD (Tuli et al., VLDB 2022) — port of the official implementation.

Source: https://github.com/imperial-qore/TranAD (BSD 3-Clause, Copyright (c) 2022 Shreshth Tuli).
Layers below follow `src/dlutils.py`, the model follows `src/models.py::TranAD`,
and the objective follows `main.py::backprop` (the `'TranAD' in model.name` branch).

Interface adaptations for this repository are marked with `ADAPTED:`.

Two operating modes:

* ``inner_window=None`` (default) — one 120-frame window in, the LAST frame
  predicted, exactly as the first phase-1 runs (TA001..TA006). Kept unchanged so
  those results stay reproducible.
* ``inner_window=w`` — the AUTHOR-FAITHFUL mode. The 120-frame evaluation window
  is unfolded into ``120-w+1`` overlapping sub-windows of length ``w`` (stride 1,
  the original setting is ``w=10``), each sub-window predicts its own last frame,
  and the per-frame errors are aggregated back to one score per evaluation
  window. This gives TranAD the same amount of scored evidence per window that
  the original streaming protocol gives it, while the unit of evaluation and the
  label vector stay identical to every other model in the paper.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import torch
from torch import nn

from industrial_ad.models.utils import as_shape, validate_3d_input, validate_window_shapes


class PositionalEncoding(nn.Module):
    """Positional encoding exactly as in TranAD `src/dlutils.py` (sin+cos summed)."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model).float() * (-math.log(10000.0) / d_model))
        pe += torch.sin(position * div_term)
        pe += torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor, pos: int = 0) -> torch.Tensor:
        x = x + self.pe[pos : pos + x.size(0), :]
        return self.dropout(x)


class TranADEncoderLayer(nn.Module):
    """TranAD encoder layer (LeakyReLU FFN, no layer norm), sequence-first."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 16, dropout: float = 0.0) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(True)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        return src


class TranADDecoderLayer(nn.Module):
    """TranAD decoder layer (self-attention + cross-attention + FFN), sequence-first."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 16, dropout: float = 0.0) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(True)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        tgt2 = self.self_attn(tgt, tgt, tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.multihead_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


class TranAD(nn.Module):
    """TranAD: two-phase transformer with self-conditioning and an adversarial objective.

    The model predicts the LAST timestep of the window (as in the original code,
    where `elem = window[-1]`), twice: `x1` without self-conditioning and `x2`
    conditioned on the phase-1 error.
    """

    def __init__(
        self,
        input_shape: Iterable[int],
        output_shape: Iterable[int],
        dim_feedforward: int = 16,
        num_layers: int = 1,
        dropout: float = 0.1,
        nhead: int | None = None,
        output_activation: str = "identity",
        inner_window: int | None = None,
        score_aggregation: str = "mean",
        train_positions: int | None = None,
        eval_position_chunk: int = 32,
    ) -> None:
        super().__init__()
        self.input_shape = as_shape(input_shape)
        self.output_shape = as_shape(output_shape)
        validate_window_shapes("TranAD", self.input_shape, self.output_shape, same_shape=True)

        self.n_window, self.n_feats = self.input_shape
        d_model = 2 * self.n_feats
        # ADAPTED: nhead defaults to `feats` as in the original (d_model = 2*feats).
        self.nhead = int(nhead) if nhead is not None else self.n_feats
        if d_model % self.nhead != 0:
            raise ValueError(f"2*channels ({d_model}) must be divisible by nhead ({self.nhead}).")
        if num_layers <= 0 or dim_feedforward <= 0:
            raise ValueError("num_layers and dim_feedforward must be positive.")

        self.num_layers = int(num_layers)
        self.dim_feedforward = int(dim_feedforward)
        self.dropout_p = float(dropout)
        self.output_activation_name = str(output_activation).lower()

        # ADAPTED: author-faithful sliding mode (see the module docstring).
        self.inner_window = None if inner_window is None else int(inner_window)
        if self.inner_window is not None and not 1 < self.inner_window <= self.n_window:
            raise ValueError(f"inner_window must be in (1, {self.n_window}], got {self.inner_window}.")
        self.score_aggregation = str(score_aggregation).lower()
        if self.score_aggregation not in {"mean", "max"}:
            raise ValueError("score_aggregation must be 'mean' or 'max'.")
        self.train_positions = None if train_positions is None else int(train_positions)
        self.eval_position_chunk = int(eval_position_chunk)

        if self.inner_window is None:
            self.n_positions = 1
            seq_len = self.n_window
        else:
            # Stride is fixed at 1: every frame from index inner_window-1 on gets a
            # prediction, so the model output covers the LAST n_positions frames of
            # the window contiguously. phase1_rescore._UnifiedReducer aligns targets
            # as y[:, -prediction.shape[1]:, :], which is only correct for stride 1.
            self.n_positions = self.n_window - self.inner_window + 1
            seq_len = self.inner_window
            self.register_buffer(
                "target_positions",
                torch.arange(self.inner_window - 1, self.n_window, dtype=torch.long),
                persistent=False,
            )

        self.pos_encoder = PositionalEncoding(d_model, dropout, seq_len)
        # ADAPTED: explicit layer stacks instead of nn.TransformerEncoder/Decoder
        # (identical for the original num_layers=1, no version-dependent fast path).
        self.transformer_encoder = nn.ModuleList(
            [TranADEncoderLayer(d_model, self.nhead, self.dim_feedforward, dropout) for _ in range(self.num_layers)]
        )
        self.transformer_decoder1 = nn.ModuleList(
            [TranADDecoderLayer(d_model, self.nhead, self.dim_feedforward, dropout) for _ in range(self.num_layers)]
        )
        self.transformer_decoder2 = nn.ModuleList(
            [TranADDecoderLayer(d_model, self.nhead, self.dim_feedforward, dropout) for _ in range(self.num_layers)]
        )
        # ADAPTED: the original head is Linear + Sigmoid (min-max scaled inputs).
        # This project's features are z-scored, so identity is the default.
        head: list[nn.Module] = [nn.Linear(d_model, self.n_feats)]
        if self.output_activation_name == "sigmoid":
            head.append(nn.Sigmoid())
        elif self.output_activation_name != "identity":
            raise ValueError("output_activation must be 'identity' or 'sigmoid'.")
        self.fcn = nn.Sequential(*head)

    def _encode(self, src: torch.Tensor, c: torch.Tensor, tgt: torch.Tensor):
        src = torch.cat((src, c), dim=2)
        src = src * math.sqrt(self.n_feats)
        src = self.pos_encoder(src)
        memory = src
        for layer in self.transformer_encoder:
            memory = layer(memory)
        tgt = tgt.repeat(1, 1, 2)
        return tgt, memory

    def _decode(self, stack: nn.ModuleList, tgt: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        out = tgt
        for layer in stack:
            out = layer(out, memory)
        return out

    def _core_phases(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Original two-phase pass over one window; returns (batch, 1, channels) twice."""
        src = x.transpose(0, 1)  # ADAPTED: (B, T, C) -> sequence-first (T, B, C)
        tgt = src[-1:, :, :]

        c = torch.zeros_like(src)
        x1 = self.fcn(self._decode(self.transformer_decoder1, *self._encode(src, c, tgt)))
        c = (x1 - src) ** 2
        x2 = self.fcn(self._decode(self.transformer_decoder2, *self._encode(src, c, tgt)))
        return x1.transpose(0, 1), x2.transpose(0, 1)

    def _sliding_phases(self, x: torch.Tensor, positions: torch.Tensor | None):
        """Run the core on every sub-window; returns (batch, n_selected, channels) twice."""
        batch = x.shape[0]
        # unfold() is a view: selecting positions first avoids materializing all of them.
        sub = x.unfold(1, self.inner_window, 1)  # (B, P, C, w)
        if positions is not None:
            sub = sub.index_select(1, positions)
        n_selected = sub.shape[1]
        sub = sub.permute(0, 1, 3, 2).reshape(batch * n_selected, self.inner_window, self.n_feats)

        chunk = self.eval_position_chunk
        if torch.is_grad_enabled() or chunk <= 0:
            x1, x2 = self._core_phases(sub)
        else:
            # Inference only: bound peak memory, the sub-window axis is ~111x the batch.
            outs1, outs2 = [], []
            step = max(1, chunk * batch)
            for start in range(0, sub.shape[0], step):
                part1, part2 = self._core_phases(sub[start : start + step])
                outs1.append(part1)
                outs2.append(part2)
            x1, x2 = torch.cat(outs1, dim=0), torch.cat(outs2, dim=0)

        x1 = x1.reshape(batch, n_selected, self.n_feats)
        x2 = x2.reshape(batch, n_selected, self.n_feats)
        return x1, x2

    def forward_phases(self, x: torch.Tensor, positions: torch.Tensor | None = None):
        """Return both phase outputs.

        Shape is (batch, 1, channels) in the default mode and
        (batch, n_positions, channels) in the author-faithful sliding mode.
        """
        validate_3d_input("TranAD", x, self.input_shape)
        if self.inner_window is None:
            return self._core_phases(x)
        return self._sliding_phases(x, positions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Phase-2 prediction of every predicted frame.

        Both phases are computed, so latency benchmarking measures the full model.
        In sliding mode the output covers the last `n_positions` frames of the
        window, which is what the unified rescoring reducer expects.
        """
        _, x2 = self.forward_phases(x)
        return x2

    def _targets(self, y: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        """The frames the model is asked to predict."""
        if self.inner_window is None:
            return y[:, -1:, :]
        index = self.target_positions if positions is None else self.target_positions.index_select(0, positions)
        return y.index_select(1, index)

    def _sample_positions(self, device: torch.device) -> torch.Tensor | None:
        """Optional speed knob: train on a random subset of sub-window positions."""
        if self.inner_window is None or not self.train_positions:
            return None
        if self.train_positions >= self.n_positions:
            return None
        return torch.randperm(self.n_positions, device=device)[: self.train_positions]

    def loss_phases(self, x: torch.Tensor, y: torch.Tensor, criterion, ctx: dict[str, Any]):
        """Original objective: (1/n)*MSE(O1, W) + (1-1/n)*MSE(O2, W), n = epoch."""
        n = max(1, int(ctx["epoch"]))
        positions = self._sample_positions(x.device)
        x1, x2 = self.forward_phases(x, positions)
        target = self._targets(y, positions)
        loss = (1.0 / n) * criterion(x1, target) + (1.0 - 1.0 / n) * criterion(x2, target)
        return [(loss, None)]

    @torch.no_grad()
    def anomaly_error(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Author score, kept per-channel: 0.5*(O1-W)^2 + 0.5*(O2-W)^2 -> (batch, channels).

        In sliding mode every predicted frame contributes its own error and the
        frame axis is aggregated (mean by default), so the score for one
        evaluation window rests on `n_positions` predictions instead of one.
        """
        x1, x2 = self.forward_phases(x)
        target = self._targets(y)
        errors = 0.5 * (x1 - target) ** 2 + 0.5 * (x2 - target) ** 2
        if self.inner_window is None:
            return errors.squeeze(1)
        if self.score_aggregation == "max":
            return errors.amax(dim=1)
        return errors.mean(dim=1)
