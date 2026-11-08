"""USAD (Audibert et al., KDD 2020) — port of the official implementation.

Source: https://github.com/manigalati/usad (BSD 3-Clause, Copyright (c) 2020 Julien Audibert et al.), `usad.py` + `USAD.ipynb`.
Encoder/decoder shapes, the two-phase adversarial objective and the test-time
score follow the original code. Interface adaptations are marked `ADAPTED:`.
"""

from __future__ import annotations

from typing import Any, Iterable

import torch
from torch import nn

from industrial_ad.models.utils import as_shape, validate_3d_input, validate_window_shapes


class _Encoder(nn.Module):
    def __init__(self, in_size: int, latent_size: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_size, int(in_size / 2))
        self.linear2 = nn.Linear(int(in_size / 2), int(in_size / 4))
        self.linear3 = nn.Linear(int(in_size / 4), latent_size)
        self.relu = nn.ReLU(True)

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.linear1(w))
        out = self.relu(self.linear2(out))
        return self.relu(self.linear3(out))


class _Decoder(nn.Module):
    def __init__(self, latent_size: int, out_size: int, output_activation: str) -> None:
        super().__init__()
        self.linear1 = nn.Linear(latent_size, int(out_size / 4))
        self.linear2 = nn.Linear(int(out_size / 4), int(out_size / 2))
        self.linear3 = nn.Linear(int(out_size / 2), out_size)
        self.relu = nn.ReLU(True)
        # ADAPTED: the original decoder ends in Sigmoid (min-max scaled inputs);
        # this project's features are z-scored, so identity is the default.
        self.output_activation = nn.Sigmoid() if output_activation == "sigmoid" else nn.Identity()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.linear1(z))
        out = self.relu(self.linear2(out))
        return self.output_activation(self.linear3(out))


class USAD(nn.Module):
    """USAD: one encoder, two decoders, two-phase adversarial training.

    The window is flattened to `time_steps * channels`, as in the original code.
    """

    def __init__(
        self,
        input_shape: Iterable[int],
        output_shape: Iterable[int],
        latent_size: int | None = None,
        hidden_size: int = 100,
        alpha: float = 0.5,
        beta: float = 0.5,
        output_activation: str = "identity",
        input_range: list[list[float]] | None = None,
    ) -> None:
        super().__init__()
        self.input_shape = as_shape(input_shape)
        self.output_shape = as_shape(output_shape)
        validate_window_shapes("USAD", self.input_shape, self.output_shape, same_shape=True)

        self.time_steps, self.channels = self.input_shape
        self.w_size = self.time_steps * self.channels
        # Original notebook: z_size = window_size * hidden_size (hidden_size=100).
        self.z_size = int(latent_size) if latent_size is not None else self.time_steps * int(hidden_size)
        if self.z_size <= 0:
            raise ValueError("latent_size must be positive.")
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.output_activation_name = str(output_activation).lower()
        if self.output_activation_name not in {"identity", "sigmoid"}:
            raise ValueError("output_activation must be 'identity' or 'sigmoid'.")

        # ADAPTED: the phase-2 objective *maximizes* reconstruction error, which is
        # bounded only because the original decoder ends in a sigmoid and the
        # original inputs are min-max scaled to [0, 1]. On raw z-scored features
        # with an identity head the term is unbounded and training diverges
        # (observed: loss -> 1e14). Passing `input_range` restores the author's
        # setup: per-channel min/max from the training split, scaling done inside
        # the model, sigmoid head.
        self.has_input_scaler = input_range is not None
        if self.has_input_scaler:
            minimum = torch.tensor(input_range[0], dtype=torch.float32)
            maximum = torch.tensor(input_range[1], dtype=torch.float32)
            if minimum.shape != (self.channels,) or maximum.shape != (self.channels,):
                raise ValueError(f"input_range must be [[min x {self.channels}], [max x {self.channels}]].")
            self.register_buffer("input_min", minimum)
            self.register_buffer("input_span", (maximum - minimum).clamp_min(1e-6))

        self.encoder = _Encoder(self.w_size, self.z_size)
        self.decoder1 = _Decoder(self.z_size, self.w_size, self.output_activation_name)
        self.decoder2 = _Decoder(self.z_size, self.w_size, self.output_activation_name)

    def _scale(self, x: torch.Tensor) -> torch.Tensor:
        """Map inputs into the [0, 1]-ish range the author's objective assumes."""
        if not self.has_input_scaler:
            return x
        return (x - self.input_min) / self.input_span

    def _unscale(self, x: torch.Tensor) -> torch.Tensor:
        if not self.has_input_scaler:
            return x
        return x * self.input_span + self.input_min

    def _flatten(self, x: torch.Tensor) -> torch.Tensor:
        return self._scale(x).reshape(x.shape[0], -1)

    def _unflatten(self, w: torch.Tensor) -> torch.Tensor:
        return w.reshape(w.shape[0], self.time_steps, self.channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """AE1 reconstruction, shaped like the input (used for latency and shape probes)."""
        validate_3d_input("USAD", x, self.input_shape)
        w = self._flatten(x)
        # Returned in the original feature space, so shape probes and latency
        # benchmarks see the same units as every other model family.
        return self._unscale(self._unflatten(self.decoder1(self.encoder(w))))

    def _phase_outputs(self, w: torch.Tensor):
        z = self.encoder(w)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        w3 = self.decoder2(self.encoder(w1))
        return w1, w2, w3

    def loss_phases(self, x: torch.Tensor, y: torch.Tensor, criterion, ctx: dict[str, Any]):
        """Original two-phase objective; n = epoch, as in `training()` (`n = epoch+1`, 1-based).

        This is a generator on purpose: the original calls `training_step` again
        after the AE1 update, so the AE2 graph must be built from the already
        updated parameters. Yielding lazily gives exactly that ordering.
        """
        n = max(1, int(ctx["epoch"]))
        window = self._flatten(x)
        target = self._flatten(y)

        # Phase 1 — train AE1 (encoder + decoder1).
        w1, _, w3 = self._phase_outputs(window)
        loss1 = (1 / n) * torch.mean((target - w1) ** 2) + (1 - 1 / n) * torch.mean((target - w3) ** 2)
        yield loss1, list(self.encoder.parameters()) + list(self.decoder1.parameters())

        # Phase 2 — train AE2 (encoder + decoder2), on a freshly built graph.
        _, w2, w3 = self._phase_outputs(window)
        loss2 = (1 / n) * torch.mean((target - w2) ** 2) - (1 - 1 / n) * torch.mean((target - w3) ** 2)
        yield loss2, list(self.encoder.parameters()) + list(self.decoder2.parameters())

    @torch.no_grad()
    def anomaly_error(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Author score `alpha*||W-W1||^2 + beta*||W-W2||^2`, kept per-channel.

        The original averages over the whole flattened window; here the average is
        taken over time only, so the configured score estimator sees a
        (batch, channels) feature vector. With the "mean" estimator the scalar
        score is identical to the original.
        """
        w = self._flatten(x)
        target = self._flatten(y)
        z = self.encoder(w)
        w1 = self.decoder1(z)
        w2 = self.decoder2(self.encoder(w1))
        errors = self.alpha * (target - w1) ** 2 + self.beta * (target - w2) ** 2
        # Errors stay in the model's own (scaled) space — that is the author's score.
        return errors.reshape(errors.shape[0], self.time_steps, self.channels).mean(dim=1)
