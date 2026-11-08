"""PCA reconstruction baseline wrapped as a torch module."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import torch
from sklearn.decomposition import PCA
from torch import nn

from .utils import as_shape, validate_window_shapes


class PCAReconstructionModel(nn.Module):
    """Linear reconstruction baseline that projects windows onto principal components."""

    def __init__(
        self,
        input_shape: Iterable[int],
        output_shape: Iterable[int],
        n_components: int,
        svd_solver: str = "auto",
    ) -> None:
        super().__init__()
        self.input_shape = as_shape(input_shape)
        self.output_shape = as_shape(output_shape)
        validate_window_shapes("PCAReconstructionModel", self.input_shape, self.output_shape, same_shape=True)

        self.n_components = int(n_components)
        if self.n_components <= 0:
            raise ValueError("PCAReconstructionModel requires n_components > 0.")

        input_dim = math.prod(self.input_shape)
        if self.n_components > input_dim:
            raise ValueError("n_components must not exceed the flattened input dimension.")

        self.svd_solver = str(svd_solver)
        self.mean = nn.Parameter(torch.zeros(input_dim, dtype=torch.float32), requires_grad=False)
        self.components = nn.Parameter(torch.zeros(self.n_components, input_dim, dtype=torch.float32), requires_grad=False)

    @torch.no_grad()
    def fit(self, flat_windows: torch.Tensor, *, seed: int = 0) -> None:
        """Fit PCA on flattened normal windows and store the resulting basis in torch tensors."""
        flat_windows = flat_windows.detach().cpu().float()
        pca = PCA(
            n_components=self.n_components,
            svd_solver=self.svd_solver,
            random_state=int(seed),
        )
        pca.fit(flat_windows.numpy())
        self.mean.copy_(torch.from_numpy(np.asarray(pca.mean_, dtype=np.float32)))
        self.components.copy_(torch.from_numpy(np.asarray(pca.components_, dtype=np.float32)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        flattened = x.reshape(batch_size, -1)
        mean = self.mean
        components = self.components
        if mean.dtype != flattened.dtype:
            mean = mean.to(flattened.dtype)
            components = components.to(flattened.dtype)

        centered = flattened - mean
        latent = centered @ components.T
        reconstruction = latent @ components + mean
        return reconstruction.reshape(batch_size, *self.output_shape)
