"""Small helpers shared by model definitions."""

from typing import Iterable, Sequence

import torch
from torch import nn


def build_activation(name: str) -> nn.Module:
    activations = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
    }
    if name.lower() not in activations:
        raise ValueError(f"Unsupported activation: {name}")
    return activations[name.lower()]()


def as_shape(values: Iterable[int]) -> tuple[int, ...]:
    """Convert a user-provided shape into an immutable integer tuple."""
    return tuple(int(value) for value in values)


def validate_window_shapes(
    name: str,
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    *,
    same_shape: bool = False,
    same_channels: bool = False,
) -> None:
    """Validate the common `(time_steps, channels)` shape contract."""
    if len(input_shape) != 2 or len(output_shape) != 2:
        raise ValueError(f"{name} expects shapes in the form (time_steps, channels).")
    if same_shape and input_shape != output_shape:
        raise ValueError(f"{name} requires input_shape and output_shape to match.")
    if same_channels and input_shape[1] != output_shape[1]:
        raise ValueError(f"{name} requires the same number of input and output channels.")


def validate_3d_input(name: str, x: torch.Tensor, input_shape: tuple[int, ...]) -> None:
    """Validate batched `(time_steps, channels)` inputs before a forward pass."""
    if x.ndim != 3:
        raise ValueError(f"{name} expects a 3D tensor, got shape {tuple(x.shape)}.")
    if x.shape[1:] != input_shape:
        raise ValueError(f"Expected input shape (*, {input_shape[0]}, {input_shape[1]}), got {tuple(x.shape)}.")


def make_dilations(num_blocks: int, dilations: Sequence[int] | None) -> list[int]:
    """Return the explicit TCN dilation schedule."""
    if dilations is None:
        return [2**index for index in range(num_blocks)]
    values = [int(value) for value in dilations]
    if len(values) != num_blocks:
        raise ValueError("len(dilations) must match num_blocks.")
    return values


def recurrent_dropout(dropout: float, num_layers: int) -> float:
    """PyTorch recurrent dropout is only active when a GRU has more than one layer."""
    return float(dropout) if int(num_layers) > 1 else 0.0
