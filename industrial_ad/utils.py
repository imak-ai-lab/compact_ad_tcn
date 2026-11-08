"""Shared helpers for configs, JSON artifacts, seeding and model size estimates."""

from __future__ import annotations

import copy
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def clone_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a deep copy of an experiment config."""
    return copy.deepcopy(config)


def flatten_dict(mapping: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a nested dictionary for dataframe-friendly summaries."""
    items: dict[str, Any] = {}
    for key, value in mapping.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            items.update(flatten_dict(value, full_key))
        else:
            items[full_key] = value
    return items


def ensure_json_serializable(value: Any) -> Any:
    """Convert numpy and torch values into JSON-friendly Python objects."""
    if isinstance(value, dict):
        return {str(key): ensure_json_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [ensure_json_serializable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Size):
        return list(value)
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def dump_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write a JSON artifact next to experiment outputs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(ensure_json_serializable(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON artifact produced by the experiment code."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch for reproducible smoke runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(module: torch.nn.Module) -> int:
    """Count trainable and non-trainable parameters in a module."""
    return sum(parameter.numel() for parameter in module.parameters())


def _tensor_key(tensor: torch.Tensor) -> tuple[Any, ...]:
    return (tensor.device.type, tensor.device.index, tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype)


def _tensor_size_bytes(tensor: torch.Tensor, seen: set[tuple[Any, ...]]) -> int:
    """Return raw tensor payload size, including qparams for quantized tensors."""
    key = _tensor_key(tensor)
    if key in seen:
        return 0
    seen.add(key)

    size = tensor.numel() * tensor.element_size()
    if tensor.is_quantized:
        qscheme = tensor.qscheme()
        if qscheme in {torch.per_channel_affine, torch.per_channel_symmetric}:
            size += _tensor_size_bytes(tensor.q_per_channel_scales(), seen)
            size += _tensor_size_bytes(tensor.q_per_channel_zero_points(), seen)
        else:
            size += 16
    return int(size)


def _tensor_tree_size_bytes(value: Any, seen: set[tuple[Any, ...]]) -> int:
    if isinstance(value, torch.Tensor):
        return _tensor_size_bytes(value, seen)
    if isinstance(value, torch.ScriptObject):
        return _tensor_tree_size_bytes(value.__getstate__(), seen)
    if isinstance(value, dict):
        return sum(_tensor_tree_size_bytes(item, seen) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_tree_size_bytes(item, seen) for item in value)
    return 0


def tensor_tree_size_bytes(value: Any) -> int:
    """Recursively count tensor bytes inside nested checkpoint/state-dict values."""
    return _tensor_tree_size_bytes(value, set())


def state_dict_size_bytes(module: torch.nn.Module) -> int:
    """Estimate the raw tensor footprint of a module state_dict."""
    return tensor_tree_size_bytes(module.state_dict())


def parameter_size_bytes(module: torch.nn.Module) -> int:
    """Estimate saved model footprint in bytes, including quantized packed weights."""
    size = state_dict_size_bytes(module)
    if size > 0:
        return size
    return sum(parameter.numel() * parameter.element_size() for parameter in module.parameters())
