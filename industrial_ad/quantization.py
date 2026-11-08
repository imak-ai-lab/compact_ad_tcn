"""Post-training quantization recipes and evaluation for trained detectors."""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any, Sequence

import torch
import tqdm
from torch import nn

from industrial_ad.analysis import benchmark_module
from industrial_ad.config import validate_quantization_config
from industrial_ad.datasets.PU.dataset import build_pu_dataloaders
from industrial_ad.experiments import load_detector_from_run
from industrial_ad.training import _close_figures, _save_figures
from industrial_ad.utils import (
    clone_config,
    dump_json,
    load_json,
    parameter_size_bytes,
    seed_everything,
)

_DYNAMIC_MODULES = {nn.Linear, nn.GRU}
_STATIC_MODULES = (nn.Conv1d, nn.Linear)
_STATIC_MODEL_NAMES = {"conv_ae", "tcn_ae", "tcn_forecaster", "tcn_light_forecaster"}
_DYNAMIC_MODEL_NAMES = {
    "window_mlp",
    "mlp_ae",
    "mlp_forecaster",
    "gru_repeated_ae",
    "gru_seq2seq_ae",
    "gru_seq2seq_forecaster",
    "transformer_ae",
}


class _StaticQuantizedLeaf(nn.Module):
    """Quantize one supported leaf while keeping surrounding custom code in FP32."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.module = module
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dequant(self.module(self.quant(x)))


def _is_static_model(model_name: str) -> bool:
    name = model_name.lower()
    if name in _STATIC_MODEL_NAMES:
        return True
    if name in _DYNAMIC_MODEL_NAMES:
        return False
    raise ValueError(f"Quantization is not defined for model: {model_name}")


def _qint_dtype(name: str) -> torch.dtype:
    if name != "qint8":
        raise ValueError(f"Unsupported quantization dtype: {name}")
    return torch.qint8


def _disable_transformer_fastpath(module: nn.Module, inputs) -> None:
    return None


def _prepare_dynamic_model(model: nn.Module, model_name: str) -> nn.Module:
    if model_name.lower() == "transformer_ae":
        for module in model.modules():
            if isinstance(module, nn.TransformerEncoderLayer):
                module.register_forward_pre_hook(_disable_transformer_fastpath)
    return model


def _wrap_static_modules(module: nn.Module) -> nn.Module:
    for name, child in list(module.named_children()):
        if isinstance(child, _STATIC_MODULES):
            setattr(module, name, _StaticQuantizedLeaf(child))
        else:
            _wrap_static_modules(child)
    return module


@torch.no_grad()
def _calibrate(model: nn.Module, dataloader, max_batches: int | None) -> None:
    model.eval()
    for batch_index, (x, _, _) in enumerate(dataloader):
        if max_batches is not None and batch_index >= max_batches:
            break
        model(x.float())


def apply_model_quantization(
    model: nn.Module,
    model_name: str,
    quantization_config: dict[str, Any],
    calibration_loader=None,
) -> nn.Module:
    """Apply the architecture-defined PTQ recipe."""
    model = model.cpu().eval()
    dtype = _qint_dtype(str(quantization_config["dtype"]))
    torch.backends.quantized.engine = str(quantization_config["backend"])

    if not _is_static_model(model_name):
        model = torch.ao.quantization.quantize_dynamic(model, _DYNAMIC_MODULES, dtype=dtype, inplace=False)
        return _prepare_dynamic_model(model, model_name)

    model = _wrap_static_modules(model)
    qconfig = torch.ao.quantization.get_default_qconfig(str(quantization_config["backend"]))
    for module in model.modules():
        if isinstance(module, _StaticQuantizedLeaf):
            module.qconfig = qconfig
    torch.ao.quantization.prepare(model, inplace=True)
    if calibration_loader is not None:
        _calibrate(model, calibration_loader, quantization_config["calibration_batches"])
    torch.ao.quantization.convert(model, inplace=True)
    return model


def _source_config(config: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    source_dir = Path(config["source"]["run_dir"])
    return source_dir, load_json(source_dir / "config.json")


def _source_data_bundle_key(source_config: dict[str, Any]) -> dict[str, Any]:
    """Return the source config subset that controls PU dataloader reuse."""
    return {
        "dataset": source_config["dataset"],
        "debug.dataset": source_config["debug"]["dataset"],
        "task": source_config["task"],
    }


def _make_config_snapshot(config: dict[str, Any], source_config: dict[str, Any]) -> dict[str, Any]:
    return {
        "run": clone_config(config["run"]),
        "source": clone_config(config["source"]),
        "task": clone_config(source_config["task"]),
        "dataset": clone_config(source_config["dataset"]),
        "debug": {"dataset": clone_config(source_config["debug"]["dataset"])},
        "model": clone_config(source_config["model"]),
        "scoring": clone_config(source_config["scoring"]),
        "runtime": clone_config(source_config["runtime"]),
        "evaluation": clone_config(config["evaluation"]),
        "quantization": clone_config(config["quantization"]),
        "benchmark": clone_config(config["benchmark"]),
    }


def _build_summary(
    run_dir: Path,
    config: dict[str, Any],
    detector: nn.Module,
    metrics: dict[str, Any],
    quantization_time: float,
    evaluation_time: float,
) -> dict[str, Any]:
    metric_name = config["evaluation"]["metric"]
    source_dir = Path(config["source"]["run_dir"])
    return {
        "best_metric_name": metric_name,
        "best_metric_value": metrics[metric_name],
        "best_metrics": metrics,
        "family": run_dir.parent.name,
        "run_name": config["run"]["name"],
        "run_dir": str(run_dir),
        "source_family": source_dir.parent.name,
        "source_run_name": source_dir.name,
        "model_name": config["model"]["name"],
        "task_type": config["task"]["type"],
        "input_shape": config["runtime"]["input_shape"],
        "target_shape": config["runtime"]["target_shape"],
        "score_feature_dim": config["runtime"]["score_feature_dim"],
        "parameter_size_bytes": parameter_size_bytes(detector),
        "quantization_time_seconds": quantization_time,
        "evaluation_time_seconds": evaluation_time,
    }


def run_quantization(
    config: dict[str, Any],
    *,
    overwrite: bool = False,
    skip_existing: bool = False,
    data_bundle: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Quantize one trained detector according to an explicit config."""
    validate_quantization_config(config)
    run_dir = Path(config["run"]["dir"])
    summary_path = run_dir / "summary.json"
    if summary_path.exists() and skip_existing:
        return load_json(summary_path), data_bundle
    if run_dir.exists() and overwrite:
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)
    print("Quantizing run:", Path(config["run"]["dir"]).parent.name, config["run"]["name"])

    source_dir, source_config = _source_config(config)
    config_snapshot = _make_config_snapshot(config, source_config)
    dump_json(run_dir / "config.json", config_snapshot)
    seed_everything(int(config_snapshot["run"]["seed"]))

    if config_snapshot["dataset"]["name"].lower() != "pu":
        raise ValueError(f"Unknown dataset: {config_snapshot['dataset']['name']}")
    if data_bundle is None:
        data_bundle = build_pu_dataloaders(config_snapshot)

    detector, _ = load_detector_from_run(source_dir, checkpoint=config_snapshot["source"]["checkpoint"])
    model_name = config_snapshot["model"]["name"]
    start = time.time()
    detector.model = apply_model_quantization(
        detector.model,
        model_name,
        config_snapshot["quantization"],
        calibration_loader=data_bundle["loaders"]["train"] if _is_static_model(model_name) else None,
    )
    quantization_time = time.time() - start

    start = time.time()
    max_batches = config_snapshot["evaluation"].get("max_batches")
    detector.fit_score_estimator(data_bundle["loaders"]["val"], max_batches=max_batches)
    detector.fit_threshold(data_bundle["loaders"]["val"], max_batches=max_batches)
    val_metrics, val_figures = detector.evaluate(data_bundle["loaders"]["val"], prefix="val", max_batches=max_batches)
    test_metrics, test_figures = detector.evaluate(data_bundle["loaders"]["test"], prefix="test", max_batches=max_batches)
    evaluation_time = time.time() - start

    _save_figures(val_figures, run_dir / "plots", "val", 0)
    _save_figures(test_figures, run_dir / "plots", "test", 0)
    _close_figures(val_figures)
    _close_figures(test_figures)

    summary = _build_summary(run_dir, config_snapshot, detector, {**val_metrics, **test_metrics}, quantization_time, evaluation_time)
    torch.save(detector.state_dict(), run_dir / "checkpoints" / "best.pt")

    if config_snapshot["benchmark"].get("enabled", False):
        benchmark_config = config_snapshot["benchmark"]
        summary["benchmark"] = benchmark_module(
            detector.model,
            input_shape=(1, *config_snapshot["runtime"]["input_shape"]),
            device=benchmark_config["device"],
            num_threads=int(benchmark_config["num_threads"]),
            warmup_runs=int(benchmark_config["warmup_runs"]),
            num_runs=int(benchmark_config["num_runs"]),
            profile_memory=bool(benchmark_config["profile_memory"]),
        )

    dump_json(summary_path, summary)
    return summary, data_bundle


def run_quantizations(
    configs: Sequence[dict[str, Any]],
    *,
    overwrite: bool = False,
    skip_existing: bool = True,
    share_data_bundle: bool = True,
    stop_on_error: bool = True,
) -> list[dict[str, Any]]:
    """Run a list of explicit quantization configs one by one."""
    results = []
    data_bundle, data_bundle_config = None, None
    for config in tqdm.tqdm(configs, desc="quantization"):
        try:
            _, source_config = _source_config(config)
            new_data_bundle_config = _source_data_bundle_key(source_config)
            if data_bundle_config != new_data_bundle_config:
                data_bundle, data_bundle_config = None, new_data_bundle_config
            summary, new_data_bundle = run_quantization(
                config,
                overwrite=overwrite,
                skip_existing=skip_existing,
                data_bundle=data_bundle,
            )
            if share_data_bundle:
                data_bundle = new_data_bundle
            results.append(summary)
        except Exception as exc:
            if stop_on_error:
                raise
            results.append(
                {
                    "run_name": config["run"]["name"],
                    "run_dir": config["run"]["dir"],
                    "error": True,
                    "exception": repr(exc),
                }
            )
    return results
