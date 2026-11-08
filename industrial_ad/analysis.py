"""Utilities for loading runs, benchmarking models and plotting diagnostics."""

from __future__ import annotations

import copy
import time
import tqdm
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D
from torch.profiler import ProfilerActivity, profile

from industrial_ad.experiments import load_detector_from_run
from industrial_ad.utils import dump_json, flatten_dict, load_json


def benchmark_module(
    module: torch.nn.Module,
    input_shape: Sequence[int],
    *,
    device: str,
    num_threads: int,
    warmup_runs: int,
    num_runs: int,
    profile_memory: bool,
) -> dict[str, Any]:
    """Benchmark a module with a synthetic input tensor of the requested shape."""
    module = module.to(device)
    module.eval()
    old_threads = torch.get_num_threads()
    torch.set_num_threads(int(num_threads))
    dummy_input = torch.randn(tuple(int(value) for value in input_shape), device=device)

    with torch.no_grad():
        for _ in range(int(warmup_runs)):
            _ = module(dummy_input)

    peak_memory_bytes = None
    if profile_memory:
        with profile(
            activities=[ProfilerActivity.CPU],
            profile_memory=True,
            record_shapes=False,
        ) as profiler:
            with torch.no_grad():
                _ = module(dummy_input.cpu())
        peak_memory_bytes = max(event.cpu_memory_usage for event in profiler.events())

    latencies = []
    with torch.no_grad():
        for _ in range(int(num_runs)):
            start = time.perf_counter()
            _ = module(dummy_input)
            latencies.append(time.perf_counter() - start)

    torch.set_num_threads(old_threads)
    return {
        "latency_mean_seconds": float(np.mean(latencies)),
        "latency_std_seconds": float(np.std(latencies)),
        "peak_memory_bytes": int(peak_memory_bytes) if peak_memory_bytes is not None else None,
        "benchmark_device": device,
        "benchmark_threads": int(num_threads),
        "benchmark_runs": int(num_runs),
    }


def benchmark_run(
    run_dir: str | Path,
    *,
    checkpoint: str = "best",
    benchmark_config: dict[str, Any] | None = None,
    update_summary: bool = True,
    skip_existing: bool = True,
) -> dict[str, Any]:
    """Benchmark the saved model checkpoint for one run directory."""
    run_dir = Path(run_dir)

    if skip_existing:
        summary = load_json(run_dir / "summary.json")
        if "benchmark" in summary:
            return summary["benchmark"]
    print(f"Benchmarking {run_dir}")

    detector, config = load_detector_from_run(run_dir, checkpoint=checkpoint)
    benchmark_config = copy.deepcopy(benchmark_config or config["benchmark"])

    result = benchmark_module(
        detector.model,
        input_shape=(1, *config["runtime"]["input_shape"]),
        device=benchmark_config["device"],
        num_threads=int(benchmark_config["num_threads"]),
        warmup_runs=int(benchmark_config["warmup_runs"]),
        num_runs=int(benchmark_config["num_runs"]),
        profile_memory=bool(benchmark_config["profile_memory"]),
    )

    if update_summary:
        summary = load_json(run_dir / "summary.json")
        summary["benchmark"] = result
        dump_json(run_dir / "summary.json", summary)
    return result


def discover_run_dirs(*roots: str | Path) -> list[Path]:
    """Recursively find run directories under `root` by the presence of `summary.json`."""
    res = []

    for root in roots:
        root = Path(root)
        res += sorted(summary_path.parent for summary_path in root.rglob("summary.json"))

    return res


def load_run_summaries(run_dirs: Sequence[str | Path]) -> list[dict[str, Any]]:
    """Load summaries for an explicit list of run directories."""
    summaries = []
    for run_dir in run_dirs:
        summary_path = Path(run_dir) / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing summary.json in run directory: {run_dir}")
        summary = load_json(summary_path)
        config_path = summary_path.parent / "config.json"
        if config_path.exists():
            summary["config"] = load_json(config_path)
        summaries.append(summary)
    return summaries


def load_run_dataframe(run_dirs: Sequence[str | Path], *, include_config: bool = True) -> pd.DataFrame:
    """Load an explicit run list into one flat dataframe."""
    rows = []
    for summary in load_run_summaries(run_dirs):
        row = flatten_dict({key: value for key, value in summary.items() if key != "config"})
        if include_config and "config" in summary:
            row.update({f"config.{key}": value for key, value in flatten_dict(summary["config"]).items()})
        rows.append(row)
    return pd.DataFrame(rows)


def benchmark_runs(
    run_dirs: Sequence[str | Path],
    *,
    checkpoint: str = "best",
    benchmark_config: dict[str, Any] | None = None,
    update_summary: bool = True,
    skip_existing: bool = True,
) -> list[dict[str, Any]]:
    """Benchmark an explicit list of run directories and optionally update their summaries."""
    results = []
    for run_dir in tqdm.tqdm(run_dirs):
        metrics = benchmark_run(
            run_dir,
            checkpoint=checkpoint,
            benchmark_config=benchmark_config,
            update_summary=update_summary,
            skip_existing=skip_existing
        )
        metrics["run_dir"] = str(run_dir)
        results.append(metrics)
    return results


def pareto_mask(values: np.ndarray, maximize: Sequence[bool], eps: Sequence[float] | float = 1e-9) -> np.ndarray:
    """Return a boolean mask for Pareto-optimal rows."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("values must be a 2D array")
    if len(maximize) != values.shape[1]:
        raise ValueError("maximize must have the same length as the number of columns in values")

    eps_array = np.asarray(eps, dtype=float)
    if eps_array.ndim == 0:
        eps_array = np.full(values.shape[1], float(eps_array))

    signed = values * np.array([1.0 if flag else -1.0 for flag in maximize], dtype=float)
    mask = np.ones(values.shape[0], dtype=bool)

    for index in range(values.shape[0]):
        if not mask[index]:
            continue
        greater_equal = np.all(signed >= (signed[index] - eps_array), axis=1)
        strictly_greater = np.any(signed > (signed[index] + eps_array), axis=1)
        dominates = greater_equal & strictly_greater
        dominates[index] = False
        if np.any(dominates):
            mask[index] = False
    return mask


def mark_pareto_front(
    df: pd.DataFrame,
    columns: Sequence[str],
    maximize: Sequence[bool],
    *,
    eps: Sequence[float] | float = 1e-9,
    output_column: str = "pareto",
) -> pd.DataFrame:
    """Annotate a dataframe with a Pareto-front membership column."""
    marked = df.copy()
    marked[output_column] = pareto_mask(marked[list(columns)].values, maximize=maximize, eps=eps)
    return marked


def plot_tradeoff_scatter(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    pareto_column: str | None = None,
    label_column: str | None = None,
    family_column: str | None = None,
    log_x: bool = False,
    log_y: bool = False,
    figsize: tuple[int, int] = (8, 5),
    title: str | None = None,
    save_path: str | Path | None = None,
):
    """Plot a metric/resource trade-off highlighting Pareto-optimal models."""
    fig, ax = plt.subplots(figsize=figsize)
    families = [None] if family_column is None else list(pd.unique(df[family_column]))
    cmap = plt.get_cmap("tab10")
    legend_handles = []

    for i, family in enumerate(families):
        family_df = df if family_column is None else df[df[family_column] == family]
        is_pareto = family_df[pareto_column].astype(bool) if pareto_column is not None else pd.Series([False] * len(family_df), index=family_df.index)
        others = family_df[~is_pareto]
        pareto = family_df[is_pareto]
        color = cmap(i)

        if not others.empty:
            ax.scatter(others[x], others[y], s=32, alpha=0.45, color=color)
        if not pareto.empty:
            ax.scatter(
                pareto[x],
                pareto[y],
                s=140,
                alpha=0.95,
                color=color,
                marker="X",
                edgecolors="black",
                linewidths=0.8,
            )
            if label_column is not None:
                for _, row in pareto.iterrows():
                    ax.annotate(str(row[label_column]), (row[x], row[y]), xytext=(5, 5), textcoords="offset points")

        if family_column is not None:
            legend_handles.append(
                Line2D(
                    [], [], marker="o", linestyle="", 
                    color=color, markerfacecolor=color, markeredgecolor=color,
                    markersize=6, alpha=1,
                    label=str(family)
                )
            )

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title or f"{y} vs {x}")
    ax.grid(True, linestyle=":", linewidth=0.6)
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    if legend_handles:
        ax.legend(handles=legend_handles)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax


def plot_metric_history(
    run_dirs: Sequence[str | Path],
    *,
    metric: str,
    title: str | None = None,
    figsize: tuple[int, int] = (8, 5),
    labels: Sequence[str] | None = None,
    save_path: str | Path | None = None,
    yscale: str = "linear",
    color_group_size: int = 1
):
    """Plot the history of one metric for several run directories."""
    fig, ax = plt.subplots(figsize=figsize)

    if labels is not None and len(labels) != len(run_dirs):
        raise ValueError("labels length must match run_dirs length")

    for index, run_dir in enumerate(run_dirs):
        run_dir = Path(run_dir)
        history = load_json(run_dir / "history.json")
        label = labels[index] if labels is not None else run_dir.name

        normalized_metric = metric.split("/", 1)[1] if metric.startswith("train/") else metric
        if normalized_metric in {"loss", "grad_norm", "lr", "epoch_time_seconds"}:
            epochs = [record["epoch"] for record in history.get("train_epochs", [])]
            values = [record.get(normalized_metric) for record in history.get("train_epochs", [])]
        elif normalized_metric == "epoch_time":
            epochs = [record["epoch"] for record in history.get("train_epochs", [])]
            values = [record.get("epoch_time_seconds") for record in history.get("train_epochs", [])]
        else:
            epochs = [record["epoch"] for record in history.get("evaluations", [])]
            values = [record["metrics"].get(metric) for record in history.get("evaluations", [])]

        ax.plot(epochs, values, marker="o", linewidth=1.8, label=label, c=f"C{index//color_group_size}")

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    ax.set_title(title or metric)
    ax.set_yscale(yscale)
    ax.grid(True, linestyle=":", linewidth=0.6)
    # ax.legend()
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax
