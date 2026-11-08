"""Public API for experiment execution, quantization and run analysis."""

# Python 3.14 changed the default multiprocessing start method on Linux to
# `forkserver`, which has two problems with our DataLoader pipeline:
#   1. forkserver pickles worker args (the dataset object's `target_builder`,
#      `file_transform`, etc.) — any closure or lambda blows up. We fixed the
#      lambdas in datasets/PU/features.py, but other unpicklable closures may
#      live elsewhere in user code.
#   2. The forkserver itself is flaky on 3.14 — ConnectionResetError during
#      worker spawn (handshake fails).
# We force `fork` (the pre-3.14 default), which copies the parent process
# memory and doesn't pickle anything. The DataLoader workers never init CUDA,
# so the usual "fork after CUDA init" caveat doesn't apply here.
import multiprocessing as _mp
try:
    _mp.set_start_method("fork", force=True)
except RuntimeError:
    # set_start_method can only be called once; ignore if already configured.
    pass

from industrial_ad.analysis import (
    benchmark_run,
    benchmark_runs,
    discover_run_dirs,
    load_run_dataframe,
    load_run_summaries,
    mark_pareto_front,
    pareto_mask,
    plot_metric_history,
    plot_tradeoff_scatter,
)
from industrial_ad.config import (
    DEFAULT_EXPERIMENT_CONFIG,
    DEFAULT_QUANTIZATION_CONFIG,
    make_default_config,
    make_default_quantization_config,
    validate_experiment_config,
    validate_quantization_config,
)
from industrial_ad.experiments import load_detector_from_run, run_experiment, run_experiments
from industrial_ad.quantization import run_quantization, run_quantizations
from industrial_ad.utils import clone_config

__all__ = [
    "DEFAULT_EXPERIMENT_CONFIG",
    "DEFAULT_QUANTIZATION_CONFIG",
    "benchmark_run",
    "benchmark_runs",
    "clone_config",
    "discover_run_dirs",
    "load_detector_from_run",
    "load_run_dataframe",
    "load_run_summaries",
    "make_default_config",
    "make_default_quantization_config",
    "mark_pareto_front",
    "pareto_mask",
    "plot_metric_history",
    "plot_tradeoff_scatter",
    "run_experiment",
    "run_experiments",
    "run_quantization",
    "run_quantizations",
    "validate_experiment_config",
    "validate_quantization_config",
]
