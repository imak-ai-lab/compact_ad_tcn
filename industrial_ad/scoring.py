"""Anomaly scoring, threshold fitting, metrics and diagnostic figures."""

from __future__ import annotations

import math
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch import nn


def _module_device(module: nn.Module) -> torch.device:
    for tensor in module.parameters():
        return tensor.device
    for tensor in module.buffers():
        return tensor.device
    return torch.device("cpu")


def _reduce_errors(errors: torch.Tensor, reduce_dims: list[int]) -> torch.Tensor:
    """Average selected axes and flatten the remaining error map per sample."""
    for dim in sorted({int(dim) for dim in reduce_dims}, reverse=True):
        errors = errors.mean(dim=dim)
    return errors.reshape(errors.shape[0], -1)


class MahalanobisScorer(nn.Module):
    """Score validation error features with a covariance-aware distance."""

    def __init__(self, feature_dim: int, eps: float) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.eps = float(eps)
        self.register_buffer("inv_cov", torch.eye(self.feature_dim))

    def fit(self, normal_features: torch.Tensor) -> None:
        """Estimate the inverse covariance on normal validation features."""
        if normal_features.shape[0] < 2:
            self.inv_cov.copy_(torch.eye(self.feature_dim, device=normal_features.device))
            return

        covariance = torch.cov(normal_features.T)
        covariance = covariance + torch.eye(self.feature_dim, device=normal_features.device) * self.eps
        self.inv_cov.copy_(torch.linalg.pinv(covariance))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return (features @ self.inv_cov * features).sum(dim=1)


class MeanScorer(nn.Module):
    """Average the error features. Reproduces the author-defined scores of the
    third-party baselines, which average their per-element error map."""

    def fit(self, normal_features: torch.Tensor) -> None:
        return None

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features.mean(dim=1)


class NormScorer(nn.Module):
    """Score validation error features with a vector norm."""

    def __init__(self, order: float) -> None:
        super().__init__()
        self.order = order

    def fit(self, normal_features: torch.Tensor) -> None:
        """Norm scorers are parameter-free, so fitting is a no-op."""
        return None

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.order == math.inf:
            return features.abs().max(dim=1).values
        return torch.linalg.vector_norm(features, ord=self.order, dim=1)


class ModelNativeErrorReducer:
    """Delegate error computation to the model's own `anomaly_error(x, y)`.

    Used for third-party baselines whose anomaly score is part of the method
    (TranAD, USAD, Anomaly Transformer). Unlike the generic reducers, this one
    needs the model and the raw input, so callers must check `needs_model`.
    """

    needs_model = True

    def __call__(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if not hasattr(model, "anomaly_error"):
            raise ValueError(
                f"error_reducer 'model_native' requires {type(model).__name__}.anomaly_error(x, y)."
            )
        errors = model.anomaly_error(x, y)
        return errors.reshape(errors.shape[0], -1)


def _align_target(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Trim the target to the frames the model actually predicts.

    Every model family in the paper predicts as many frames as its target has, so
    this returns `target` unchanged for all of them. It matters for a model that
    predicts a suffix of the window: TranAD in author-faithful sliding mode emits
    the last `120 - inner_window + 1` frames. Same rule as
    `scripts/phase1/phase1_rescore.py::_UnifiedReducer`, so the training-time and
    the rescoring paths agree.
    """
    if prediction.ndim != 3 or target.ndim != 3 or prediction.shape[1] == target.shape[1]:
        return target
    if prediction.shape[1] > target.shape[1]:
        raise ValueError(
            f"prediction is longer than the target ({prediction.shape[1]} > {target.shape[1]}); "
            "the scoring path can only trim a target, not extend it."
        )
    return target[:, -prediction.shape[1]:, :]


def build_error_reducer(config: dict[str, Any]) -> Callable[..., torch.Tensor]:
    """Return the error-to-feature function selected in the config."""
    name = config["name"].lower()
    if name == "model_native":
        return ModelNativeErrorReducer()
    reduce_dims = list(config["params"]["reduce_dims"])
    if name == "mean_abs":
        return lambda prediction, target: _reduce_errors(
            (prediction - _align_target(prediction, target)).abs(), reduce_dims)
    if name == "mean_squared":
        return lambda prediction, target: _reduce_errors(
            (prediction - _align_target(prediction, target)).pow(2), reduce_dims)
    raise ValueError(f"Unknown error reducer: {config['name']}")


def build_score_estimator(config: dict[str, Any], feature_dim: int) -> nn.Module:
    """Return the feature-to-score module selected in the config."""
    name = config["name"].lower()
    params = config["params"]
    if name == "mahalanobis":
        return MahalanobisScorer(feature_dim=feature_dim, eps=float(params["eps"]))
    if name == "mean":
        return MeanScorer()
    if name == "l1":
        return NormScorer(order=1)
    if name == "l2":
        return NormScorer(order=2)
    if name in {"linf", "l_inf"}:
        return NormScorer(order=math.inf)
    raise ValueError(f"Unknown score estimator: {config['name']}")


def fit_best_f1_threshold(scores: np.ndarray, labels: np.ndarray, one_class_quantile: float = 0.99) -> float:
    """Pick the threshold that maximizes F1; one-class validation falls back to a high quantile."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if scores.size == 0:
        return 0.0
    if np.unique(labels).size < 2:
        return float(np.quantile(scores, one_class_quantile))

    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if thresholds.size == 0:
        return float(np.quantile(scores, one_class_quantile))

    f1_values = 2.0 * precision * recall / (precision + recall + 1e-8)
    best_index = min(int(np.nanargmax(f1_values)), thresholds.size - 1)
    return float(thresholds[best_index])


def _safe_metric(metric_fn, labels: np.ndarray, values: np.ndarray) -> float:
    """Return NaN instead of failing on one-class labels."""
    try:
        return float(metric_fn(labels, values))
    except ValueError:
        return float("nan")


def _safe_rank_metric(metric_fn, labels: np.ndarray, scores: np.ndarray) -> float:
    """Metrics based on score ordering need both classes to be present."""
    if np.unique(labels).size < 2:
        return float("nan")
    return float(metric_fn(labels, scores))


def _build_figures(prefix: str, labels: np.ndarray, scores: np.ndarray, preds: np.ndarray):
    """Create ROC, PR and confusion-matrix plots for one evaluation split."""
    figures = {}

    # ROC and PR only exist when both classes are present in the evaluated split.
    if np.unique(labels).size >= 2:
        fpr, tpr, _ = roc_curve(labels, scores)
        fig_roc, ax_roc = plt.subplots(figsize=(4, 4), dpi=100)
        ax_roc.plot(
            fpr,
            tpr,
            label=f"AUC = {_safe_rank_metric(roc_auc_score, labels, scores):.4f}",
            color="tab:blue",
        )
        ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax_roc.set_title(f"{prefix.upper()} ROC")
        ax_roc.set_xlabel("False positive rate")
        ax_roc.set_ylabel("True positive rate")
        ax_roc.grid(True, alpha=0.3)
        ax_roc.legend(loc="lower right")
        fig_roc.tight_layout()
        figures["roc"] = fig_roc

        precision_curve, recall_curve, _ = precision_recall_curve(labels, scores)
        fig_pr, ax_pr = plt.subplots(figsize=(4, 4), dpi=100)
        ax_pr.plot(
            recall_curve,
            precision_curve,
            label=f"AUC = {_safe_rank_metric(average_precision_score, labels, scores):.4f}",
            color="tab:red",
        )
        ax_pr.set_title(f"{prefix.upper()} PR")
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.grid(True, alpha=0.3)
        ax_pr.legend(loc="lower left")
        fig_pr.tight_layout()
        figures["pr"] = fig_pr

    # Confusion matrix is always produced because it is needed even for one-class smoke runs.
    matrix = confusion_matrix(labels, preds, labels=[0, 1])
    fig_cm, ax_cm = plt.subplots(figsize=(4, 4), dpi=100)
    ax_cm.imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
    ax_cm.set_title(f"{prefix.upper()} Confusion Matrix")
    ax_cm.set_xticks([0, 1], ["Normal", "Anomaly"])
    ax_cm.set_yticks([0, 1], ["Normal", "Anomaly"])
    threshold = matrix.max() / 2 if matrix.size else 0.0
    for index in np.ndindex(matrix.shape):
        ax_cm.text(
            index[1],
            index[0],
            int(matrix[index]),
            ha="center",
            va="center",
            color="white" if matrix[index] > threshold else "black",
        )
    ax_cm.set_xlabel("Predicted label")
    ax_cm.set_ylabel("True label")
    fig_cm.tight_layout()
    figures["cm"] = fig_cm
    return figures


class AnomalyDetectorWrapper(nn.Module):
    """Pair a predictor model with error reduction, score fitting and thresholding."""

    def __init__(
        self,
        model: nn.Module,
        error_reducer: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        score_estimator: nn.Module,
        threshold_config: dict[str, Any],
    ) -> None:
        super().__init__()
        if threshold_config["name"].lower() != "best_f1":
            raise ValueError(f"Unknown threshold strategy: {threshold_config['name']}")
        self.model = model
        self.error_reducer = error_reducer
        self.score_estimator = score_estimator
        self.register_buffer("threshold", torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def get_errors(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Turn model predictions into per-sample error features."""
        if getattr(self.error_reducer, "needs_model", False):
            # Model-defined scores (third-party baselines) need the raw input.
            return self.error_reducer(self.model, x, y)
        return self.error_reducer(self.model(x), y)

    def get_anomaly_scores(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Map error features to scalar anomaly scores."""
        return self.score_estimator(self.get_errors(x, y))

    @torch.no_grad()
    def fit_score_estimator(self, dataloader, max_batches: int | None) -> None:
        """Fit the score estimator using only normal validation windows."""
        self.model.eval()
        device = _module_device(self.model)
        normal_errors = []

        # Validation contains both normal and anomalous windows, but the scorer
        # itself must only model how normal errors look.
        for batch_index, (x, y, is_anomaly) in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break
            mask = ~is_anomaly.bool()
            if mask.any():
                normal_errors.append(self.get_errors(x[mask].to(device), y[mask].to(device)))

        if not normal_errors:
            raise RuntimeError("No normal validation samples were available to fit the anomaly scorer.")
        self.score_estimator.fit(torch.cat(normal_errors, dim=0))

    @torch.no_grad()
    def get_scores_and_labels(self, dataloader, max_batches: int | None):
        """Run one split and return NumPy arrays used by threshold fitting and metrics."""
        self.model.eval()
        device = _module_device(self.model)
        scores, labels = [], []

        # Keep computations on the model device and move only final score vectors back to CPU.
        for batch_index, (x, y, is_anomaly) in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break
            scores.append(self.get_anomaly_scores(x.to(device), y.to(device)).cpu())
            labels.append(is_anomaly.cpu())

        if not scores:
            return np.array([], dtype=float), np.array([], dtype=int)
        return torch.cat(scores).numpy(), torch.cat(labels).numpy().astype(int)

    def fit_threshold(self, dataloader, max_batches: int | None) -> float:
        """Fit the scalar decision threshold on validation scores."""
        scores, labels = self.get_scores_and_labels(dataloader, max_batches=max_batches)
        threshold = fit_best_f1_threshold(scores, labels)
        self.threshold.fill_(float(threshold))
        return threshold

    def evaluate(self, dataloader, *, prefix: str, max_batches: int | None):
        """Compute scalar metrics and diagnostic plots for one split."""
        scores, labels = self.get_scores_and_labels(dataloader, max_batches=max_batches)
        preds = (scores >= float(self.threshold.item())).astype(int)
        matrix = confusion_matrix(labels, preds, labels=[0, 1])

        # The same threshold is reused for both scalar metrics and saved plots so
        # the artifacts always match the numbers written to history.json.
        metrics = {
            f"{prefix}/roc_auc": _safe_rank_metric(roc_auc_score, labels, scores),
            f"{prefix}/pr_auc": _safe_rank_metric(average_precision_score, labels, scores),
            f"{prefix}/accuracy": _safe_metric(accuracy_score, labels, preds),
            f"{prefix}/precision": _safe_metric(lambda y_true, y_pred: precision_score(y_true, y_pred, zero_division=0), labels, preds),
            f"{prefix}/recall": _safe_metric(lambda y_true, y_pred: recall_score(y_true, y_pred, zero_division=0), labels, preds),
            f"{prefix}/f1": _safe_metric(lambda y_true, y_pred: f1_score(y_true, y_pred, zero_division=0), labels, preds),
            f"{prefix}/mcc": _safe_metric(matthews_corrcoef, labels, preds),
            f"{prefix}/threshold": float(self.threshold.item()),
            f"{prefix}/confusion_matrix_values": matrix.tolist(),
        }
        return metrics, _build_figures(prefix, labels, scores, preds)
