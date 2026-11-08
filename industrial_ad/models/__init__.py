"""Explicit model factory for all architectures used in the experiments."""

from __future__ import annotations

from typing import Any

from industrial_ad.models.mlp_ae import MLPAutoencoder
from industrial_ad.models.mlp_forecaster import MLPForecaster
from industrial_ad.models.tcn_ae import TCNAutoencoder
from industrial_ad.models.tcn_forecaster import TCNForecaster
from industrial_ad.models.tcn_light_forecaster import TCNLightForecaster
from industrial_ad.models.gru_seq2seq_ae import GRUSeq2seqAutoencoder
from industrial_ad.models.gru_repeated_ae import GRURepeatedAutoencoder
from industrial_ad.models.pca import PCAReconstructionModel
from industrial_ad.models.transformer_ae import TransformerAutoencoder
from industrial_ad.models.conv_ae import ConvAutoencoder
from industrial_ad.models.gru_seq2seq_forecaster import GRUSeq2seqForecaster
from industrial_ad.models.sota import AnomalyTransformerDetector, TranAD, USAD


def build_model(model_config: dict[str, Any], input_shape: tuple[int, ...], target_shape: tuple[int, ...]):
    """Build a model family selected by `model.name`."""
    model_name = model_config["name"].lower()
    model_params = dict(model_config["params"])

    if model_name in ["window_mlp", "mlp_ae"]:
        return MLPAutoencoder(input_shape=input_shape, output_shape=target_shape, **model_params)
    if model_name == "tcn_ae":
        return TCNAutoencoder(input_shape=input_shape, output_shape=target_shape, **model_params)
    if model_name == "gru_seq2seq_ae":
        return GRUSeq2seqAutoencoder(input_shape=input_shape, output_shape=target_shape, **model_params)
    if model_name == "gru_repeated_ae":
        return GRURepeatedAutoencoder(input_shape=input_shape, output_shape=target_shape, **model_params)
    if model_name == "transformer_ae":
        return TransformerAutoencoder(input_shape=input_shape, output_shape=target_shape, **model_params)
    if model_name == "conv_ae":
        return ConvAutoencoder(input_shape=input_shape, output_shape=target_shape, **model_params)
    if model_name == "pca":
        return PCAReconstructionModel(input_shape=input_shape, output_shape=target_shape, **model_params)

    if model_name == "mlp_forecaster":
        return MLPForecaster(input_shape=input_shape, output_shape=target_shape, **model_params)
    if model_name == "tcn_forecaster":
        return TCNForecaster(input_shape=input_shape, output_shape=target_shape, **model_params)
    if model_name == "tcn_light_forecaster":
        return TCNLightForecaster(input_shape=input_shape, output_shape=target_shape, **model_params)
    if model_name == "gru_seq2seq_forecaster":
        return GRUSeq2seqForecaster(input_shape=input_shape, output_shape=target_shape, **model_params)

    # Third-party SOTA baselines; see models/sota/.
    if model_name == "tranad":
        return TranAD(input_shape=input_shape, output_shape=target_shape, **model_params)
    if model_name == "usad":
        return USAD(input_shape=input_shape, output_shape=target_shape, **model_params)
    if model_name in {"anomaly_transformer", "anomaly_transformer_ad"}:
        return AnomalyTransformerDetector(input_shape=input_shape, output_shape=target_shape, **model_params)

    raise ValueError(f"Unknown model: {model_config['name']}")


__all__ = ["build_model"]
