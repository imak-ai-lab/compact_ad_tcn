"""Handcrafted PU bearing features, normalization and task target builders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pywt
import scipy.stats as stats


CLASSIC_27D_FEATURE_STATS = {
    "name": "pu_n15_classic_27d_v1",
    "mean": [
        0.0008720479672774673,
        0.06539903581142426,
        0.7519486546516418,
        0.7700129747390747,
        -0.002356527838855982,
        9950.859375,
        648.1812744140625,
        0.0009613807778805494,
        0.0004422074125614017,
        0.05746344104409218,
        0.06589289754629135,
        0.7566071152687073,
        0.7672395706176758,
        0.004345756955444813,
        10073.275390625,
        648.4998168945312,
        0.0009026339394040406,
        0.00019069609697908163,
        -0.01557720173150301,
        0.01998274214565754,
        0.5922828316688538,
        3.2385776042938232,
        0.00812938529998064,
        47.34132766723633,
        3318.11376953125,
        0.30952945351600647,
        0.4029441475868225,
    ],
    "std": [
        1.5481231212615967,
        0.06363021582365036,
        0.42847079038619995,
        0.42573168873786926,
        0.438698410987854,
        8310.8251953125,
        707.4442138671875,
        0.0020546717569231987,
        0.003612906439229846,
        1.556606411933899,
        0.06330733746290207,
        0.4266241788864136,
        0.42232587933540344,
        0.43618273735046387,
        8500.1005859375,
        706.6968994140625,
        0.0021666893735527992,
        0.0032320318277925253,
        0.03497103601694107,
        0.045265082269907,
        0.6493504047393799,
        5.226101398468018,
        1.2407475709915161,
        102.82904052734375,
        2159.624267578125,
        0.9808717966079712,
        1.3291105031967163,
    ],
}


def extract_features_vectorized(x: np.ndarray, window: int, step: int, fs: float = 1.0) -> dict[str, np.ndarray]:
    """Extract the handcrafted 27D feature set from one denoised 1D signal."""
    windows = np.lib.stride_tricks.sliding_window_view(x, window)[::step]
    eps = 1e-12

    variance = np.var(windows, axis=1)
    peak_to_peak = np.ptp(windows, axis=1)
    line_integral = np.sum(np.abs(np.diff(windows, axis=1)), axis=1)
    skewness = stats.skew(windows, axis=1)

    freqs = np.fft.rfftfreq(window, d=1.0 / fs)
    fft_values = np.fft.rfft(windows, axis=1)
    power_spectrum = np.abs(fft_values) ** 2
    spectral_energy = np.sum(power_spectrum, axis=1)
    centroid = np.sum(freqs * power_spectrum, axis=1) / (spectral_energy + eps)
    spectral_spread = np.sqrt(
        np.sum((freqs - centroid[:, None]) ** 2 * power_spectrum, axis=1) / (spectral_energy + eps)
    )

    coeffs = pywt.wavedec(windows, "db4", level=3, axis=1)
    wavelet_energy_d2 = np.sum(coeffs[2] ** 2, axis=1)
    wavelet_energy_d1 = np.sum(coeffs[3] ** 2, axis=1)

    return {
        "Mean": np.mean(windows, axis=1),
        "Variance": variance,
        "Peak-to-Peak": peak_to_peak,
        "Line Integral": line_integral,
        "Skewness": skewness,
        "Spectral Energy": spectral_energy,
        "Spectral Spread": spectral_spread,
        "Wavelet Energy D2": wavelet_energy_d2,
        "Wavelet Energy D1": wavelet_energy_d1,
    }


def denoise_signal_dwt(x: np.ndarray, wavelet: str, level: int) -> np.ndarray:
    """Denoise a raw signal with soft-thresholded wavelet coefficients."""
    coeffs = pywt.wavedec(x, wavelet, mode="per", level=level)
    detail_coeffs = coeffs[-1]
    mad = np.median(np.abs(detail_coeffs - np.median(detail_coeffs)))
    sigma = mad / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(x)))

    denoised = [coeffs[0]]
    for detail in coeffs[1:]:
        denoised.append(pywt.threshold(detail, value=threshold, mode="soft"))

    restored = pywt.waverec(denoised, wavelet, mode="per")
    return restored[: len(x)]


def process_bearing_signals_to_features(
    data: np.ndarray,
    window_len: int,
    step_len: int,
    fs: int,
    drop_edges: int,
) -> tuple[np.ndarray, list[str]]:
    """Apply denoising and handcrafted features to the 3 raw PU channels."""
    channel_specs = [
        ("phase1", "sym8", 6),
        ("phase2", "sym8", 6),
        ("vibration", "db4", 4),
    ]
    all_channel_features = []
    feature_names: list[str] = []

    for channel_index, (channel_name, wavelet, level) in enumerate(channel_specs):
        # Each raw channel uses the same handcrafted feature family but its own denoising recipe.
        clean_signal = denoise_signal_dwt(data[channel_index], wavelet=wavelet, level=level)
        if drop_edges:
            clean_signal = clean_signal[drop_edges:-drop_edges]
        features = extract_features_vectorized(clean_signal, window=window_len, step=step_len, fs=fs)

        channel_features = []
        for feature_name, values in features.items():
            channel_features.append(values)
            if channel_index == 0:
                feature_names.append(feature_name)
        all_channel_features.append(np.asarray(channel_features))

    combined = np.vstack(all_channel_features)
    full_names = [
        f"{channel_name}_{feature_name}"
        for channel_name, _, _ in channel_specs
        for feature_name in feature_names
    ]
    return combined, full_names


@dataclass
class PUFeaturePipeline:
    """Transform one raw PU file into a normalized feature matrix."""

    aggregation_window: int
    aggregation_step: int
    sampling_rate: int
    drop_edges: int
    raw_channel_indices: tuple[int, ...]
    scaler_mean: np.ndarray | None
    scaler_std: np.ndarray | None

    def __call__(self, raw_data: np.ndarray) -> np.ndarray:
        # The dataset loader keeps only the requested raw channels, then this
        # pipeline converts them into the 27D handcrafted feature timeline.
        selected = raw_data[:, self.raw_channel_indices].T
        feature_matrix, _ = process_bearing_signals_to_features(
            selected,
            window_len=self.aggregation_window,
            step_len=self.aggregation_step,
            fs=self.sampling_rate,
            drop_edges=self.drop_edges,
        )
        feature_matrix = feature_matrix.T.astype(np.float32)
        if self.scaler_mean is not None and self.scaler_std is not None:
            # The preset scaler is fixed from a reference training set and keeps smoke runs deterministic.
            feature_matrix = (feature_matrix - self.scaler_mean) / self.scaler_std
        return feature_matrix.astype(np.float32)


def build_pu_feature_pipeline(config: dict[str, Any]) -> PUFeaturePipeline:
    """Build the feature pipeline selected in `dataset.params.feature_pipeline`."""
    if config["name"].lower() != "classic_27d_v1":
        raise ValueError(f"Unknown PU feature pipeline: {config['name']}")

    params = dict(config["params"])
    if params["use_preset_scaler"]:
        scaler_mean = np.asarray(CLASSIC_27D_FEATURE_STATS["mean"], dtype=np.float32)
        scaler_std = np.asarray(CLASSIC_27D_FEATURE_STATS["std"], dtype=np.float32)
    else:
        scaler_mean = None
        scaler_std = None

    return PUFeaturePipeline(
        aggregation_window=int(params["aggregation_window"]),
        aggregation_step=int(params["aggregation_step"]),
        sampling_rate=int(params["sampling_rate"]),
        drop_edges=int(params["drop_edges"]),
        raw_channel_indices=tuple(int(index) for index in params["raw_channel_indices"]),
        scaler_mean=scaler_mean,
        scaler_std=scaler_std,
    )


def _target_window(window, future):
    """Reconstruction target: return the input window unchanged."""
    return window


def _target_future(window, future):
    """Forecasting target: return the future window."""
    return future


def build_target_builder(task_type: str) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Return the target builder for reconstruction or forecasting.

    Module-level functions (not lambdas) so that DataLoader workers can pickle
    them under Python 3.14's stricter multiprocessing start methods.
    """
    task_type = task_type.lower()
    if task_type == "reconstruction":
        return _target_window
    if task_type == "forecasting":
        return _target_future
    raise ValueError(f"Unknown task type: {task_type}")
