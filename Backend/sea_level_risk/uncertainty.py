from __future__ import annotations

import numpy as np


def build_uncertainty_calibration(
    residuals_m: np.ndarray,
    horizon_scale_power: float = 0.5,
) -> dict:
    residuals = np.asarray(residuals_m, dtype=np.float32).reshape(-1)
    if residuals.size == 0:
        residuals = np.array([0.0], dtype=np.float32)

    return {
        "method": "validation_residual_quantiles",
        "residual_q10_m": float(np.quantile(residuals, 0.10)),
        "residual_q50_m": float(np.quantile(residuals, 0.50)),
        "residual_q90_m": float(np.quantile(residuals, 0.90)),
        "residual_std_m": float(np.std(residuals)),
        "residual_mae_m": float(np.mean(np.abs(residuals))),
        "horizon_scale_power": float(horizon_scale_power),
    }


def fallback_uncertainty_calibration(
    recent_values: np.ndarray,
    horizon_scale_power: float = 0.5,
) -> dict:
    recent = np.asarray(recent_values, dtype=np.float32).reshape(-1)
    if recent.size < 3:
        scale = 0.05
    else:
        scale = float(np.std(np.diff(recent[-24:])))
        scale = max(scale, 0.03)

    return {
        "method": "recent_volatility",
        "residual_q10_m": -1.2816 * scale,
        "residual_q50_m": 0.0,
        "residual_q90_m": 1.2816 * scale,
        "residual_std_m": scale,
        "residual_mae_m": abs(scale),
        "horizon_scale_power": float(horizon_scale_power),
    }


def calibrated_quantile_forecast(
    point_forecast_m: np.ndarray,
    recent_values: np.ndarray,
    calibration: dict | None,
) -> dict[str, np.ndarray]:
    point = np.asarray(point_forecast_m, dtype=np.float32).reshape(-1)
    use_calibration = calibration or fallback_uncertainty_calibration(recent_values)

    q10 = float(use_calibration.get("residual_q10_m", 0.0))
    q50 = float(use_calibration.get("residual_q50_m", 0.0))
    q90 = float(use_calibration.get("residual_q90_m", 0.0))
    power = float(use_calibration.get("horizon_scale_power", 0.5))

    scales = np.power(np.arange(1, point.size + 1, dtype=np.float32), power)
    p10 = point + q10 * scales
    p50 = point + q50 * scales
    p90 = point + q90 * scales

    low = np.minimum(np.minimum(p10, p50), p90)
    high = np.maximum(np.maximum(p10, p50), p90)
    median = np.clip(p50, low, high)

    return {
        "p10": low.astype(np.float32),
        "p50": median.astype(np.float32),
        "p90": high.astype(np.float32),
        "calibration": use_calibration,
    }
