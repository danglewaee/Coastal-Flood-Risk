from __future__ import annotations

import numpy as np
import pandas as pd


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


def build_horizon_quantile_calibration(
    window_forecasts: pd.DataFrame,
    central_interval_coverage: float = 0.80,
) -> dict:
    if window_forecasts.empty:
        raise ValueError("Need non-empty window forecasts to build horizon quantile calibration.")

    required = {"horizon_step", "y_true_m", "y_pred_p50_m"}
    missing = required.difference(window_forecasts.columns)
    if missing:
        raise ValueError(f"Missing required backtest columns for calibration: {sorted(missing)}")

    lower_q = (1.0 - float(central_interval_coverage)) / 2.0
    upper_q = 1.0 - lower_q
    horizon_entries: list[dict] = []

    for horizon_step, group in window_forecasts.groupby("horizon_step"):
        residuals = group["y_true_m"].to_numpy(dtype=np.float32) - group["y_pred_p50_m"].to_numpy(dtype=np.float32)
        if residuals.size == 0:
            continue
        q_low = float(np.quantile(residuals, lower_q))
        q_med = float(np.quantile(residuals, 0.50))
        q_high = float(np.quantile(residuals, upper_q))
        coverage = np.logical_and(residuals >= q_low, residuals <= q_high)
        horizon_entries.append(
            {
                "horizon_step": int(horizon_step),
                "count": int(residuals.size),
                "residual_q10_m": q_low,
                "residual_q50_m": q_med,
                "residual_q90_m": q_high,
                "residual_std_m": float(np.std(residuals)),
                "residual_mae_m": float(np.mean(np.abs(residuals))),
                "empirical_coverage": float(np.mean(coverage)),
            }
        )

    if not horizon_entries:
        raise ValueError("No horizon-level residuals were available for calibration.")

    residuals_all = window_forecasts["y_true_m"].to_numpy(dtype=np.float32) - window_forecasts["y_pred_p50_m"].to_numpy(
        dtype=np.float32
    )
    overall_low = float(np.quantile(residuals_all, lower_q))
    overall_med = float(np.quantile(residuals_all, 0.50))
    overall_high = float(np.quantile(residuals_all, upper_q))
    overall_cov = np.logical_and(residuals_all >= overall_low, residuals_all <= overall_high)

    return {
        "method": "rolling_backtest_horizon_quantiles",
        "central_interval_coverage_target": float(central_interval_coverage),
        "lower_tail_quantile": float(lower_q),
        "upper_tail_quantile": float(upper_q),
        "residual_q10_m": overall_low,
        "residual_q50_m": overall_med,
        "residual_q90_m": overall_high,
        "residual_std_m": float(np.std(residuals_all)),
        "residual_mae_m": float(np.mean(np.abs(residuals_all))),
        "empirical_coverage": float(np.mean(overall_cov)),
        "horizon_scale_power": 0.0,
        "horizon_quantiles": horizon_entries,
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

    if use_calibration.get("method") == "rolling_backtest_horizon_quantiles":
        entries = sorted(use_calibration.get("horizon_quantiles") or [], key=lambda item: int(item.get("horizon_step", 0)))
        if entries:
            last = entries[-1]
            lows = []
            meds = []
            highs = []
            for idx in range(point.size):
                horizon_step = idx + 1
                entry = next((item for item in entries if int(item.get("horizon_step", 0)) == horizon_step), last)
                lows.append(point[idx] + float(entry.get("residual_q10_m", 0.0)))
                meds.append(point[idx] + float(entry.get("residual_q50_m", 0.0)))
                highs.append(point[idx] + float(entry.get("residual_q90_m", 0.0)))

            low = np.minimum(np.minimum(np.array(lows, dtype=np.float32), np.array(meds, dtype=np.float32)), np.array(highs, dtype=np.float32))
            high = np.maximum(np.maximum(np.array(lows, dtype=np.float32), np.array(meds, dtype=np.float32)), np.array(highs, dtype=np.float32))
            median = np.clip(np.array(meds, dtype=np.float32), low, high)
            return {
                "p10": low.astype(np.float32),
                "p50": median.astype(np.float32),
                "p90": high.astype(np.float32),
                "calibration": use_calibration,
            }

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
