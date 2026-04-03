import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

from .config import TrainConfig
from .data_utils import create_supervised_sequences, invert_zscore, load_series, save_metadata, zscore_normalize
from .evaluation import evaluate_peak_metrics
from .exogenous import load_exogenous_series, resolve_driver_columns
from .features import build_feature_bundle, resolve_feature_mode
from .model import build_model, weighted_peak_mse
from .uncertainty import build_uncertainty_calibration


def train_model(
    csv_path: str,
    value_col: str,
    time_col: str | None,
    output_dir: Path,
    cfg: TrainConfig,
    model_type: str = "lstm",
    drivers_csv: str | None = None,
    drivers_time_col: str = "timestamp",
    driver_columns: str | list[str] | tuple[str, ...] | None = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_series(csv_path=csv_path, time_col=time_col, value_col=value_col)
    values = df[value_col].to_numpy(dtype=np.float32)
    feature_mode = resolve_feature_mode(cfg.feature_mode)
    resolved_driver_columns = resolve_driver_columns(driver_columns) if feature_mode == "multivariate_v2" else []

    exogenous_frame = None
    if feature_mode == "multivariate_v2":
        if not drivers_csv:
            raise ValueError("multivariate_v2 training requires --drivers-csv with hourly exogenous driver data.")
        exogenous_frame = load_exogenous_series(
            drivers_csv,
            time_col=drivers_time_col,
            driver_columns=resolved_driver_columns,
        )

    norm_values, mean, std = zscore_normalize(values)
    feature_frame, feature_context = build_feature_bundle(
        values_normalized=norm_values,
        timestamps=df[time_col] if time_col and time_col in df.columns else None,
        feature_mode=feature_mode,
        exogenous_frame=exogenous_frame,
        driver_columns=resolved_driver_columns,
    )
    x_all, y_all = create_supervised_sequences(
        feature_frame.to_numpy(dtype=np.float32),
        cfg.lookback_hours,
        targets=norm_values,
    )

    split_idx = int(len(x_all) * (1 - cfg.validation_split))
    x_train, x_val = x_all[:split_idx], x_all[split_idx:]
    y_train, y_val = y_all[:split_idx], y_all[split_idx:]

    peak_threshold = float(np.quantile(y_train, cfg.peak_quantile))

    model = build_model(
        model_type=model_type,
        lookback=cfg.lookback_hours,
        hidden_units=cfg.hidden_units,
        lstm_layers=cfg.lstm_layers,
        dropout=cfg.dropout,
        learning_rate=cfg.learning_rate,
        n_features=x_train.shape[-1],
    )

    model.compile(
        optimizer=model.optimizer,
        loss=weighted_peak_mse(peak_threshold, cfg.peak_weight_alpha, cfg.peak_weight_temperature),
        metrics=["mae"],
    )

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        callbacks=[EarlyStopping(monitor="val_loss", patience=cfg.early_stop_patience, restore_best_weights=True)],
        verbose=1,
    )

    val_metrics = model.evaluate(x_val, y_val, verbose=0)
    y_val_pred = model.predict(x_val, verbose=0)

    y_val_true_real = invert_zscore(y_val.reshape(-1), mean, std)
    y_val_pred_real = invert_zscore(y_val_pred.reshape(-1), mean, std)
    residuals_real = y_val_true_real - y_val_pred_real
    peak_metrics = evaluate_peak_metrics(y_val_true_real, y_val_pred_real)
    uncertainty = build_uncertainty_calibration(
        residuals_m=residuals_real,
        horizon_scale_power=cfg.uncertainty_horizon_scale_power,
    )

    model_path = output_dir / f"sea_level_{model_type}.keras"
    metadata_path = output_dir / "metadata.json"

    model.save(model_path)
    save_metadata(
        metadata_path,
        {
            "model_type": model_type,
            "lookback_hours": cfg.lookback_hours,
            "mean": mean,
            "std": std,
            "peak_threshold_normalized": peak_threshold,
            "value_col": value_col,
            "time_col": time_col,
            "feature_mode": feature_mode,
            "feature_names": feature_context["feature_names"],
            "exogenous": {
                "enabled": feature_mode == "multivariate_v2",
                "source_csv": drivers_csv,
                "time_col": drivers_time_col if feature_mode == "multivariate_v2" else None,
                "driver_columns": (feature_context.get("exogenous") or {}).get("driver_columns"),
                "stats": (feature_context.get("exogenous") or {}).get("stats"),
                "future_strategy": "last_observation_persistence" if feature_mode == "multivariate_v2" else None,
            },
            "train_size": int(len(x_train)),
            "val_size": int(len(x_val)),
            "uncertainty": uncertainty,
            "forecast_output": "p50_with_quantile_bands",
        },
    )

    return {
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "model_type": model_type,
        "feature_mode": feature_mode,
        "n_features": int(x_train.shape[-1]),
        "feature_names": feature_context["feature_names"],
        "final_train_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
        "val_loss": float(val_metrics[0]),
        "val_mae": float(val_metrics[1]) if len(val_metrics) > 1 else None,
        "peak_metrics": peak_metrics,
        "uncertainty": uncertainty,
    }


def main():
    parser = argparse.ArgumentParser(description="Train one-step model for sea level forecasting.")
    parser.add_argument("--csv", required=True, help="Input CSV containing sea level time series")
    parser.add_argument("--value-col", default="sea_level", help="Sea level column name")
    parser.add_argument("--time-col", default=None, help="Optional timestamp column")
    parser.add_argument("--model-type", default="lstm", choices=["lstm", "temporal_cnn", "axial_lstm"])
    parser.add_argument("--feature-mode", default="multivariate_v1", choices=["univariate_v0", "multivariate_v1", "multivariate_v2"])
    parser.add_argument("--drivers-csv", default=None, help="Optional hourly exogenous driver CSV for multivariate_v2.")
    parser.add_argument("--drivers-time-col", default="timestamp", help="Timestamp column for the driver CSV.")
    parser.add_argument("--driver-cols", default=None, help="Comma-separated driver columns. Defaults to built-in driver set.")
    parser.add_argument("--out", default="Backend/sea_level_risk/outputs", help="Output directory")
    args = parser.parse_args()

    cfg = TrainConfig(feature_mode=args.feature_mode)
    result = train_model(
        args.csv,
        args.value_col,
        args.time_col,
        Path(args.out),
        cfg,
        model_type=args.model_type,
        drivers_csv=args.drivers_csv,
        drivers_time_col=args.drivers_time_col,
        driver_columns=args.driver_cols,
    )
    print(result)


if __name__ == "__main__":
    tf.random.set_seed(42)
    np.random.seed(42)
    main()
