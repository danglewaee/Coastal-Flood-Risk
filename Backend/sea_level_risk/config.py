from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainConfig:
    lookback_hours: int = 24
    hidden_units: int = 64
    lstm_layers: int = 2
    dropout: float = 0.15
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 80
    validation_split: float = 0.15
    early_stop_patience: int = 10
    peak_quantile: float = 0.9
    peak_weight_alpha: float = 3.0
    peak_weight_temperature: float = 0.35
    feature_mode: str = "multivariate_v1"
    uncertainty_horizon_scale_power: float = 0.5


@dataclass(frozen=True)
class PathConfig:
    project_root: Path = Path(__file__).resolve().parents[1]
    output_dir: Path = project_root / "sea_level_risk" / "outputs"
