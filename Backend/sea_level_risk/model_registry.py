from __future__ import annotations

import json
from pathlib import Path


DEFAULT_MODELS_ROOT = Path("Backend/sea_level_risk/outputs/models")


def city_model_dir(city_key: str, models_root: str | Path = DEFAULT_MODELS_ROOT) -> Path:
    return Path(models_root) / city_key.strip().lower()


def resolve_city_model(city_key: str, models_root: str | Path = DEFAULT_MODELS_ROOT) -> dict | None:
    city_dir = city_model_dir(city_key, models_root=models_root)
    if not city_dir.exists():
        return None

    metadata_path = city_dir / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
        except Exception:
            metadata = {}

    model_type = metadata.get("model_type")
    model_path = None
    if model_type:
        candidate = city_dir / f"sea_level_{model_type}.keras"
        if candidate.exists():
            model_path = candidate

    if model_path is None:
        keras_files = sorted(city_dir.glob("*.keras"))
        if len(keras_files) == 1:
            model_path = keras_files[0]

    if model_path is None or not metadata_path.exists():
        return None

    return {
        "city": city_key.strip().lower(),
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "model_type": model_type,
        "model_dir": str(city_dir),
    }


def list_available_city_models(models_root: str | Path = DEFAULT_MODELS_ROOT) -> list[dict]:
    root = Path(models_root)
    if not root.exists():
        return []

    records: list[dict] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        spec = resolve_city_model(child.name, models_root=root)
        if spec:
            records.append(spec)
    return records
