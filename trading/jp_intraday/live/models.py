"""年次学習済みridgeの永続化（ライブでMLスリーブを動かすため）.

研究側 walk_forward_predictions と同一の学習（当年より前の全年・標準化はtrainのみ・
alpha=10・demeanedターゲット）を行い、係数/平均/標準偏差をJSONで保存する。
年次リトレインで十分なことは検証済み（R4）なので、年初に一度 `train` を実行すればよい。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from trading.jp_intraday.daily_model import BASE_FEATURES, _ridge
from .config import PROJECT_ROOT

MODEL_DIR = PROJECT_ROOT / "data" / "live_models"


def model_path(year: int) -> Path:
    return MODEL_DIR / f"ridge_{year}.json"


def train_and_save(panel: pd.DataFrame, year: int, alpha: float = 10.0,
                   features: list[str] | None = None) -> Path:
    """Train on all rows strictly before ``year`` and persist the model."""
    feats = features or BASE_FEATURES
    p = panel.copy()
    p["date"] = pd.to_datetime(p["date"])
    train = p[p["date"].dt.year < year].dropna(subset=feats + ["target"])
    if len(train) < 5000:
        raise ValueError(f"not enough training rows before {year}: {len(train)}")
    mean = train[feats].mean()
    std = train[feats].std().replace(0, 1).fillna(1)
    beta = _ridge(((train[feats] - mean) / std).to_numpy(), train["target"].to_numpy(), alpha)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = model_path(year)
    path.write_text(json.dumps({
        "year": year, "alpha": alpha, "features": feats,
        "mean": mean.tolist(), "std": std.tolist(), "beta": beta.tolist(),
        "train_rows": int(len(train)),
        "train_end": str(train["date"].max().date()),
    }, indent=2))
    return path


def load_model(year: int) -> dict:
    path = model_path(year)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} がありません。`python -m trading.jp_intraday.live.run_live train` を実行してください")
    return json.loads(path.read_text())


def predict(model: dict, frame: pd.DataFrame) -> pd.Series:
    """Score today's frame with a persisted model (NaN rows -> NaN pred)."""
    feats = model["features"]
    x = (frame[feats] - pd.Series(model["mean"], index=feats)) / pd.Series(model["std"], index=feats)
    valid = x.notna().all(axis=1)
    pred = pd.Series(np.nan, index=frame.index)
    if valid.any():
        pred.loc[valid] = x.loc[valid].to_numpy() @ np.array(model["beta"])
    return pred
