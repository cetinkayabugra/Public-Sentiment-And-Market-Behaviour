from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import statsmodels.api as sm


@dataclass
class CorrelationResult:
    method: str
    value: float | None
    p_value: float | None
    observations: int


def merge_sentiment_market(
    daily_sentiment: pd.DataFrame,
    market: pd.DataFrame,
    join: str = "inner",
    missing: str = "drop",
) -> pd.DataFrame:
    merged = daily_sentiment.merge(market, on="date", how=join)
    if missing == "drop":
        merged = merged.dropna(subset=["daily_sentiment", "log_return"])
    elif missing == "ffill":
        merged = merged.sort_values("date").ffill().dropna(subset=["daily_sentiment", "log_return"])
    else:
        raise ValueError("missing must be one of: drop, ffill")
    return merged.sort_values("date").reset_index(drop=True)


def detect_extremes(merged: pd.DataFrame, z_threshold: float = 2.0) -> pd.DataFrame:
    frame = merged.copy()
    std = frame["daily_sentiment"].std(ddof=0)
    if std == 0 or np.isnan(std):
        frame["sentiment_z"] = 0.0
    else:
        frame["sentiment_z"] = (frame["daily_sentiment"] - frame["daily_sentiment"].mean()) / std
    extremes = frame[np.abs(frame["sentiment_z"]) >= z_threshold].copy()
    return extremes[["date", "daily_sentiment", "sentiment_z", "log_return"]].reset_index(drop=True)


def run_correlation(merged: pd.DataFrame, method: str = "pearson") -> CorrelationResult:
    frame = merged[["daily_sentiment", "log_return"]].dropna()
    n = len(frame)
    if n < 3:
        return CorrelationResult(method=method, value=None, p_value=None, observations=n)

    if method == "pearson":
        value, p_val = pearsonr(frame["daily_sentiment"], frame["log_return"])
    elif method == "spearman":
        value, p_val = spearmanr(frame["daily_sentiment"], frame["log_return"])
    else:
        raise ValueError("method must be pearson or spearman")

    return CorrelationResult(method=method, value=float(value), p_value=float(p_val), observations=n)


def run_rolling_correlation(
    merged: pd.DataFrame,
    window: int = 30,
    method: str = "pearson",
) -> pd.DataFrame:
    frame = merged[["date", "daily_sentiment", "log_return"]].copy()
    if method == "pearson":
        frame["rolling_corr"] = frame["daily_sentiment"].rolling(window).corr(frame["log_return"])
    elif method == "spearman":
        frame["daily_sentiment_rank"] = frame["daily_sentiment"].rank()
        frame["log_return_rank"] = frame["log_return"].rank()
        frame["rolling_corr"] = frame["daily_sentiment_rank"].rolling(window).corr(frame["log_return_rank"])
        frame = frame.drop(columns=["daily_sentiment_rank", "log_return_rank"])
    else:
        raise ValueError("method must be pearson or spearman")
    return frame


def _ols(y: pd.Series, x: pd.DataFrame) -> Any:
    x_with_const = sm.add_constant(x, has_constant="add")
    model = sm.OLS(y, x_with_const, missing="drop")
    return model.fit()


def regression_same_day(merged: pd.DataFrame) -> Any:
    frame = merged[["log_return", "daily_sentiment"]].dropna()
    return _ols(frame["log_return"], frame[["daily_sentiment"]])


def regression_lagged(merged: pd.DataFrame, lag: int = 1) -> Any:
    frame = merged[["log_return", "daily_sentiment"]].copy()
    frame[f"daily_sentiment_lag_{lag}"] = frame["daily_sentiment"].shift(lag)
    frame = frame.dropna()
    return _ols(frame["log_return"], frame[[f"daily_sentiment_lag_{lag}"]])


def regression_multilag(merged: pd.DataFrame, lag_count: int = 3) -> Any:
    frame = merged[["log_return", "daily_sentiment"]].copy()
    lag_columns: list[str] = []
    for lag in range(1, lag_count + 1):
        col = f"daily_sentiment_lag_{lag}"
        frame[col] = frame["daily_sentiment"].shift(lag)
        lag_columns.append(col)
    frame = frame.dropna()
    return _ols(frame["log_return"], frame[lag_columns])


def economic_interpretation(beta: float | None, delta_sentiment: float = 0.10) -> dict[str, float | None]:
    if beta is None or np.isnan(beta):
        return {"delta_sentiment": delta_sentiment, "expected_log_return_change": None}
    return {
        "delta_sentiment": delta_sentiment,
        "expected_log_return_change": float(beta * delta_sentiment),
    }


def model_summary_dict(model_result: Any) -> dict[str, Any]:
    params = {k: float(v) for k, v in model_result.params.to_dict().items()}
    pvalues = {k: float(v) for k, v in model_result.pvalues.to_dict().items()}
    tvalues = {k: float(v) for k, v in model_result.tvalues.to_dict().items()}
    return {
        "nobs": int(model_result.nobs),
        "r_squared": float(model_result.rsquared),
        "params": params,
        "p_values": pvalues,
        "t_values": tvalues,
    }


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def correlation_result_to_dict(result: CorrelationResult) -> dict[str, Any]:
    return asdict(result)
