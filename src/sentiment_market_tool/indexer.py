from __future__ import annotations

import numpy as np
import pandas as pd


def _winsorize_series(series: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
    low = series.quantile(lower_q)
    high = series.quantile(upper_q)
    return series.clip(lower=low, upper=high)


def build_daily_sentiment_index(
    article_frame: pd.DataFrame,
    method: str = "mean",
    winsorize: bool = False,
) -> pd.DataFrame:
    if article_frame.empty:
        return pd.DataFrame(columns=["date", "daily_sentiment", "article_count"])

    frame = article_frame.copy()
    if winsorize:
        frame["sentiment_score"] = _winsorize_series(frame["sentiment_score"])

    method = method.lower().strip()
    if method not in {"mean", "median", "weighted"}:
        raise ValueError("aggregation method must be one of: mean, median, weighted")

    daily_rows: list[dict[str, float | int | pd.Timestamp]] = []
    grouped = frame.groupby("date", as_index=False)

    for _, group in grouped:
        if method == "mean":
            value = float(group["sentiment_score"].mean())
        elif method == "median":
            value = float(group["sentiment_score"].median())
        else:
            weights = group["confidence"].to_numpy(dtype=float)
            scores = group["sentiment_score"].to_numpy(dtype=float)
            if np.allclose(weights.sum(), 0):
                value = float(scores.mean())
            else:
                value = float(np.average(scores, weights=weights))

        daily_rows.append(
            {
                "date": pd.to_datetime(group["date"].iloc[0]),
                "daily_sentiment": value,
                "article_count": int(len(group)),
                "earnings_article_count": int(group["is_earnings_related"].sum()),
            }
        )

    return pd.DataFrame(daily_rows).sort_values("date").reset_index(drop=True)
