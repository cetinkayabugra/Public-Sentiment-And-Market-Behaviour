from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf


def fetch_market_data(
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str = "1d",
) -> pd.DataFrame:
    frame = yf.download(
        tickers=ticker,
        start=start_date,
        end=end_date,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )

    if frame.empty:
        return pd.DataFrame(columns=["date", "close", "log_return"])

    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)

    frame = frame.reset_index()
    if "Date" in frame.columns:
        frame = frame.rename(columns={"Date": "date"})
    if "Datetime" in frame.columns:
        frame = frame.rename(columns={"Datetime": "date"})

    frame = frame.rename(columns={"Close": "close"})
    frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None).dt.normalize()

    frame["log_return"] = np.log(frame["close"] / frame["close"].shift(1))
    return frame[["date", "close", "log_return"]]
