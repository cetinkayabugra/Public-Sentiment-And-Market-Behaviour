from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
import requests


NEWS_API_URL = "https://newsapi.org/v2/everything"
EARNINGS_KEYWORDS = (
    "earnings",
    "eps",
    "quarterly report",
    "revenue beat",
    "profit warning",
    "guidance",
)


def validate_date(date_str: str) -> str:
    datetime.strptime(date_str, "%Y-%m-%d")
    return date_str


def fetch_news(
    api_key: str,
    query: str,
    start_date: str,
    end_date: str,
    language: str = "en",
    sort_by: str = "publishedAt",
    page_size: int = 100,
    max_pages: int = 5,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for page in range(1, max_pages + 1):
        params = {
            "q": query,
            "from": start_date,
            "to": end_date,
            "language": language,
            "sortBy": sort_by,
            "pageSize": page_size,
            "page": page,
            "apiKey": api_key,
        }
        response = requests.get(NEWS_API_URL, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()

        if payload.get("status") != "ok":
            message = payload.get("message", "Unknown News API error")
            raise RuntimeError(f"News API error: {message}")

        articles = payload.get("articles", [])
        if not articles:
            break

        for article in articles:
            rows.append(
                {
                    "source": (article.get("source") or {}).get("name"),
                    "author": article.get("author"),
                    "title": article.get("title"),
                    "description": article.get("description"),
                    "content": article.get("content"),
                    "url": article.get("url"),
                    "published_at": article.get("publishedAt"),
                }
            )

        total_results = int(payload.get("totalResults", 0))
        if page * page_size >= total_results:
            break

    if not rows:
        return pd.DataFrame(
            columns=[
                "source",
                "author",
                "title",
                "description",
                "content",
                "url",
                "published_at",
                "date",
                "is_earnings_related",
            ]
        )

    frame = pd.DataFrame(rows)
    frame["published_at"] = pd.to_datetime(frame["published_at"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["published_at"]).reset_index(drop=True)
    frame = frame.drop_duplicates(subset=["url"]).reset_index(drop=True)

    frame["date"] = pd.to_datetime(frame["published_at"].dt.date)
    text = (
        frame[["title", "description", "content"]]
        .fillna("")
        .agg(" ".join, axis=1)
        .str.lower()
    )
    frame["is_earnings_related"] = text.apply(
        lambda value: any(keyword in value for keyword in EARNINGS_KEYWORDS)
    )
    return frame
