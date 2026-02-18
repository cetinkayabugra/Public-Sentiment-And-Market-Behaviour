from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sentiment_market_tool.analytics import (
    correlation_result_to_dict,
    detect_extremes,
    economic_interpretation,
    merge_sentiment_market,
    model_summary_dict,
    regression_lagged,
    regression_multilag,
    regression_same_day,
    run_correlation,
    run_rolling_correlation,
    save_json,
)
from sentiment_market_tool.audit import ResearchSession
from sentiment_market_tool.config import AppConfig
from sentiment_market_tool.indexer import build_daily_sentiment_index
from sentiment_market_tool.market import fetch_market_data
from sentiment_market_tool.news import fetch_news
from sentiment_market_tool.sentiment import FinBERTScorer, build_article_text


@dataclass
class PipelineOptions:
    scope: str
    query: str
    ticker: str
    start_date: str
    end_date: str
    aggregation: str
    winsorize: bool
    z_threshold: float
    corr_method: str
    rolling_window: int
    rolling_method: str
    regression_mode: str
    lag_count: int
    join_method: str
    missing_method: str
    earnings_focus: bool
    output_dir: str


def run_research_pipeline(config: AppConfig, options: PipelineOptions) -> dict[str, Any]:
    out_dir = Path(options.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    session = ResearchSession()
    session.log(
        "start_session",
        scope=options.scope,
        query=options.query,
        ticker=options.ticker,
        start_date=options.start_date,
        end_date=options.end_date,
    )

    news = fetch_news(
        api_key=config.newsapi_key,
        query=options.query,
        start_date=options.start_date,
        end_date=options.end_date,
    )
    if options.earnings_focus:
        news = news[news["is_earnings_related"]].reset_index(drop=True)
    if news.empty:
        raise RuntimeError("No news records available for selected scope and date range.")

    news.to_csv(out_dir / "raw_news.csv", index=False)
    session.log("fetch_news", records=int(len(news)), earnings_focus=options.earnings_focus)

    text_series = build_article_text(news)
    sentiment = FinBERTScorer().score(text_series)
    article_sentiment = news.reset_index(drop=True).join(sentiment)
    article_sentiment.to_csv(out_dir / "article_sentiment.csv", index=False)
    session.log("run_finbert", records=int(len(article_sentiment)))

    daily_index = build_daily_sentiment_index(
        article_sentiment,
        method=options.aggregation,
        winsorize=options.winsorize,
    )
    daily_index.to_csv(out_dir / "daily_sentiment_index.csv", index=False)
    session.log(
        "build_sentiment_index",
        method=options.aggregation,
        winsorize=options.winsorize,
        days=int(len(daily_index)),
    )

    market = fetch_market_data(
        ticker=options.ticker,
        start_date=options.start_date,
        end_date=options.end_date,
    )
    if market.empty:
        raise RuntimeError("No market data available for selected ticker and date range.")
    market.to_csv(out_dir / "market_data.csv", index=False)
    session.log("fetch_market_data", records=int(len(market)))

    merged = merge_sentiment_market(
        daily_sentiment=daily_index,
        market=market,
        join=options.join_method,
        missing=options.missing_method,
    )
    if merged.empty:
        raise RuntimeError("Merged dataset is empty after join and missing handling.")
    merged.to_csv(out_dir / "merged_dataset.csv", index=False)
    session.log(
        "merge_datasets",
        rows_before=len(daily_index),
        market_rows=len(market),
        rows_after=len(merged),
        join=options.join_method,
        missing=options.missing_method,
    )

    corr = run_correlation(merged, method=options.corr_method)
    rolling = run_rolling_correlation(
        merged,
        window=options.rolling_window,
        method=options.rolling_method,
    )
    rolling.to_csv(out_dir / "rolling_correlation.csv", index=False)
    session.log(
        "run_correlation",
        method=options.corr_method,
        value=corr.value,
        p_value=corr.p_value,
        observations=corr.observations,
    )

    extremes = detect_extremes(merged, z_threshold=options.z_threshold)
    extremes.to_csv(out_dir / "extreme_sentiment_days.csv", index=False)
    session.log("detect_extremes", z_threshold=options.z_threshold, count=int(len(extremes)))

    if options.regression_mode == "same_day":
        model = regression_same_day(merged)
        beta_name = "daily_sentiment"
    elif options.regression_mode == "lagged":
        model = regression_lagged(merged, lag=1)
        beta_name = "daily_sentiment_lag_1"
    elif options.regression_mode == "multilag":
        model = regression_multilag(merged, lag_count=options.lag_count)
        beta_name = "daily_sentiment_lag_1"
    else:
        raise ValueError("regression_mode must be one of: same_day, lagged, multilag")

    reg_summary = model_summary_dict(model)
    beta = reg_summary["params"].get(beta_name)
    econ = economic_interpretation(beta)
    session.log(
        "run_regression",
        mode=options.regression_mode,
        lag_count=options.lag_count,
        r_squared=reg_summary["r_squared"],
    )

    summary_payload = {
        "session_id": session.session_id,
        "scope": options.scope,
        "query": options.query,
        "ticker": options.ticker,
        "date_range": {"start": options.start_date, "end": options.end_date},
        "correlation": correlation_result_to_dict(corr),
        "regression": reg_summary,
        "economic_interpretation": econ,
        "limitations": [
            "Correlation does not imply causation.",
            "News API coverage can be incomplete and rate-limited.",
            "Model outputs depend on data quality and omitted variables.",
        ],
        "ethical_disclaimer": "Research-only output. Not investment advice.",
    }

    save_json(out_dir / "summary.json", summary_payload)
    session.log("export_complete", output_dir=str(out_dir))
    session.save_json(out_dir / "audit_trail.json")

    return {
        "summary": summary_payload,
        "output_dir": str(out_dir),
        "session_id": session.session_id,
        "rows": {
            "news": int(len(news)),
            "daily_index": int(len(daily_index)),
            "market": int(len(market)),
            "merged": int(len(merged)),
            "extremes": int(len(extremes)),
        },
    }
