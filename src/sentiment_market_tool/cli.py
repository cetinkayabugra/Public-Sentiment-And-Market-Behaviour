from __future__ import annotations

import argparse

from sentiment_market_tool.config import assert_research_acknowledged, load_config
from sentiment_market_tool.news import validate_date
from sentiment_market_tool.pipeline import PipelineOptions, run_research_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Market Context Intelligence research CLI. Research-only tool, not financial advice."
    )

    parser.add_argument("--ack-research-only", action="store_true", help="Required ethics acknowledgement")

    parser.add_argument("--scope", choices=["global", "sector", "stock"], default="stock")
    parser.add_argument("--query", required=True, help="News query string")
    parser.add_argument("--ticker", required=True, help="Market ticker (stock or sector ETF proxy)")
    parser.add_argument("--start-date", required=True, type=validate_date, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, type=validate_date, help="YYYY-MM-DD")

    parser.add_argument("--aggregation", choices=["mean", "median", "weighted"], default="mean")
    parser.add_argument("--winsorize", action="store_true")
    parser.add_argument("--earnings-focus", action="store_true")
    parser.add_argument("--z-threshold", type=float, default=2.0)

    parser.add_argument("--corr-method", choices=["pearson", "spearman"], default="pearson")
    parser.add_argument("--rolling-method", choices=["pearson", "spearman"], default="pearson")
    parser.add_argument("--rolling-window", type=int, default=30)

    parser.add_argument("--regression-mode", choices=["same_day", "lagged", "multilag"], default="lagged")
    parser.add_argument("--lag-count", type=int, default=3)

    parser.add_argument("--join-method", choices=["inner", "left"], default="inner")
    parser.add_argument("--missing-method", choices=["drop", "ffill"], default="drop")

    parser.add_argument("--output-dir", default="outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assert_research_acknowledged(args.ack_research_only)

    config = load_config()
    options = PipelineOptions(
        scope=args.scope,
        query=args.query,
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        aggregation=args.aggregation,
        winsorize=args.winsorize,
        z_threshold=args.z_threshold,
        corr_method=args.corr_method,
        rolling_window=args.rolling_window,
        rolling_method=args.rolling_method,
        regression_mode=args.regression_mode,
        lag_count=args.lag_count,
        join_method=args.join_method,
        missing_method=args.missing_method,
        earnings_focus=args.earnings_focus,
        output_dir=args.output_dir,
    )

    result = run_research_pipeline(config=config, options=options)
    summary = result["summary"]

    print("Research pipeline completed")
    print(f"Session ID: {result['session_id']}")
    print(f"Output directory: {result['output_dir']}")
    print(f"Merged observations: {result['rows']['merged']}")
    print(f"Correlation ({summary['correlation']['method']}): {summary['correlation']['value']}")
    print(f"Regression R^2: {summary['regression']['r_squared']}")
    print("Research-only disclaimer: Not investment advice")


if __name__ == "__main__":
    main()
