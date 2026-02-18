from __future__ import annotations

from datetime import date
import io
import json
from pathlib import Path
import zipfile

from fpdf import FPDF
import pandas as pd
import streamlit as st
from statsmodels.stats.stattools import durbin_watson

from sentiment_market_tool.analytics import (
    merge_sentiment_market,
    model_summary_dict,
    regression_lagged,
    regression_multilag,
    regression_same_day,
    run_correlation,
    run_rolling_correlation,
)
from sentiment_market_tool.audit import ResearchSession
from sentiment_market_tool.config import load_config
from sentiment_market_tool.indexer import build_daily_sentiment_index
from sentiment_market_tool.market import fetch_market_data
from sentiment_market_tool.news import fetch_news
from sentiment_market_tool.sentiment import FinBERTScorer, build_article_text


STEP_LABELS = [
    "Settings — API Keys",
    "Step 0 — Welcome & Ethics Gate",
    "Step 1 — Choose Research Scope",
    "Step 2 — Data Sources & API Settings",
    "Step 3 — Fetch News (Raw Dataset)",
    "Step 4 — Run FinBERT",
    "Step 5 — Build Daily Sentiment Index",
    "Step 6 — Extreme Events (Z-Score)",
    "Step 7 — Fetch Price Data",
    "Step 8 — Merge Datasets",
    "Step 9 — Correlation Lab",
    "Step 10 — Regression Lab",
    "Step 11 — Economic Interpretation",
    "Step 12 — Export & Reproducibility",
]


def _default_state() -> dict:
    return {
        "start_date": date(2024, 1, 1),
        "end_date": date(2024, 12, 31),
        "scope": "stock",
        "region": "Global",
        "sector": "Technology",
        "earnings_focus": False,
        "news_source": "NewsAPI",
        "query_keywords": "",
        "query_exclusions": "",
        "query": "",
        "language": "en",
        "country": "global",
        "ticker": "AAPL",
        "newsapi_key": "",
        "gnews_api_key": "",
        "research_started": False,
        "research_session": None,
        "news_preview": None,
        "news_raw": None,
        "news_clean": None,
        "article_sentiment": None,
        "daily_index": None,
        "extreme_days": None,
        "market_data": None,
        "merged_data": None,
        "correlation_result": None,
        "rolling_result": None,
        "regression_summary": None,
        "regression_econ": None,
        "narrative_text": "",
    }


def init_state() -> None:
    for key, value in _default_state().items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_session() -> ResearchSession:
    if st.session_state["research_session"] is None:
        st.session_state["research_session"] = ResearchSession()
    return st.session_state["research_session"]


def log_event(step: str, **details: object) -> None:
    session = get_session()
    session.log(step, **details)


def get_config_or_error() -> tuple[bool, object]:
    key_from_settings = str(st.session_state.get("newsapi_key", "")).strip()
    if key_from_settings:
        return True, key_from_settings

    try:
        config = load_config()
    except Exception as exc:
        st.error(str(exc))
        return False, None
    return True, config.newsapi_key


def set_env_var(var_name: str, var_value: str, env_path: Path = Path(".env")) -> None:
    lines: list[str] = []
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()

    updated = False
    new_lines: list[str] = []
    for line in lines:
        if line.startswith(f"{var_name}="):
            new_lines.append(f"{var_name}={var_value}")
            updated = True
        else:
            new_lines.append(line)

    if not updated:
        new_lines.append(f"{var_name}={var_value}")

    env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def date_range_days() -> int:
    start = pd.to_datetime(st.session_state["start_date"])
    end = pd.to_datetime(st.session_state["end_date"])
    return int((end - start).days + 1)


def coverage_score(frame: pd.DataFrame) -> float:
    if frame is None or frame.empty:
        return 0.0
    unique_days = frame["date"].nunique()
    total_days = max(date_range_days(), 1)
    return float(unique_days / total_days)


def build_query() -> str:
    include_part = (st.session_state["query_keywords"] or "").strip()
    exclude_part = (st.session_state["query_exclusions"] or "").strip()
    if include_part and exclude_part:
        return f"{include_part} -{exclude_part}"
    if include_part:
        return include_part
    return st.session_state["ticker"]


def generic_filter(frame: pd.DataFrame) -> pd.DataFrame:
    generic_patterns = [
        "market update",
        "stocks to watch",
        "live updates",
        "breaking news",
        "top stories",
    ]
    title_series = frame["title"].fillna("").str.lower()
    mask = title_series.apply(lambda title: any(pattern in title for pattern in generic_patterns))
    return frame.loc[~mask].reset_index(drop=True)


def sentiment_extremes_from_index(index_frame: pd.DataFrame, z_threshold: float) -> pd.DataFrame:
    frame = index_frame.copy()
    std_value = frame["daily_sentiment"].std(ddof=0)
    if pd.isna(std_value) or std_value == 0:
        frame["sentiment_z"] = 0.0
    else:
        frame["sentiment_z"] = (frame["daily_sentiment"] - frame["daily_sentiment"].mean()) / std_value
    return frame.loc[frame["sentiment_z"].abs() >= z_threshold].reset_index(drop=True)


def extreme_top_headlines(extreme_frame: pd.DataFrame, article_frame: pd.DataFrame) -> pd.DataFrame:
    if extreme_frame is None or extreme_frame.empty or article_frame is None or article_frame.empty:
        return pd.DataFrame(columns=["date", "top_headlines"])

    rows: list[dict[str, object]] = []
    for current_date in extreme_frame["date"]:
        day_articles = article_frame.loc[article_frame["date"] == current_date].copy()
        day_articles["abs_score"] = day_articles["sentiment_score"].abs()
        top_titles = day_articles.sort_values("abs_score", ascending=False)["title"].fillna("").head(3).tolist()
        rows.append({"date": current_date, "top_headlines": " | ".join(top_titles)})
    return pd.DataFrame(rows)


def interpret_correlation(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "Insufficient data to interpret correlation."
    magnitude = abs(value)
    if magnitude >= 0.7:
        strength = "strong"
    elif magnitude >= 0.4:
        strength = "moderate"
    elif magnitude >= 0.2:
        strength = "weak"
    else:
        strength = "very weak"
    direction = "positive" if value >= 0 else "negative"
    return f"{strength.capitalize()} {direction} association in this sample."


def regression_interpretation(regression_summary: dict | None) -> dict[str, object]:
    if not regression_summary:
        return {"beta": None, "statement": "No regression results yet."}

    candidate_names = [
        "daily_sentiment",
        "daily_sentiment_lag_1",
        "daily_sentiment_lag_2",
        "daily_sentiment_lag_3",
        "daily_sentiment_lag_4",
        "daily_sentiment_lag_5",
    ]
    params = regression_summary.get("params", {})
    beta = None
    for name in candidate_names:
        if name in params:
            beta = float(params[name])
            break

    if beta is None:
        return {"beta": None, "statement": "No sentiment coefficient found in model output."}

    effect = beta * 0.10
    return {
        "beta": beta,
        "statement": f"A +0.10 change in sentiment is associated with {effect:.4f} change in log return.",
    }


def dump_state_payload() -> dict:
    session = get_session()
    return {
        "settings": {
            "start_date": str(st.session_state["start_date"]),
            "end_date": str(st.session_state["end_date"]),
            "scope": st.session_state["scope"],
            "region": st.session_state["region"],
            "sector": st.session_state["sector"],
            "earnings_focus": st.session_state["earnings_focus"],
            "news_source": st.session_state["news_source"],
            "query_keywords": st.session_state["query_keywords"],
            "query_exclusions": st.session_state["query_exclusions"],
            "query": st.session_state["query"],
            "language": st.session_state["language"],
            "country": st.session_state["country"],
            "ticker": st.session_state["ticker"],
            "newsapi_key": st.session_state["newsapi_key"],
            "gnews_api_key": st.session_state["gnews_api_key"],
        },
        "results": {
            "correlation_result": st.session_state["correlation_result"],
            "regression_summary": st.session_state["regression_summary"],
            "regression_econ": st.session_state["regression_econ"],
            "narrative_text": st.session_state["narrative_text"],
        },
        "audit": session.to_dict(),
    }


def load_state_payload(payload: dict) -> None:
    settings = payload.get("settings", {})
    for key, value in settings.items():
        if key in {"start_date", "end_date"}:
            st.session_state[key] = pd.to_datetime(value).date()
        else:
            st.session_state[key] = value

    results = payload.get("results", {})
    st.session_state["correlation_result"] = results.get("correlation_result")
    st.session_state["regression_summary"] = results.get("regression_summary")
    st.session_state["regression_econ"] = results.get("regression_econ")
    st.session_state["narrative_text"] = results.get("narrative_text", "")

    audit = payload.get("audit")
    if audit:
        session = ResearchSession()
        session.session_id = audit.get("session_id", session.session_id)
        session.created_at = audit.get("created_at", session.created_at)
        for event in audit.get("events", []):
            session.log(event.get("step", "loaded_event"), **event.get("details", {}))
        st.session_state["research_session"] = session
        st.session_state["research_started"] = True


def add_dataframe_to_zip(zip_file: zipfile.ZipFile, name: str, frame: pd.DataFrame | None) -> None:
    if frame is None:
        return
    csv_bytes = frame.to_csv(index=False).encode("utf-8")
    zip_file.writestr(name, csv_bytes)


def build_pdf_summary() -> bytes:
    correlation = st.session_state.get("correlation_result") or {}
    regression = st.session_state.get("regression_summary") or {}
    narrative = st.session_state.get("narrative_text") or ""

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    lines = [
        "Market Context Intelligence Model",
        "Research Summary Report",
        "",
        f"Scope: {st.session_state['scope']}",
        f"Query: {st.session_state.get('query', '')}",
        f"Ticker: {st.session_state.get('ticker', '')}",
        f"Date Range: {st.session_state['start_date']} to {st.session_state['end_date']}",
        "",
        f"Correlation Method: {correlation.get('method')}",
        f"Correlation Value: {correlation.get('value')}",
        f"Correlation p-value: {correlation.get('p_value')}",
        "",
        f"Regression R^2: {regression.get('r_squared')}",
        "",
        "Narrative:",
        narrative or "No narrative available.",
        "",
        "Disclaimer: Research-only output. Not investment advice.",
    ]

    for line in lines:
        pdf.multi_cell(0, 7, txt=str(line))

    return bytes(pdf.output(dest="S"))


def render_top_shell() -> None:
    st.title("Market Context Intelligence Model — Research Session")

    col_a, col_b, col_c, col_d = st.columns([1.2, 1.2, 1.5, 1.5])
    with col_a:
        st.session_state["start_date"] = st.date_input("Start Date", value=st.session_state["start_date"])
    with col_b:
        st.session_state["end_date"] = st.date_input("End Date", value=st.session_state["end_date"])
    with col_c:
        payload_bytes = json.dumps(dump_state_payload(), indent=2).encode("utf-8")
        st.download_button(
            "Save Session",
            data=payload_bytes,
            file_name="research_session.json",
            mime="application/json",
            use_container_width=True,
        )
    with col_d:
        uploaded = st.file_uploader("Load Session", type=["json"], accept_multiple_files=False)
        if uploaded is not None:
            try:
                payload = json.loads(uploaded.read().decode("utf-8"))
                load_state_payload(payload)
                st.success("Session loaded")
            except Exception as exc:
                st.error(f"Failed to load session: {exc}")


def render_settings_page() -> None:
    st.subheader("Settings — API Keys")
    st.caption("Keys are used for research workflow only and never for trading recommendations.")

    st.session_state["newsapi_key"] = st.text_input(
        "NewsAPI Key",
        value=st.session_state.get("newsapi_key", ""),
        type="password",
        help="Required for fetching news in Step 2 and Step 3.",
    )

    st.session_state["gnews_api_key"] = st.text_input(
        "GNews Key (optional)",
        value=st.session_state.get("gnews_api_key", ""),
        type="password",
        help="Reserved for future source integration.",
    )

    persist_to_env = st.checkbox("Persist NewsAPI key to .env", value=False)

    if st.button("Save API Settings", type="primary"):
        newsapi_key = str(st.session_state.get("newsapi_key", "")).strip()
        if not newsapi_key:
            st.error("NewsAPI key is empty. Please add a valid key.")
            return

        if persist_to_env:
            try:
                set_env_var("NEWSAPI_KEY", newsapi_key)
                st.success("API settings saved and NEWSAPI_KEY persisted to .env")
            except Exception as exc:
                st.error(f"Failed to persist .env: {exc}")
                return
        else:
            st.success("API settings saved for current session")

        log_event("save_api_settings", persisted_to_env=persist_to_env)

    key_set = bool(str(st.session_state.get("newsapi_key", "")).strip())
    st.write(f"NewsAPI key status: {'Configured' if key_set else 'Missing'}")
    st.info("If no key is entered here, the app falls back to NEWSAPI_KEY from your environment/.env.")


def render_step_0() -> None:
    st.subheader("Welcome & Ethics Gate")
    ack_research = st.checkbox("I understand this tool is for research/education, not financial advice.")
    ack_limits = st.checkbox("I understand data sources may be incomplete and rate-limited.")

    if st.button("Start Research", type="primary"):
        if not ack_research or not ack_limits:
            st.error("Both acknowledgements are required to continue.")
        else:
            st.session_state["research_started"] = True
            st.session_state["research_session"] = ResearchSession()
            log_event("start_research", acknowledged_research_only=True, acknowledged_limits=True)
            st.success("Research session started")

    if st.session_state["research_started"]:
        st.info(f"Research Session ID: {get_session().session_id}")


def render_step_1() -> None:
    st.subheader("Choose Research Scope (Hierarchy)")
    st.session_state["scope"] = st.radio("Scope", ["global", "sector", "stock"], horizontal=True)
    st.session_state["region"] = st.selectbox("Region", ["US", "EU", "UK", "Global"], index=3)
    st.session_state["sector"] = st.selectbox(
        "Sector",
        ["Technology", "Healthcare", "Financials", "Energy", "Industrials", "Consumer Discretionary"],
        index=0,
    )
    st.session_state["earnings_focus"] = st.toggle("Earnings focus", value=st.session_state["earnings_focus"])

    if st.button("Save Scope"):
        log_event(
            "scope_selected",
            scope=st.session_state["scope"],
            region=st.session_state["region"],
            sector=st.session_state["sector"],
            earnings_focus=st.session_state["earnings_focus"],
        )
        st.success("Scope saved")

    st.markdown(
        f"**Summary**  \\n+Scope: `{st.session_state['scope']}`  \\n+Region: `{st.session_state['region']}`  \\n+Sector: `{st.session_state['sector']}`  \\n+Date range: `{st.session_state['start_date']}` → `{st.session_state['end_date']}`  \\n+Earnings focus: `{st.session_state['earnings_focus']}`"
    )


def render_step_2() -> None:
    st.subheader("Data Sources & API Settings")

    st.session_state["news_source"] = st.selectbox("News source", ["NewsAPI", "GNews", "RSS"])
    if st.session_state["news_source"] != "NewsAPI":
        st.warning("Current backend supports NewsAPI. For now, use NewsAPI or map your feed to NewsAPI-compatible input.")

    st.session_state["query_keywords"] = st.text_input(
        "Sector/stock keywords",
        value=st.session_state["query_keywords"],
        placeholder="e.g., healthcare earnings guidance",
    )
    st.session_state["query_exclusions"] = st.text_input(
        "Exclusion keywords",
        value=st.session_state["query_exclusions"],
        placeholder="e.g., sports",
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.session_state["language"] = st.selectbox("Language", ["en", "de", "fr", "es"], index=0)
    with col_b:
        st.session_state["country"] = st.selectbox("Country", ["global", "us", "gb", "de", "fr"], index=0)

    st.session_state["query"] = build_query()
    st.code(f"Query: {st.session_state['query']}")

    if st.button("Test Query"):
        ok, newsapi_key = get_config_or_error()
        if ok:
            with st.spinner("Testing query..."):
                preview = fetch_news(
                    api_key=newsapi_key,
                    query=st.session_state["query"],
                    start_date=str(st.session_state["start_date"]),
                    end_date=str(st.session_state["end_date"]),
                    language=st.session_state["language"],
                    page_size=20,
                    max_pages=1,
                )
                st.session_state["news_preview"] = preview
                log_event("test_query", query=st.session_state["query"], rows=int(len(preview)))

    preview_frame = st.session_state.get("news_preview")
    if preview_frame is not None:
        st.dataframe(preview_frame[["date", "source", "title", "url"]].head(10), use_container_width=True)
        duplicate_count = int(preview_frame.duplicated(subset=["url"]).sum())
        score = coverage_score(preview_frame)
        st.write(f"Duplicate count: `{duplicate_count}`")
        st.write(f"Coverage score: `{score:.2%}` of days with at least one article")


def render_step_3() -> None:
    st.subheader("Fetch News (Raw Dataset)")
    max_pages = st.slider("Max pages", min_value=1, max_value=20, value=5)

    if st.button("Fetch News", type="primary"):
        ok, newsapi_key = get_config_or_error()
        if ok:
            with st.spinner("Fetching news..."):
                raw_news = fetch_news(
                    api_key=newsapi_key,
                    query=st.session_state["query"] or build_query(),
                    start_date=str(st.session_state["start_date"]),
                    end_date=str(st.session_state["end_date"]),
                    language=st.session_state["language"],
                    page_size=100,
                    max_pages=max_pages,
                )
                if st.session_state["earnings_focus"]:
                    raw_news = raw_news.loc[raw_news["is_earnings_related"]].reset_index(drop=True)
                st.session_state["news_raw"] = raw_news
                st.session_state["news_clean"] = raw_news.copy()
                log_event("fetch_news", rows=int(len(raw_news)), earnings_focus=st.session_state["earnings_focus"])

    raw_news = st.session_state.get("news_raw")
    if raw_news is not None:
        st.dataframe(raw_news[["date", "source", "title", "description", "url"]].head(100), use_container_width=True)

        all_days = pd.date_range(st.session_state["start_date"], st.session_state["end_date"], freq="D")
        observed_days = pd.to_datetime(raw_news["date"]).dt.normalize().unique()
        missing_days = len(set(all_days) - set(observed_days))
        duplicate_urls = int(raw_news.duplicated(subset=["url"]).sum())
        st.write(f"Missing dates: `{missing_days}`")
        st.write(f"Duplicate articles by URL: `{duplicate_urls}`")

        apply_generic_filter = st.checkbox("Filter overly generic articles", value=False)
        if st.button("Clean & Continue"):
            cleaned = raw_news.copy()
            if apply_generic_filter:
                cleaned = generic_filter(cleaned)
            st.session_state["news_clean"] = cleaned
            log_event("clean_news", raw_rows=int(len(raw_news)), clean_rows=int(len(cleaned)))
            st.success(f"Cleaned dataset ready: {len(cleaned)} rows")


def render_step_4() -> None:
    st.subheader("Run FinBERT (Article-Level Sentiment)")
    model_name = st.text_input("Model version", value="ProsusAI/finbert")
    batch_size = st.slider("Batch size", min_value=4, max_value=64, value=16, step=4)

    if st.button("Run FinBERT", type="primary"):
        clean_news = st.session_state.get("news_clean")
        if clean_news is None or clean_news.empty:
            st.error("No cleaned news dataset available. Complete Step 3 first.")
            return

        with st.spinner("Running FinBERT..."):
            scorer = FinBERTScorer(model_name=model_name, batch_size=batch_size)
            texts = build_article_text(clean_news)
            sentiment_frame = scorer.score(texts)
            article_sentiment = clean_news.reset_index(drop=True).join(sentiment_frame)
            st.session_state["article_sentiment"] = article_sentiment
            log_event("run_finbert", rows=int(len(article_sentiment)), model_name=model_name, batch_size=batch_size)

    article_sentiment = st.session_state.get("article_sentiment")
    if article_sentiment is not None:
        st.dataframe(
            article_sentiment[
                ["date", "title", "sentiment_score", "sentiment_label", "confidence", "p_positive", "p_neutral", "p_negative"]
            ].head(100),
            use_container_width=True,
        )
        with st.expander("What this does"):
            st.markdown(
                "FinBERT estimates financial sentiment probabilities. This tool computes:  \n"
                "`Sentiment = Ppositive - Pnegative` with range from `-1` to `+1`."
            )


def render_step_5() -> None:
    st.subheader("Build Daily Sentiment Index")
    aggregation = st.selectbox("Aggregation", ["mean", "median", "weighted"], index=0)
    winsorize = st.toggle("Winsorize outliers", value=False)

    if st.button("Build Index", type="primary"):
        article_sentiment = st.session_state.get("article_sentiment")
        if article_sentiment is None or article_sentiment.empty:
            st.error("No article sentiment available. Complete Step 4 first.")
            return

        index_frame = build_daily_sentiment_index(article_sentiment, method=aggregation, winsorize=winsorize)
        st.session_state["daily_index"] = index_frame
        log_event("build_index", method=aggregation, winsorize=winsorize, rows=int(len(index_frame)))

    index_frame = st.session_state.get("daily_index")
    if index_frame is not None:
        chart_frame = index_frame.set_index("date")[["daily_sentiment"]]
        st.line_chart(chart_frame)
        st.dataframe(index_frame.head(100), use_container_width=True)
        score = coverage_score(index_frame)
        st.progress(min(max(int(score * 100), 0), 100), text=f"Coverage: {score:.2%}")


def render_step_6() -> None:
    st.subheader("Extreme Events (Z-Score)")
    standardize = st.toggle("Standardize sentiment", value=True)
    z_threshold = st.slider("Z-threshold", min_value=1.0, max_value=4.0, value=2.0, step=0.1)

    if st.button("Detect Extremes", type="primary"):
        index_frame = st.session_state.get("daily_index")
        article_sentiment = st.session_state.get("article_sentiment")
        if index_frame is None or index_frame.empty:
            st.error("No daily sentiment index available. Complete Step 5 first.")
            return

        if not standardize:
            st.warning("Standardization is required by methodology. Please enable 'Standardize sentiment'.")
            return

        extreme_days = sentiment_extremes_from_index(index_frame, z_threshold=z_threshold)
        headline_frame = extreme_top_headlines(extreme_days, article_sentiment)
        output = extreme_days.merge(headline_frame, on="date", how="left")
        st.session_state["extreme_days"] = output
        log_event("detect_extremes", threshold=z_threshold, rows=int(len(output)))

    extreme_days = st.session_state.get("extreme_days")
    index_frame = st.session_state.get("daily_index")
    if index_frame is not None and extreme_days is not None:
        overlay_frame = index_frame[["date", "daily_sentiment"]].copy()
        overlay_frame["extreme_points"] = pd.NA
        if not extreme_days.empty:
            extreme_lookup = extreme_days.set_index("date")["daily_sentiment"].to_dict()
            overlay_frame["extreme_points"] = overlay_frame["date"].map(extreme_lookup)
        st.line_chart(overlay_frame.set_index("date")[["daily_sentiment", "extreme_points"]])
        st.dataframe(extreme_days.head(100), use_container_width=True)


def render_step_7() -> None:
    st.subheader("Fetch Price Data (Yahoo Finance)")
    target_type = st.selectbox(
        "Target",
        ["Sector ETF proxy", "Specific ticker", "Market benchmark"],
        index=1,
    )
    default_ticker = st.session_state.get("ticker", "AAPL")
    st.session_state["ticker"] = st.text_input("Ticker", value=default_ticker)

    if st.button("Fetch Prices", type="primary"):
        with st.spinner("Fetching market prices..."):
            market_frame = fetch_market_data(
                ticker=st.session_state["ticker"],
                start_date=str(st.session_state["start_date"]),
                end_date=str(st.session_state["end_date"]),
            )
            st.session_state["market_data"] = market_frame
            log_event("fetch_prices", target_type=target_type, ticker=st.session_state["ticker"], rows=int(len(market_frame)))

    market_frame = st.session_state.get("market_data")
    if market_frame is not None:
        st.dataframe(market_frame.head(100), use_container_width=True)
        st.line_chart(market_frame.set_index("date")[["log_return"]])

        business_days = pd.bdate_range(st.session_state["start_date"], st.session_state["end_date"])
        missing = len(set(business_days) - set(pd.to_datetime(market_frame["date"])))
        st.write(f"Trading days count: `{len(market_frame)}`")
        st.write(f"Missing business days: `{missing}`")


def render_step_8() -> None:
    st.subheader("Merge Datasets on Date")
    join_method = st.selectbox("Merge method", ["inner", "left"], index=0)
    missing_method = st.radio("Missing handling", ["drop", "ffill"], index=0, horizontal=True)

    if st.button("Merge", type="primary"):
        daily_index = st.session_state.get("daily_index")
        market_frame = st.session_state.get("market_data")
        if daily_index is None or daily_index.empty:
            st.error("No daily sentiment index available. Complete Step 5 first.")
            return
        if market_frame is None or market_frame.empty:
            st.error("No market data available. Complete Step 7 first.")
            return

        merged = merge_sentiment_market(daily_sentiment=daily_index, market=market_frame, join=join_method, missing=missing_method)
        st.session_state["merged_data"] = merged
        drop_pct = 100.0 * (1 - (len(merged) / max(len(daily_index), 1)))
        log_event(
            "merge_datasets",
            join_method=join_method,
            missing_method=missing_method,
            before_rows=int(len(daily_index)),
            after_rows=int(len(merged)),
            dropped_pct=drop_pct,
        )

    merged = st.session_state.get("merged_data")
    daily_index = st.session_state.get("daily_index")
    if merged is not None and daily_index is not None:
        st.dataframe(merged.head(100), use_container_width=True)
        rows_before = len(daily_index)
        rows_after = len(merged)
        dropped_pct = 100.0 * (1 - (rows_after / max(rows_before, 1)))
        st.write(f"Rows before merge: `{rows_before}`")
        st.write(f"Rows after merge: `{rows_after}`")
        st.write(f"Dropped days: `{dropped_pct:.2f}%`")


def render_step_9() -> None:
    st.subheader("Correlation Lab")
    method = st.selectbox("Correlation type", ["pearson", "spearman"], index=0)
    rolling_enabled = st.toggle("Enable rolling correlation", value=True)
    rolling_window = st.select_slider("Rolling window", options=[7, 14, 30, 60], value=30)

    if st.button("Run Correlation", type="primary"):
        merged = st.session_state.get("merged_data")
        if merged is None or merged.empty:
            st.error("No merged dataset available. Complete Step 8 first.")
            return

        correlation = run_correlation(merged, method=method)
        st.session_state["correlation_result"] = {
            "method": correlation.method,
            "value": correlation.value,
            "p_value": correlation.p_value,
            "observations": correlation.observations,
        }

        if rolling_enabled:
            rolling = run_rolling_correlation(merged, window=rolling_window, method=method)
            st.session_state["rolling_result"] = rolling
        else:
            st.session_state["rolling_result"] = None

        log_event("run_correlation", method=method, rolling_enabled=rolling_enabled, window=rolling_window)

    correlation = st.session_state.get("correlation_result")
    rolling = st.session_state.get("rolling_result")
    if correlation is not None:
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Correlation", f"{correlation['value']:.4f}" if correlation["value"] is not None else "N/A")
        col_b.metric("p-value", f"{correlation['p_value']:.4g}" if correlation["p_value"] is not None else "N/A")
        col_c.metric("Observations", str(correlation["observations"]))
        st.info(interpret_correlation(correlation["value"]))

        if rolling is not None:
            st.line_chart(rolling.set_index("date")[["rolling_corr"]])
            valid = rolling.dropna(subset=["rolling_corr"]).copy()
            if not valid.empty:
                max_row = valid.loc[valid["rolling_corr"].idxmax()]
                min_row = valid.loc[valid["rolling_corr"].idxmin()]
                st.write(
                    "Regime notes: highest rolling correlation on "
                    f"`{pd.to_datetime(max_row['date']).date()}` ({max_row['rolling_corr']:.4f}), "
                    f"lowest on `{pd.to_datetime(min_row['date']).date()}` ({min_row['rolling_corr']:.4f})."
                )


def render_step_10() -> None:
    st.subheader("Regression Lab")
    model_choice = st.selectbox("Model choice", ["same_day", "lagged", "multilag"], index=1)
    lag_count = st.slider("Lag count (for multilag)", min_value=1, max_value=5, value=3)

    if st.button("Run Regression", type="primary"):
        merged = st.session_state.get("merged_data")
        if merged is None or merged.empty:
            st.error("No merged dataset available. Complete Step 8 first.")
            return

        if model_choice == "same_day":
            model = regression_same_day(merged)
        elif model_choice == "lagged":
            model = regression_lagged(merged, lag=1)
        else:
            model = regression_multilag(merged, lag_count=lag_count)

        summary = model_summary_dict(model)
        st.session_state["regression_summary"] = summary
        interpretation = regression_interpretation(summary)
        st.session_state["regression_econ"] = interpretation

        residual_frame = pd.DataFrame(
            {
                "fitted": model.fittedvalues,
                "residual": model.resid,
            }
        )
        st.session_state["regression_residuals"] = residual_frame

        log_event("run_regression", model_choice=model_choice, lag_count=lag_count, r_squared=summary.get("r_squared"))

    summary = st.session_state.get("regression_summary")
    if summary is not None:
        table_rows = []
        for name, coefficient in summary["params"].items():
            table_rows.append(
                {
                    "term": name,
                    "coef": coefficient,
                    "t_stat": summary["t_values"].get(name),
                    "p_value": summary["p_values"].get(name),
                }
            )
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)
        st.write(f"R²: `{summary['r_squared']:.4f}`")

        residual_frame = st.session_state.get("regression_residuals")
        if residual_frame is not None:
            st.scatter_chart(residual_frame, x="fitted", y="residual")
            dw_stat = float(durbin_watson(residual_frame["residual"]))
            if dw_stat < 1.5 or dw_stat > 2.5:
                st.warning(f"Autocorrelation warning: Durbin-Watson = {dw_stat:.3f}")
            else:
                st.success(f"Residual autocorrelation check is acceptable: Durbin-Watson = {dw_stat:.3f}")

        interpretation = st.session_state.get("regression_econ")
        if interpretation:
            st.info(str(interpretation.get("statement", "")))


def render_step_11() -> None:
    st.subheader("Economic Interpretation & Narrative Builder")
    correlation = st.session_state.get("correlation_result")
    regression = st.session_state.get("regression_summary")
    regression_text = st.session_state.get("regression_econ", {}).get("statement", "")

    auto_lines = [
        "Correlation findings:",
        f"- Method: {correlation.get('method') if correlation else 'N/A'}",
        f"- Value: {correlation.get('value') if correlation else 'N/A'}",
        f"- p-value: {correlation.get('p_value') if correlation else 'N/A'}",
        "",
        "Regression findings:",
        f"- R²: {regression.get('r_squared') if regression else 'N/A'}",
        f"- Economic interpretation: {regression_text or 'N/A'}",
        "",
        "Limitations: API coverage, omitted variables, and correlation does not imply causation.",
    ]
    default_text = "\n".join(auto_lines)

    if not st.session_state["narrative_text"]:
        st.session_state["narrative_text"] = default_text

    st.session_state["narrative_text"] = st.text_area(
        "Editable research summary",
        value=st.session_state["narrative_text"],
        height=220,
    )

    confidence_tag = "Weak evidence"
    if correlation and correlation.get("p_value") is not None and correlation["p_value"] < 0.05:
        value = abs(correlation.get("value", 0.0) or 0.0)
        if value >= 0.5:
            confidence_tag = "Strong evidence"
        else:
            confidence_tag = "Moderate evidence"

    st.write(f"Confidence level: **{confidence_tag}**")
    st.caption("Research-only disclaimer: This output is not financial advice.")

    if st.button("Save Narrative"):
        log_event("save_narrative", length=len(st.session_state["narrative_text"]))
        st.success("Narrative saved")


def render_step_12() -> None:
    st.subheader("Export & Reproducibility")

    export_raw = st.checkbox("Include raw news CSV", value=True)
    export_article = st.checkbox("Include article sentiment CSV", value=True)
    export_daily = st.checkbox("Include daily sentiment index CSV", value=True)
    export_merged = st.checkbox("Include merged dataset CSV", value=True)
    export_report_pdf = st.checkbox("Include PDF summary", value=True)
    export_session_json = st.checkbox("Include session JSON", value=True)

    if st.button("Prepare Export", type="primary"):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            if export_raw:
                add_dataframe_to_zip(zip_file, "raw_news.csv", st.session_state.get("news_raw"))
            if export_article:
                add_dataframe_to_zip(zip_file, "article_sentiment.csv", st.session_state.get("article_sentiment"))
            if export_daily:
                add_dataframe_to_zip(zip_file, "daily_sentiment_index.csv", st.session_state.get("daily_index"))
            if export_merged:
                add_dataframe_to_zip(zip_file, "merged_dataset.csv", st.session_state.get("merged_data"))

            if export_report_pdf:
                zip_file.writestr("research_summary.pdf", build_pdf_summary())

            if export_session_json:
                zip_file.writestr("session.json", json.dumps(dump_state_payload(), indent=2).encode("utf-8"))

            audit_payload = get_session().to_dict()
            zip_file.writestr("audit_trail.json", json.dumps(audit_payload, indent=2).encode("utf-8"))

        st.session_state["export_bytes"] = buffer.getvalue()
        log_event("prepare_export")
        st.success("Export package is ready")

    export_bytes = st.session_state.get("export_bytes")
    if export_bytes:
        st.download_button(
            "Download Export ZIP",
            data=export_bytes,
            file_name="research_export_bundle.zip",
            mime="application/zip",
            use_container_width=True,
        )

    st.info("Always include data limitations and remember: correlation ≠ causation.")


def render_run_log() -> None:
    st.markdown("---")
    st.subheader("Run Log / Audit Trail")
    session = st.session_state.get("research_session")
    if session is None or not session.events:
        st.caption("No audit events yet.")
        return

    rows = [
        {
            "timestamp": event.timestamp,
            "step": event.step,
            "details": json.dumps(event.details),
        }
        for event in session.events
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Market Context Intelligence UI", layout="wide")
    init_state()
    render_top_shell()

    selected_step = st.sidebar.radio("Steps Navigation", STEP_LABELS)
    st.sidebar.caption("Research-only interface. No buy/sell recommendations.")

    if selected_step == STEP_LABELS[0]:
        render_settings_page()
    elif selected_step == STEP_LABELS[1]:
        render_step_0()
    elif selected_step == STEP_LABELS[2]:
        render_step_1()
    elif selected_step == STEP_LABELS[3]:
        render_step_2()
    elif selected_step == STEP_LABELS[4]:
        render_step_3()
    elif selected_step == STEP_LABELS[5]:
        render_step_4()
    elif selected_step == STEP_LABELS[6]:
        render_step_5()
    elif selected_step == STEP_LABELS[7]:
        render_step_6()
    elif selected_step == STEP_LABELS[8]:
        render_step_7()
    elif selected_step == STEP_LABELS[9]:
        render_step_8()
    elif selected_step == STEP_LABELS[10]:
        render_step_9()
    elif selected_step == STEP_LABELS[11]:
        render_step_10()
    elif selected_step == STEP_LABELS[12]:
        render_step_11()
    elif selected_step == STEP_LABELS[13]:
        render_step_12()

    render_run_log()


if __name__ == "__main__":
    main()
