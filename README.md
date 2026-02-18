# Market Context Intelligence Model  
## Research Tool for Understanding the Relationship Between Public Sentiment and Market Behaviour

---

## 1. Project Overview

This project investigates whether financial news sentiment — extracted using FinBERT — aligns with and explains observable financial market behaviour.

Instead of treating each stock independently, this model assumes sentiment is hierarchical:

Global Sentiment → Sector Sentiment → Stock Sentiment

Earnings reports are treated as high-impact sentiment triggers.

The system is designed as an interactive research tool that:

- Builds a structured sentiment index
- Tests correlation with financial returns
- Runs regression models
- Interprets economic significance
- Maintains ethical transparency

This tool is strictly for academic and research purposes.

---

## 2. Research Objectives

1. Determine whether sector-level sentiment aligns with sector returns.
2. Evaluate whether sentiment leads or reacts to returns.
3. Measure time-varying relationships using rolling correlation.
4. Test whether earnings-related sentiment events create abnormal returns.
5. Assess economic significance beyond statistical significance.

---

## 3. Data Pipeline Architecture

News API → FinBERT → Daily Sentiment Index  
Yahoo Finance → Daily Log Returns  
Merge on Date → Correlation / Regression  

---

## 4. Sentiment Index Construction

### 4.1 Article-Level Sentiment

FinBERT returns:
- P(positive)
- P(neutral)
- P(negative)

Sentiment score is computed as:

Sentiment = P_positive − P_negative

Range: -1 to +1

---

### 4.2 Daily Aggregation

Daily Sentiment Index:

DailySentiment_t = Mean(Sentiment_i)

Alternative aggregation options:
- Median
- Confidence-weighted mean
- Winsorized mean (outlier control)

The index can be computed at three levels:

- Global market sentiment
- Sector sentiment
- Individual stock sentiment

---

## 5. Financial Performance Measurement

Daily log returns are calculated as:

Return_t = ln(Price_t / Price_{t-1})

Log returns are preferred because:
- They are additive over time
- They improve statistical properties
- They are standard in financial econometrics

---

## 6. Correlation Analysis

### 6.1 Pearson Correlation

Measures linear relationship between sentiment and returns.

Used when:
- Data is approximately normal
- Relationship is expected to be linear

---

### 6.2 Spearman Correlation

Rank-based correlation.

Used when:
- Data is non-normal
- Relationship is monotonic but not strictly linear
- Outliers are present

---

### 6.3 Rolling Correlation

Instead of one static number:

RollingCorr_t = Corr(Sentiment_{t−window:t}, Return_{t−window:t})

Purpose:
- Detect regime changes
- Identify crisis sensitivity
- Observe time-varying impact of sentiment

---

## 7. Regression Models

### 7.1 Baseline Model

Return_t = α + β × Sentiment_t + ε_t

Interpretation:
- α = baseline return
- β = sentiment impact coefficient
- ε = unexplained variation

If β is statistically significant:
Sentiment is associated with return variation.

---

### 7.2 Lagged Model (Predictive Test)

Return_t = α + β × Sentiment_{t−1} + ε_t

Tests whether sentiment predicts future returns.

---

### 7.3 Multi-Layer Hierarchical Model

Return_t = α  
           + β1 × GlobalSent_t  
           + β2 × SectorSent_t  
           + β3 × StockSent_t  
           + ε_t

Purpose:
Identify which sentiment layer drives performance.

---

## 8. Extreme Sentiment Detection

Sentiment is standardized:

Z_t = (Sentiment_t − μ) / σ

Extreme thresholds:
- Z > +2 → extreme optimism
- Z < −2 → extreme pessimism

Used to detect:
- Sentiment shocks
- Earnings spikes
- Crisis reactions

---

## 9. Economic Significance

Statistical significance does not imply economic relevance.

Example:

If β = 0.02  
A 0.10 increase in sentiment  
→ 0.002 (0.2%) change in daily return  

In financial markets, this magnitude may be economically meaningful when compounded.

---

## 10. System Architecture

Core Components:

- News Data Collector
- FinBERT Sentiment Engine
- Sentiment Index Builder
- Financial Data Fetcher
- Merge & Validation Engine
- Correlation Lab
- Regression Lab
- Z-Score Analyzer
- Export & Audit Module

The system maintains reproducibility by logging:
- Data sources
- Date ranges
- Model parameters
- Output statistics

---

## 11. Ethical Framework

This tool:

- Does NOT generate buy/sell signals
- Does NOT provide financial advice
- Maintains transparency in methodology
- Clearly states data limitations
- Separates statistical results from investment recommendations

Users must acknowledge research-only usage before running analysis.

---

## 12. Limitations

- Free News APIs may have rate limits.
- Historical news coverage may be incomplete.
- FinBERT download required on first run.
- Omitted variables may affect regression results.
- Correlation does not imply causation.

---

## 13. Expected Contribution

This project moves beyond simple sentiment classification by:

- Introducing hierarchical sentiment modeling
- Measuring regime-dependent impact
- Testing lagged predictive effects
- Distinguishing statistical vs economic significance
- Embedding transparency and reproducibility into UI

The result is a Market-Context Intelligence Model rather than a standalone NLP classifier.

---

## 14. Future Extensions

- Granger causality testing
- Volatility modeling (GARCH)
- Event study around earnings announcements
- Multi-factor regression including macro controls
- Backtesting framework (research-only)

---

## 15. Disclaimer

This project is intended solely for academic research and educational purposes.  
It does not constitute investment advice, financial recommendation, or trading guidance.

---

## 16. Implementation (Python CLI)

This repository includes a Python implementation aligned with the methodology above.

### Installation

```bash
pip install -e .
```

Set environment variable:

```bash
NEWSAPI_KEY=your_newsapi_key
```

### Run Research Pipeline

```bash
market-context-tool \
    --ack-research-only \
    --scope sector \
    --query "healthcare earnings" \
    --ticker XLV \
    --start-date 2024-01-01 \
    --end-date 2024-12-31 \
    --aggregation weighted \
    --winsorize \
    --corr-method spearman \
    --rolling-method spearman \
    --rolling-window 30 \
    --regression-mode multilag \
    --lag-count 3 \
    --join-method inner \
    --missing-method drop \
    --output-dir outputs/session_001
```

### Output Files

- `raw_news.csv`
- `article_sentiment.csv`
- `daily_sentiment_index.csv`
- `market_data.csv`
- `merged_dataset.csv`
- `rolling_correlation.csv`
- `extreme_sentiment_days.csv`
- `summary.json`
- `audit_trail.json`

### Run Interactive UI (Streamlit Wizard)

```bash
streamlit run streamlit_app.py
```

This UI follows the step-by-step wizard in `uispec.md`, including ethics gate, data quality checks, correlation/regression labs, and export with audit trail.

Before running query/fetch steps, open the `Settings — API Keys` page in the sidebar and add your `NewsAPI` key. You can optionally persist it to `.env` directly from the UI.
