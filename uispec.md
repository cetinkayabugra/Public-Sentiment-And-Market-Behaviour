# Interactive Research UI Spec (Step-by-Step Wizard)

## Purpose
Build an interactive user interface that guides a user through the full research workflow:

1) Define scope (global → sector → stock)  
2) Fetch news  
3) Run FinBERT sentiment  
4) Build sentiment index  
5) Fetch price data  
6) Merge datasets  
7) Run correlation (Pearson / Spearman / Rolling)  
8) Run regression (with lags)  
9) Interpret economic impact  
10) Export + audit trail (reproducibility)

This UI is **not** for trading signals. It is a research tool that explains *what is happening at each stage*.

---

## UX Model: “Research Wizard”
### Key Design Principles
- **One step at a time** (wizard navigation)
- Every step shows:
  - Inputs
  - Output preview
  - Quality checks (missing dates, outliers)
  - “What this means” explanation
  - Limitations and ethical notes
- **No buy/sell language** anywhere

---

## Global Layout
### Main Shell
- Top bar:
  - Project Name (Research Session)
  - Data Range selector
  - Save/Load Session
  - Export (CSV + PDF summary)
- Left side: **Steps Navigation**
- Right side: **Step Workspace**
- Bottom: **Run Log / Audit Trail**

---

## Step 0 — Welcome & Ethics Gate
### UI Elements
- Checkbox: “I understand this tool is for research/education, not financial advice.”
- Checkbox: “I understand data sources may be incomplete and rate-limited.”
- Button: `Start Research`

### Output
- Creates a **Research Session ID**
- Starts an **audit log** (timestamped actions)

---

## Step 1 — Choose Research Scope (Hierarchy)
### UI Elements
- Radio buttons:
  - Global (macro market sentiment)
  - Sector (healthcare, tourism, tech, etc.)
  - Stock (single ticker)
- Dropdowns:
  - Region (US/EU/UK/Global)
  - Sector list (GICS-like categories)
- Optional: `Earnings focus` toggle (On/Off)

### Output Preview
- Summary card:
  - Scope: Sector
  - Sector: Healthcare
  - Date range: 2020-01-01 → 2021-12-31
  - Earnings focus: ON

---

## Step 2 — Data Sources & API Settings
### UI Elements
- News source selector:
  - NewsAPI
  - GNews
  - (Optional) RSS feeds
- Rate-limit indicator:
  - Requests remaining today
- “Query Builder”:
  - Sector keywords
  - Exclusion keywords
  - Language / country filters
- Button: `Test Query`

### Output Preview
- Sample of 10 headlines
- Duplicate detection count
- Coverage score (days with at least 1 article)

---

## Step 3 — Fetch News (Raw Dataset)
### UI Elements
- Button: `Fetch News`
- Progress bar
- Table preview:
  - Date
  - Source
  - Headline
  - Snippet
  - URL

### Quality Checks Panel
- Missing dates
- Duplicate articles
- Overly generic articles filter (optional)
- Button: `Clean & Continue`

---

## Step 4 — Run FinBERT (Article-Level Sentiment)
### UI Elements
- Button: `Run FinBERT`
- Settings:
  - Model version (FinBERT default)
  - Batch size
- Output table:
  - Article sentiment score: `Ppos - Pneg`
  - Label (pos/neu/neg)
  - Confidence score

### Explanation Box (“What this does”)
FinBERT estimates sentiment from financial language. We convert probabilities into a numeric score:
`Sentiment = Ppos - Pneg` (range -1 to +1)

---

## Step 5 — Build Daily Sentiment Index
### UI Elements
- Aggregation method:
  - Mean (default)
  - Median
  - Weighted (by confidence)
- Toggle: “Winsorize outliers” (optional)
- Button: `Build Index`

### Output
- Chart: Daily Sentiment Index over time
- Table:
  - Date
  - Daily sentiment
  - Article count
- Coverage meter:
  - % of days with sentiment value

---

## Step 6 — Extreme Events (Z-Score)
### UI Elements
- Toggle: “Standardize sentiment”
- Slider: Z-threshold (default 2.0)
- Button: `Detect Extremes`

### Output
- Table of extreme days:
  - Date
  - Sentiment
  - Z-score
  - Top contributing headlines
- Chart overlay:
  - Sentiment index with extreme points highlighted

---

## Step 7 — Fetch Price Data (Yahoo Finance)
### UI Elements
- Target:
  - Sector ETF proxy (recommended for sector-level research)
  - Specific ticker (if stock scope)
  - Market benchmark (S&P 500 / FTSE 100 etc.)
- Button: `Fetch Prices`

### Output
- Table:
  - Date
  - Close
  - Return (log)
- Chart:
  - Return series
- Data completeness:
  - Trading days count
  - Missing days report

---

## Step 8 — Merge Datasets on Date
### UI Elements
- Merge method:
  - Inner join (default)
  - Left join (keep sentiment days)
- Missing handling:
  - Forward fill (not recommended)
  - Drop missing (recommended)
- Button: `Merge`

### Output
Merged dataset preview:
- Date
- Sentiment index
- Returns

Quality checks:
- Rows before merge vs after merge
- % dropped days

---

## Step 9 — Correlation Lab
### UI Elements
- Correlation type:
  - Pearson
  - Spearman
- Rolling correlation toggle:
  - Window length slider (7 / 14 / 30 / 60)
- Button: `Run Correlation`

### Output
- Correlation results card:
  - Corr value
  - p-value (if available)
  - Interpretation
- Rolling plot (if enabled):
  - Rolling correlation line over time
- “Regime notes” panel:
  - Identify periods of high/low correlation

---

## Step 10 — Regression Lab
### UI Elements
- Model choice:
  - OLS: `Return_t = α + β Sentiment_t + ε`
  - Lagged OLS: `Return_t = α + β Sentiment_{t-1} + ε`
  - Multi-lag: choose lag count (1–5)
- Button: `Run Regression`

### Output
- Regression table:
  - α, β estimates
  - t-stat, p-value
  - R²
- Diagnostics:
  - Residual plot
  - Autocorrelation warning (basic)
- Interpretation panel:
  - “A +0.10 change in sentiment is associated with X% change in return”

---

## Step 11 — Economic Interpretation & Narrative Builder
### UI Elements
- Auto-generated explanation (editable):
  - What correlation shows
  - What regression shows
  - Whether effect is meaningful economically
- Confidence level tags:
  - Strong / Moderate / Weak evidence
- “Limitations” auto section:
  - API coverage, causality, omitted variables

### Output
A draft “Research Summary” paragraph ready to paste into report.

---

## Step 12 — Export & Reproducibility
### UI Elements
- Export options:
  - CSV (raw news, article sentiment, daily index, merged dataset)
  - PDF research report (summary + charts + tables)
  - JSON session file (re-run later)
- Button: `Export`

### Audit Trail
Downloadable run log:
- Inputs used
- Dates
- Data sources
- Parameters (rolling window, lag count)
- Model outputs

---

## Non-Negotiable Ethical UI Rules
- No “buy”, “sell”, “signal”, “recommendation”
- Always show:
  - Data limitations
  - Correlation ≠ causation warning
  - Research-only disclaimer
- Provide reproducibility (session export + logs)

---

## Suggested Components (Implementation)
- Stepper / Wizard (left nav)
- Data tables with pagination
- Plot components:
  - Sentiment index chart
  - Rolling correlation chart
  - Return chart
  - Regression diagnostics (basic)
- Session state store:
  - Selected scope + filters
  - Cached datasets
  - Results objects

---

## Success Criteria (What examiner will like)
- Clear methodology mirrored in UI
- Transparent transformation at each step
- Repeatability (session export)
- Ethical framing built into interaction
- Supports hierarchical concept (global/sector/stock)
