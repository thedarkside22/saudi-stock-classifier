# Tadawul Stock Direction Predictor

An end-to-end machine learning system that predicts the probability of a >1% next-day price increase for 43 actively-traded Saudi (Tadawul) stocks. Random Forest classifier on engineered technical features, deployed as a containerized FastAPI service with a live demo.

**Live demo:** [YOUR_LIVE_URL]
**API docs:** [YOUR_LIVE_URL]/docs

---

## Overview

The system takes a ticker symbol, fetches recent OHLCV data from Yahoo Finance, computes technical indicators, and returns the probability that the stock will close >1% higher on the next trading day.

- **Model:** Random Forest Classifier (200 estimators, max depth 10)
- **Features:** 7 engineered features — Volume, returns, RSI, rolling momentum, rolling volatility, ATR, Bollinger Band width
- **Training data:** ~34,000 (ticker, date) observations across 43 tickers, January 2023 – May 2026
- **Test ROC-AUC:** 0.58 (temporal hold-out)
- **Stack:** Python, scikit-learn, pandas, FastAPI, Docker, yfinance, Render

---

## Architecture
yfinance API  →  data.py (inference loader)  →  features.py (engineered features)
↓
stock_price_model.joblib
↓
FastAPI /predict
↓
static/index.html (frontend)

The training pipeline (notebooks) and inference pipeline (src) share the same feature engineering code in `src/features.py`, guaranteeing consistency between how the model was trained and how it sees new data.

---

## Project structure
saudi-stock-classifier/
├── data/                          # cleaned training data (parquet)
├── models/
│   ├── stock_price_model.joblib   # trained Random Forest
│   └── features_cols.json         # canonical feature column order
├── notebooks/
│   ├── EDA.ipynb                  # exploratory data analysis
│   └── feature_engineering.ipynb  # feature design + model training
├── src/
│   ├── api.py                     # FastAPI app
│   ├── data.py                    # inference data loader + ticker validation
│   └── features.py                # feature engineering (RSI, ATR, BB width, etc.)
├── static/
│   └── index.html                 # frontend page
├── Dockerfile
├── requirements.txt
└── README.md

---

## Approach

### Data pipeline

Daily OHLCV for 50 large-cap Tadawul tickers was pulled from Yahoo Finance covering 2023-01-01 through 2026-05-07. The cleaning pipeline:

1. Dropped phantom rows (yfinance occasionally fabricates rows for Saudi non-trading days with `Volume=0`)
2. Dropped tickers with median daily volume below ~100K shares (insufficient liquidity)
3. Dropped tickers with fewer than 500 days of history (insufficient training data)

Final dataset: **43 tickers, 33,903 rows**.

### Features

All rolling features use `.shift(1)` to avoid look-ahead bias — feature values at time `t` use only data available before time `t`.

| Feature | Description |
|---|---|
| `Volume` | Daily traded volume |
| `Return` | Daily percent change in close |
| `RSI` | 14-day Relative Strength Index |
| `ret_14days` | 14-day rolling mean of daily returns |
| `Volatility_std` | 14-day rolling std of daily returns |
| `ATR` | 14-day Average True Range |
| `BB_width` | 20-day Bollinger Band width |

### Target
1 if Close[t+1] > Close[t] * 1.01 else 0

Predicting >1% next-day moves rather than any direction. The 1% threshold filters out micro-fluctuations and class-imbalances the target to ~22/78, focusing the model on meaningful directional moves.

### Validation

Temporal hold-out split (first 80% of dates → train, last 20% → test). Random k-fold validation would leak future information on time-series data and produce inflated scores. Confirmed by exploratory experiments showing random splits produced unstable AUCs (0.42-0.48 across seeds) while temporal splits gave deterministic results.

---

## Results

| Configuration | Test ROC-AUC |
|---|---|
| Baseline (predict majority class) | 0.500 |
| Random Forest, 7 engineered features | **0.585** |

A 0.58 AUC indicates the model extracts real but modest signal from technical features. This is consistent with the broader finding that daily-resolution stock direction prediction has a low signal-to-noise ratio.

---

## Limitations and honesty

This is a **technical demonstration of an ML engineering pipeline**, not a tradeable strategy. Specifically:

- A 0.58 AUC, while above chance, is far below what would be needed for profitable trading after transaction costs
- The model has no awareness of news, earnings, macro events, or other fundamental drivers
- Backtesting was on a relatively short window (~8 months of held-out data)
- Saudi market microstructure (limit-up/limit-down circuit breakers, Sunday-Thursday trading) is not explicitly modeled

The value of this project is in demonstrating end-to-end ML engineering: leakage-safe feature engineering, temporal validation, model deployment, and a working production API — not in beating the market.

---

## Running locally

### With Docker (recommended)

```bash
docker build -t saudi-stock-classifier .
docker run -p 8000:8000 saudi-stock-classifier
```

Visit `http://localhost:8000`.

### Without Docker

```bash
pip install -r requirements.txt
uvicorn src.api:app --reload
```

Visit `http://localhost:8000`.

---

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Frontend page with ticker dropdown |
| GET | `/health` | Health check |
| GET | `/tickers` | List of supported ticker symbols |
| GET | `/predict?ticker=2222.SR` | Predict for a single ticker |
| GET | `/docs` | Interactive Swagger UI |

### Example response

```json
{
  "ticker": "2222.SR",
  "probability": 0.1733,
  "as_of": "2026-05-12"
}
```

---

## Author

**Ziyad Alrasheedi**
B.Sc. Computer Science (AI specialization), Imam Mohammad Ibn Saud Islamic University (CCIS), Riyadh
[GitHub](https://github.com/thedarkside22)