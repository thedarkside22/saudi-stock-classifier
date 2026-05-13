import yfinance as yf
import pandas as pd
import numpy as np
import json
from features import build_features
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

TICKERS = ['7010.SR', '2222.SR', '2310.SR', '4200.SR', '2380.SR', '2020.SR', '1211.SR',
 '2010.SR', '4030.SR', '2290.SR', '1010.SR', '1030.SR', '7020.SR', '8010.SR',
 '1020.SR', '1050.SR', '5110.SR', '4190.SR', '4280.SR', '1180.SR', '2280.SR',
 '1150.SR', '1140.SR', '4002.SR', '4013.SR', '1120.SR', '8210.SR', '1080.SR',
 '4220.SR', '4250.SR', '1060.SR', '4300.SR', '8230.SR', '1303.SR', '2223.SR',
 '1111.SR', '4164.SR', '2082.SR', '6015.SR', '7202.SR', '4142.SR', '2382.SR',
 '4263.SR']

with open(MODELS_DIR / "features_cols.json", "r") as f:
    FEATURE_COLS = json.load(f)

def load_inference_features(ticker:str) -> pd.DataFrame:

    if ticker not in TICKERS:
        raise ValueError("Ticker is not supported for predictions.")
    ticker_1 = yf.Ticker(ticker)
    df = yf.download(ticker_1.ticker, period="3mo", multi_level_index=False)

    if df.empty:
        raise ValueError(f"Invalid Ticker entered. {ticker}")
    df = df[df["Volume"] > 0]

    if len(df) < 25:
        raise ValueError(f"Insufficient data for {ticker}")

    df_features = build_features(df)

    if df_features.empty:
        raise ValueError(f"Feature computation produced no rows for {ticker}")

    last_row = df_features.iloc[[-1]]
    return last_row[FEATURE_COLS]

try:
    s = load_inference_features("Fake")
except ValueError as e:
    print(e)
