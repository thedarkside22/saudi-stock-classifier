import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score




def get_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    ma_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    ma_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_gain / ma_loss
    return 100 - (100 / (1 + rs))


def calculate_atr(df, period=14):
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    ranges = pd.concat([high_low, high_close,low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.ewm(alpha=1/period, adjust=False).mean()


def calculate_bb_width(series, period=20):
    ma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = ma + (std * 2)
    lower = ma - (std * 2)
    return (upper - lower) / ma


def build_features(df:pd.DataFrame) -> pd.DataFrame:
    df["Return"] = df["Close"].pct_change()
    df["ret_14days"] = df["Return"].rolling(14).mean().shift(1)
    df["RSI"] = get_rsi(df["Close"])
    df["Volatility_std"] = df["Return"].rolling(14).std().shift(1)
    df["ATR"] = calculate_atr(df)
    df["BB_width"] = calculate_bb_width(df["Close"])
    df = df.dropna()
    return df


