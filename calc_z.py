import pandas as pd
import numpy as np

def rolling_zscore(series: pd.Series, window: int, ddof: int = 0) -> pd.Series:
    """
    Rolling Z-score: (x - rolling_mean) / rolling_std.
    - window: rolling window length in observations
    - ddof: degrees of freedom for std (0 for population-like, 1 for sample)
    """
    roll_mean = series.rolling(window).mean()
    roll_std = series.rolling(window).std(ddof=ddof)
    z = (series - roll_mean) / roll_std
    return z

def zscore(series: pd.Series, ddof: int = 0) -> pd.Series:
    """
    Standard Z-score across the entire series: (x - mean) / std.
    """
    mu = series.mean()
    sigma = series.std(ddof=ddof)
    return (series - mu) / sigma

# Example:
import yfinance as yf
px = yf.download("AAPL", period="6mo", interval="1d")["Close"]
z_roll_20 = rolling_zscore(px, window=20, ddof=1)
z_full = zscore(px, ddof=1)
print("Rolling Z-score (20-day):")
print(z_roll_20.tail())
print("\nFull-series Z-score:")
print(z_full.tail())