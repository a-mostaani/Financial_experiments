"""
data_pipeline.py
----------------
Assembles a historical training DataFrame for ArbitrageTradingEnv.

Data sources
~~~~~~~~~~~~
* Prices          : yfinance (free, no API key required)
* Cointegration   : arbit_gpu_vectorized.scan_cointegration_windows_gpu (GPU)
* Z-score         : calc_z.rolling_zscore
* Volatility      : rolling std of returns, annualised
* Bid-ask spread  : PROXY — fixed fraction of mid price.
                    yfinance does not expose historical bid-ask data.
                    Default is 0.05 % which is conservative for large-cap
                    liquid equities. Replace with tick data when available.
* Sentiment       : "neutral" placeholder for all rows.
                    Swap in real labels from the PostgreSQL sentiment table
                    once you have accumulated enough history from the pipeline.
"""

import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf

# Allow importing sibling modules (arbit_gpu_vectorized, calc_z) from the repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from arbit_gpu_vectorized import scan_cointegration_windows_gpu
from calc_z import rolling_zscore


def build_training_data(
    symbol_a: str,
    symbol_b: str,
    period: str = "60d",
    interval: str = "1h",
    coint_window: int = 20,
    vol_window: int = 20,
    spread_proxy_pct: float = 0.0005,
) -> pd.DataFrame:
    """
    Download historical prices and assemble a per-timestep training DataFrame.

    Parameters
    ----------
    symbol_a, symbol_b  : Ticker symbols for the cointegrated pair (e.g. "AMD", "NVDA").
    period              : yfinance history length  — "7d", "30d", "60d", "1y", etc.
    interval            : yfinance bar size        — "1m", "5m", "1h", "1d", etc.
                          Note: intraday intervals are limited to 60 days of history
                          by the yfinance / Yahoo Finance API.
    coint_window        : Rolling window length (bars) for cointegration and z-score.
    vol_window          : Rolling window length (bars) for volatility estimate.
    spread_proxy_pct    : Constant bid-ask spread proxy as a fraction of mid price.
                          0.0005 = 0.05 %.

    Returns
    -------
    pd.DataFrame indexed by timestamp, one row per bar, with columns:
        z_score          float   rolling z-score of the spread residual
        spread_pct       float   bid-ask spread proxy (constant)
        beta             float   OLS hedge ratio from latest coint window
        adf_t            float   ADF t-statistic from latest coint window
        pvalue           float   MacKinnon p-value from latest coint window
        cointegrated     bool    cointegration flag from latest coint window
        market_volatility float  annualised rolling volatility of symbol_a returns
        sentiment_label  str     "neutral" placeholder — replace with real labels
    """

    # ---------------------------------------------------------------------- #
    # 1.  Download OHLCV prices
    # ---------------------------------------------------------------------- #
    print(f"[pipeline] Downloading {symbol_a}/{symbol_b}  period={period}  interval={interval}")
    raw = yf.download(
        [symbol_a, symbol_b],
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=True,
    )

    # yfinance returns a MultiIndex column frame; extract Close prices.
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"][[symbol_a, symbol_b]].copy()
    else:
        prices = raw[["Close"]].copy()

    prices = prices.dropna()
    print(f"[pipeline] {len(prices)} bars after dropna.")

    min_rows = coint_window + vol_window + 5
    if len(prices) < min_rows:
        raise ValueError(
            f"Only {len(prices)} rows available but need at least {min_rows}. "
            "Use a longer period or shorter interval."
        )

    # ---------------------------------------------------------------------- #
    # 2.  Cointegration stats — GPU (beta, adf_t, pvalue, cointegrated)
    # ---------------------------------------------------------------------- #
    print("[pipeline] Running cointegration scan on GPU...")
    coint_df = scan_cointegration_windows_gpu(
        prices,
        symbol_a,
        symbol_b,
        window=coint_window,
        regression="c",
        alpha=0.05,
    )

    # Index by the window-end timestamp so we can join to the price series.
    coint_df = (
        coint_df
        .set_index("end")[["beta", "adf_t", "pvalue", "cointegrated"]]
        .copy()
    )
    coint_df.index = pd.DatetimeIndex(coint_df.index)

    # ---------------------------------------------------------------------- #
    # 3.  Spread z-score
    #
    #   Use the median beta across all windows as a stable hedge ratio.
    #   This prevents the z-score from jumping each bar as beta fluctuates.
    #   Spread = price_B - median_beta * price_A  (cointegration residual).
    # ---------------------------------------------------------------------- #
    median_beta = float(coint_df["beta"].dropna().median())
    spread_series = prices[symbol_b] - median_beta * prices[symbol_a]
    z_series = rolling_zscore(spread_series, window=coint_window, ddof=1)
    z_series.name = "z_score"

    # ---------------------------------------------------------------------- #
    # 4.  Market volatility — annualised rolling std of symbol_a returns
    #
    #   For 1-hour bars: annualise with sqrt(252 * 6.5) ≈ sqrt(1638).
    #   For daily bars:  annualise with sqrt(252).
    #   We use a generic sqrt(252) here; adjust if using intraday intervals.
    # ---------------------------------------------------------------------- #
    returns = prices[symbol_a].pct_change()
    vol_series = returns.rolling(vol_window).std() * np.sqrt(252)
    vol_series.name = "market_volatility"

    # ---------------------------------------------------------------------- #
    # 5.  Bid-ask spread proxy (constant)
    # ---------------------------------------------------------------------- #
    spread_pct_series = pd.Series(
        spread_proxy_pct, index=prices.index, name="spread_pct"
    )

    # ---------------------------------------------------------------------- #
    # 6.  Sentiment placeholder
    #
    #   All rows default to "neutral" (maps to 0.0 in feature_builder).
    #   To use real sentiment:
    #     - Query the PostgreSQL `sentiment` table by ticker + timestamp
    #     - Join on date (or nearest bar) and replace these values.
    # ---------------------------------------------------------------------- #
    sentiment_series = pd.Series(
        "neutral", index=prices.index, name="sentiment_label"
    )

    # ---------------------------------------------------------------------- #
    # 7.  Assemble and align
    #
    #   Inner join keeps only bars where both price-derived features AND
    #   cointegration results are available (first `coint_window - 1` bars
    #   will be dropped because cointegration requires a full window).
    # ---------------------------------------------------------------------- #
    base = pd.concat([z_series, spread_pct_series, vol_series, sentiment_series], axis=1)
    merged = base.join(coint_df, how="inner")

    # Drop any remaining NaN rows (can appear at the edges of rolling windows).
    required_cols = ["z_score", "beta", "adf_t", "pvalue", "market_volatility"]
    merged = merged.dropna(subset=required_cols).sort_index()

    print(f"[pipeline] Training dataset ready: {len(merged)} timesteps.")
    return merged
