import numpy as np

# Maps sentiment_worker text labels to float scores
_SENTIMENT_MAP = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}

OBS_DIM = 9  # must stay in sync with agentClass.py observation_space shape


def encode_sentiment(label: str) -> float:
    """Convert sentiment_worker label to float. Unknown labels map to 0.0."""
    return _SENTIMENT_MAP.get(label.lower().strip(), 0.0)


def build_observation(
    sentiment_label: str,
    spread_pct: float,
    z_score: float,
    beta: float,
    adf_t: float,
    coint_pvalue: float,
    cointegrated: bool,
    available_cash: float,
    initial_cash: float,
    market_volatility: float,
) -> np.ndarray:
    """
    Build a normalized float32 observation vector for the RL trader.

    Parameters
    ----------
    sentiment_label   : sentiment_worker output — "positive" | "neutral" | "negative"
    spread_pct        : calc_spread  — get_bid_ask_spread()["spread_pct"]
    z_score           : calc_z       — latest value of rolling_zscore() or zscore()
    beta              : arbit_gpu_vectorized — OLS hedge ratio from latest window
    adf_t             : arbit_gpu_vectorized — ADF t-stat from latest window
    coint_pvalue      : arbit_gpu_vectorized — MacKinnon p-value from latest window
    cointegrated      : arbit_gpu_vectorized — cointegration flag for latest window
    available_cash    : current investable cash
    initial_cash      : starting cash (used to normalise cash_ratio)
    market_volatility : rolling or annualised return volatility of the traded asset

    Returns
    -------
    (9,) float32 array:
      [0] sentiment_score   {-1.0, 0.0, 1.0}
      [1] spread_pct        tanh(x * 50)         — spreads are tiny; scale before squash
      [2] z_score           tanh(x / 3)           — ±3σ → ±0.995
      [3] beta              tanh(x)               — hedge ratio, usually near 1
      [4] adf_t             tanh(x / 3)           — ADF stats typically in −5..0
      [5] coint_pvalue_inv  1 − pvalue            — higher = more statistically significant
      [6] cointegrated      0.0 or 1.0
      [7] cash_ratio        clip(cash / initial, 0, 1)
      [8] market_volatility tanh(x * 10)          — annualised ~0.2 → tanh(2) ≈ 0.96
    """
    cash_ratio = (
        float(np.clip(available_cash / initial_cash, 0.0, 1.0))
        if initial_cash > 0
        else 0.0
    )

    return np.array(
        [
            encode_sentiment(sentiment_label),          # [0]
            float(np.tanh(spread_pct * 50)),            # [1]
            float(np.tanh(z_score / 3.0)),              # [2]
            float(np.tanh(beta)),                       # [3]
            float(np.tanh(adf_t / 3.0)),                # [4]
            float(np.clip(1.0 - coint_pvalue, 0.0, 1.0)),  # [5]
            1.0 if cointegrated else 0.0,               # [6]
            cash_ratio,                                 # [7]
            float(np.tanh(market_volatility * 10)),     # [8]
        ],
        dtype=np.float32,
    )
