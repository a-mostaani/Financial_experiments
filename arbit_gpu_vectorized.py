# ============================================================
# Batch Engle–Granger cointegration test on GPU (PyTorch)
# Vectorized over ALL rolling windows in one go.
# ============================================================

import torch
import pandas as pd
from typing import Tuple

import numpy as np
import torch

def _to_gpu_1d(x, dtype=torch.float64):
    """
    x: array-like (NumPy array, pandas Series, list)
    Returns a 1D torch.Tensor on CUDA without the 'non-writable' warning.
    """
    # Force a writable NumPy array (copy=True guarantees writeable, contiguous buffer)
    np_x = np.array(x, dtype=np.float64, copy=True)
    return torch.tensor(np_x, dtype=dtype, device="cuda")  # torch.tensor() copies


def _rolling_unfold_1d(x: torch.Tensor, window: int) -> torch.Tensor:
    """
    Return shape: (num_windows, window)
    """
    # x must be 1D
    if x.dim() != 1:
        raise ValueError("x must be 1D")
    # Use unfold over a dummy batch dimension
    x2 = x.unsqueeze(0)  # (1, T)
    # Unfold over last dimension
    return x2.unfold(dimension=1, size=window, step=1).squeeze(0)  # (num_windows, window)

@torch.no_grad()
def engle_granger_adf_batch(
    series_a, series_b, window: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute Engle–Granger (ADF on residuals) across ALL rolling windows on GPU.

    series_a, series_b: array-like (length T)
    window: int rolling window length

    Returns:
        beta_windows: (num_windows,) OLS slope β per window for Y ~ βX
        adf_t_windows: (num_windows,) ADF t-stat per window on residuals
        valid_mask: (num_windows,) bool mask for windows with finite stats
    """

    # Move to GPU
    X = _to_gpu_1d(series_a)  # (T,)
    Y = _to_gpu_1d(series_b)  # (T,)

    T = X.numel()
    if T != Y.numel():
        raise ValueError("Series lengths must match.")
    if T < window + 2:
        raise ValueError("Series too short for requested window.")

    # Build all rolling windows (N, W)
    Xw = _rolling_unfold_1d(X, window)  # (N, W)
    Yw = _rolling_unfold_1d(Y, window)  # (N, W)
    N, W = Xw.shape

    # --- Step 1: OLS β per window for Y ~ βX (with intercept absorbed by centering)
    Xc = Xw - Xw.mean(dim=1, keepdim=True)
    Yc = Yw - Yw.mean(dim=1, keepdim=True)
    Sxx = torch.sum(Xc * Xc, dim=1)              # (N,)
    Sxy = torch.sum(Xc * Yc, dim=1)              # (N,)
    beta = Sxy / Sxx                             # (N,)

    # Guard against zero variance windows (flat price) -> set NaN beta
    beta = torch.where(Sxx > 0, beta, torch.full_like(beta, float("nan")))

    # --- Step 2: residuals e_t = Y - βX (broadcast β over time dimension)
    Ew = Yw - beta.unsqueeze(1) * Xw              # (N, W)

    # --- Step 3: ADF(1) on residuals, window-wise, with intercept
    # Δe_t = α + ρ e_{t-1} + u_t
    lagged = Ew[:, :-1]                           # (N, W-1) -> e_{t-1}
    diff   = Ew[:, 1:] - Ew[:, :-1]               # (N, W-1) -> Δe_t

    # OLS with intercept can be done via covariance/variance on centered data:
    x = lagged
    y = diff
    x_mean = x.mean(dim=1, keepdim=True)          # (N,1)
    y_mean = y.mean(dim=1, keepdim=True)          # (N,1)
    x_c = x - x_mean
    y_c = y - y_mean

    Sxx_adf = torch.sum(x_c * x_c, dim=1)         # (N,)
    Sxy_adf = torch.sum(x_c * y_c, dim=1)         # (N,)
    rho = Sxy_adf / Sxx_adf                        # (N,)

    # α = ȳ - ρ x̄  (not used for t-stat directly but needed for residuals)
    alpha = (y_mean.squeeze(1) - rho * x_mean.squeeze(1))  # (N,)

    # Residuals of the regression to estimate s^2
    y_hat = alpha.unsqueeze(1) + rho.unsqueeze(1) * x      # (N, W-1)
    u = y - y_hat                                          # (N, W-1)
    dof = (W - 1) - 2  # parameters: α and ρ
    s2 = torch.sum(u * u, dim=1) / dof                     # (N,)

    # Var(ρ) = s^2 / Sxx
    var_rho = s2 / Sxx_adf
    adf_t = rho / torch.sqrt(var_rho)                      # (N,)

    # Valid windows mask (finite stats & positive Sxx terms)
    valid = torch.isfinite(adf_t) & torch.isfinite(beta) & (Sxx > 0) & (Sxx_adf > 0)

    return beta, adf_t, valid

def scan_cointegration_windows_gpu(
    df: pd.DataFrame,
    symbol_a: str,
    symbol_b: str,
    window: int,
    adf_crit: float = -3.4,   # tune this to your needs or pass exact value
    dropna: bool = True,
) -> pd.DataFrame:
    """
    df: DataFrame with columns [symbol_a, symbol_b] (e.g., Close prices)
    Returns a DataFrame of all windows and the subset that pass the ADF threshold.
    """

    # Clean and align
    sub = df[[symbol_a, symbol_b]].dropna() if dropna else df[[symbol_a, symbol_b]]

    if len(sub) < window + 2:
        raise ValueError("Not enough rows after dropna for the given window.")

    # Batch GPU computation
    beta, adf_t, valid = engle_granger_adf_batch(
        sub[symbol_a].to_numpy(copy=True), sub[symbol_b].to_numpy(copy=True), window=window
    )

    # Map windows to start/end timestamps
    idx = sub.index
    N = len(sub) - window + 1
    start_idx = idx[:N]
    end_idx = idx[window - 1:]

    # Move results back to CPU
    beta_c = beta.detach().cpu()
    adf_c  = adf_t.detach().cpu()
    valid_c = valid.detach().cpu()

    out = pd.DataFrame({
        "start": start_idx.values,
        "end": end_idx.values,
        "beta": beta_c.numpy(),
        "adf_t": adf_c.numpy(),
        "valid": valid_c.numpy().astype(bool),
    })

    # Flag cointegrated windows using the threshold and validity
    out["cointegrated"] = (out["adf_t"] < adf_crit) & out["valid"]

    return out



import yfinance as yf

symbol_a, symbol_b = "PEP", "KO"
df = yf.download([symbol_a, symbol_b], period="8d", interval="1m")["Close"]

# Batch scan: e.g., a ~ 200-minute window
res = scan_cointegration_windows_gpu(df, symbol_a, symbol_b, window=20, adf_crit=-3.4)
res_cointegrated = res[res["cointegrated"]]

print(f"Total windows: {len(res)}; Cointegrated: {len(res_cointegrated)}")
print(res_cointegrated.head())