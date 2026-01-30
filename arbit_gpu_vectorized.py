# ============================================================
# Batch Engle–Granger cointegration test on GPU (PyTorch)
# Vectorized over ALL rolling windows in one go.
# ============================================================

# ============================================================
# Add MacKinnon (1994/2010) critical values & p-values
# to the batch GPU Engle–Granger pipeline.
# ============================================================

import numpy as np
import torch
import pandas as pd
from typing import Tuple
from statsmodels.tsa.adfvalues import mackinnoncrit, mackinnonp  # MacKinnon CVs & p-values
# Alternative import also works:
# from statsmodels.tsa.stattools import mackinnoncrit, mackinnonp
# (stattools re-exports them)  # docs: statsmodels API  [4](https://www.statsmodels.org/stable/_modules/statsmodels/tsa/stattools.html)[5](https://www.statsmodels.org/stable/api.html)

# ---------- (same helpers; includes the "no-warning" tensor conversion) ----------
def _to_gpu_1d(x, dtype=torch.float64) -> torch.Tensor:
    np_x = np.array(x, dtype=np.float64, copy=True)  # force writable, contiguous
    return torch.tensor(np_x, dtype=dtype, device="cuda")

def _rolling_unfold_1d(x: torch.Tensor, window: int) -> torch.Tensor:
    if x.dim() != 1:
        raise ValueError("x must be 1D")
    return x.unsqueeze(0).unfold(dimension=1, size=window, step=1).squeeze(0)

@torch.no_grad()
def engle_granger_adf_batch(series_a, series_b, window: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute Engle–Granger (ADF on residuals) across ALL rolling windows on GPU.
    Returns:
        beta  : (num_windows,) OLS slope Y ~ βX per window
        adf_t : (num_windows,) ADF(1) t-stat on residuals (intercept only)
        valid : (num_windows,) boolean mask of finite/valid windows
    """
    X = _to_gpu_1d(series_a)
    Y = _to_gpu_1d(series_b)

    if X.numel() != Y.numel():
        raise ValueError("Series lengths must match.")
    if X.numel() < window + 2:
        raise ValueError("Series too short for requested window.")

    # Build all rolling windows (N, W)
    Xw = _rolling_unfold_1d(X, window)  # (N, W)
    Yw = _rolling_unfold_1d(Y, window)  # (N, W)
    N, W = Xw.shape

    # ----- OLS β per window: Y ~ βX (demeaned)
    Xc = Xw - Xw.mean(dim=1, keepdim=True)
    Yc = Yw - Yw.mean(dim=1, keepdim=True)
    Sxx = torch.sum(Xc * Xc, dim=1)           # (N,)
    Sxy = torch.sum(Xc * Yc, dim=1)           # (N,)
    beta = Sxy / Sxx                          # (N,)
    beta = torch.where(Sxx > 0, beta, torch.full_like(beta, float("nan")))

    # Residuals
    Ew = Yw - beta.unsqueeze(1) * Xw          # (N, W)

    # ----- ADF(1) with intercept: Δe_t = α + ρ e_{t-1} + u_t
    lagged = Ew[:, :-1]                       # (N, W-1)
    diff   = Ew[:, 1:] - Ew[:, :-1]           # (N, W-1)

    # OLS of diff on [1, lagged]
    x = lagged
    y = diff
    x_mean = x.mean(dim=1, keepdim=True)
    y_mean = y.mean(dim=1, keepdim=True)
    x_c = x - x_mean
    y_c = y - y_mean

    Sxx_adf = torch.sum(x_c * x_c, dim=1)     # (N,)
    Sxy_adf = torch.sum(x_c * y_c, dim=1)     # (N,)
    rho = Sxy_adf / Sxx_adf                   # (N,)

    alpha = (y_mean.squeeze(1) - rho * x_mean.squeeze(1))  # (N,)
    y_hat = alpha.unsqueeze(1) + rho.unsqueeze(1) * x      # (N, W-1)
    u = y - y_hat                                          # (N, W-1)

    dof = (W - 1) - 2  # params: α and ρ
    s2 = torch.sum(u * u, dim=1) / dof                     # (N,)
    var_rho = s2 / Sxx_adf
    adf_t = rho / torch.sqrt(var_rho)                      # (N,)

    valid = torch.isfinite(adf_t) & torch.isfinite(beta) & (Sxx > 0) & (Sxx_adf > 0)

    return beta, adf_t, valid


# ---------- NEW: MacKinnon critical values & p-values ----------
def compute_mackinnon_cv_pvalues(adf_t_vals: np.ndarray, window: int, regression: str = "c"):
    """
    Get MacKinnon critical values (1%, 5%, 10%) and p-values for ADF stats.
    - Critical values use MacKinnon (2010) with finite-sample adjustment via nobs.
    - P-values use MacKinnon asymptotic response surfaces.
    regression: 'c' | 'ct' | 'ctt' | 'nc'  (must match the ADF regression form you used)
    """
    nobs = window - 1  # ADF(1) uses Δe_t of length W-1; this matches adfuller conventions. [3](https://www.statsmodels.org/0.8.0/generated/statsmodels.tsa.stattools.adfuller.html)

    # Critical values for ADF (N=1) at {1%, 5%, 10%}
    # mackinnoncrit returns the updated (2010) CVs; finite-sample if nobs is finite. [1](https://tedboy.github.io/statsmodels_doc/generated/statsmodels.tsa.adfvalues.mackinnoncrit.html)
    cv = mackinnoncrit(N=1, regression=regression, nobs=nobs)
    # Handle return type (list/array vs dict), depending on statsmodels version
    if isinstance(cv, dict):
        crit_1, crit_5, crit_10 = cv.get("1%"), cv.get("5%"), cv.get("10%")
    else:
        # Most versions return a length-3 array [1%, 5%, 10%]
        crit_1, crit_5, crit_10 = float(cv[0]), float(cv[1]), float(cv[2])

    # P-values from MacKinnon asymptotics (ADF, N=1). [2](https://tedboy.github.io/statsmodels_doc/generated/statsmodels.tsa.adfvalues.mackinnonp.html)
    pvals = np.array([mackinnonp(float(t), regression=regression, N=1) for t in adf_t_vals], dtype=float)

    return crit_1, crit_5, crit_10, pvals


def scan_cointegration_windows_gpu(
    df: pd.DataFrame,
    symbol_a: str,
    symbol_b: str,
    window: int,
    regression: str = "c",   # must match the GPU ADF regression (we use 'c' by default)
    alpha: float = 0.05,     # threshold significance level for cointegration flag
    dropna: bool = True,
) -> pd.DataFrame:
    """
    Return a DataFrame with window start/end, β, ADF t-stat, MacKinnon CVs, p-values, and flags.
    """
    sub = df[[symbol_a, symbol_b]].copy()
    if dropna:
        sub = sub.dropna()
    if len(sub) < window + 2:
        raise ValueError("Not enough rows after dropna for the given window.")

    # ---- GPU numerics
    beta, adf_t, valid = engle_granger_adf_batch(
        sub[symbol_a].to_numpy(copy=True),
        sub[symbol_b].to_numpy(copy=True),
        window=window,
    )

    # Map windows to timestamps
    idx = sub.index
    N = len(sub) - window + 1
    start_idx = idx[:N]
    end_idx = idx[window - 1:]

    # Bring results to CPU/NumPy
    beta_c = beta.detach().cpu().numpy()
    adf_c  = adf_t.detach().cpu().numpy()
    valid_c = valid.detach().cpu().numpy().astype(bool)

    # ---- NEW: MacKinnon CVs & p-values (CPU; tiny cost)
    crit_1, crit_5, crit_10, pvals = compute_mackinnon_cv_pvalues(adf_c, window=window, regression=regression)

    out = pd.DataFrame({
        "start": start_idx.values,
        "end": end_idx.values,
        "beta": beta_c,
        "adf_t": adf_c,
        "valid": valid_c,
        "crit_1pct": crit_1,
        "crit_5pct": crit_5,
        "crit_10pct": crit_10,
        "pvalue": pvals,
    })

    # Flag cointegration at chosen alpha using the corresponding critical value
    crit = crit_5 if alpha == 0.05 else (crit_1 if alpha == 0.01 else crit_10)
    out["cointegrated"] = (out["adf_t"] < crit) & out["valid"]

    return out



import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

symbol_a, symbol_b = "PEP", "KO"
df = yf.download([symbol_a, symbol_b], period="8d", interval="1m")["Close"]

# Batch scan: e.g., a ~ 200-minute window
res = scan_cointegration_windows_gpu(df, symbol_a, symbol_b, window=20,  regression="c", alpha=0.05)
res_cointegrated = res[res["cointegrated"]]

print(f"Total windows: {len(res)}; Cointegrated: {len(res_cointegrated)}")
print(res_cointegrated.head())



fig, ax = plt.subplots(figsize=(14, 6))

# Plot both series
ax.plot(df.index, df[symbol_a], label=symbol_a, linewidth=1.5, alpha=0.8)
ax.plot(df.index, df[symbol_b], label=symbol_b, linewidth=1.5, alpha=0.8)

# Shade cointegrated windows
for _, row in res_cointegrated.iterrows():
    ax.axvspan(row["start"], row["end"], alpha=0.2, color="green")

ax.set_xlabel("Time")
ax.set_ylabel("Price")
ax.set_title(f"Cointegration Windows: {symbol_a} vs {symbol_b}")
ax.legend()
ax.grid(True, alpha=0.3)

# Add legend for shaded regions
green_patch = mpatches.Patch(color="green", alpha=0.2, label="Cointegrated Windows")
ax.legend(handles=ax.get_legend_handles_labels()[0] + [green_patch])

plt.tight_layout()
plt.show()