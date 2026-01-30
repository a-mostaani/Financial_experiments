import torch
import torch.nn.functional as F

# Force GPU
device = torch.device("cuda")


# ---------------------------
# 1. GPU OLS Regression (Y ~ βX)
# ---------------------------
def gpu_ols_beta(X, Y):
    """
    Computes OLS slope β for Y = βX + e using GPU tensors.
    X, Y: shape (window_size,)
    Returns β
    """
    X_mean = X.mean()
    Y_mean = Y.mean()

    beta = torch.sum((X - X_mean) * (Y - Y_mean)) / torch.sum((X - X_mean)**2)
    return beta


# ---------------------------
# 2. GPU ADF Test on Residuals
# ---------------------------
def gpu_adf_test(residuals):
    """
    Simplified ADF(1) test on GPU.
    Full statistical tables are omitted; we return the ADF statistic.
    """

    # Δe_t
    diff = residuals[1:] - residuals[:-1]

    # lagged e_{t-1}
    lagged = residuals[:-1]

    # OLS for diff = α + ρ * e_{t-1}
    X = torch.stack([torch.ones_like(lagged), lagged], dim=1)  # (N,2)
    beta = torch.linalg.lstsq(X, diff).solution  # (2,)
    rho = beta[1]

    # Compute ADF statistic t = ρ / SE(ρ)
    y_pred = X @ beta
    residual = diff - y_pred
    s2 = torch.sum(residual**2) / (len(diff) - 2)
    var_rho = s2 / torch.sum((lagged - lagged.mean())**2)
    adf_stat = rho / torch.sqrt(var_rho)

    return adf_stat.item()


# ---------------------------
# 3. GPU Engle–Granger Cointegration Test
# ---------------------------
def gpu_cointegration_test(series_a, series_b):
    """
    series_a, series_b: numpy arrays or torch tensors (prices)
    Returns the ADF statistic of residuals.
    """

    # Convert to GPU tensors
    X = torch.tensor(series_a, dtype=torch.float32, device=device)
    Y = torch.tensor(series_b, dtype=torch.float32, device=device)

    # Step 1: compute beta
    beta = gpu_ols_beta(X, Y)

    # Step 2: compute residuals e = Y - βX
    residuals = Y - beta * X

    # Step 3: ADF test
    adf_stat = gpu_adf_test(residuals)

    return adf_stat


# ---------------------------
# 4. Rolling-Window GPU Cointegration Scan
# ---------------------------
def find_cointegrated_periods_gpu(df, symbol_a, symbol_b, window_size):
    leashed_periods = []

    data_a = df[symbol_a].values
    data_b = df[symbol_b].values

    for start in range(0, len(df) - window_size):
        end = start + window_size

        window_a = data_a[start:end]
        window_b = data_b[start:end]

        adf_stat = gpu_cointegration_test(window_a, window_b)

        # ADF threshold ~ -3.4 for 5% significance (approx)
        if adf_stat < -3.4:
            leashed_periods.append((
                df.index[start],
                df.index[end - 1],
                adf_stat
            ))

        print(f"Progress: {(start+1)/(len(df)-window_size)*100:.1f}%", end="\r")

    return leashed_periods



#### Example Usage
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

symbol_a = "PEP"
symbol_b = "KO"

df = yf.download([symbol_a, symbol_b], period="8d", interval="1m")["Close"].dropna()

periods = find_cointegrated_periods_gpu(df, symbol_a, symbol_b, window_size=20)

print("Found periods:", periods)



# Plot the two series with cointegrated periods highlighted
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

ax1.plot(df.index, df[symbol_a], label=symbol_a, linewidth=1.5)
ax2.plot(df.index, df[symbol_b], label=symbol_b, linewidth=1.5, color='orange')

# Highlight cointegrated periods
for start_date, end_date, adf_stat in periods:
    ax1.axvspan(start_date, end_date, alpha=0.3, color='green')
    ax2.axvspan(start_date, end_date, alpha=0.3, color='green')

ax1.set_ylabel('Price')
ax1.set_title(f'{symbol_a} with Cointegrated Periods Highlighted')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.set_ylabel('Price')
ax2.set_xlabel('Time')
ax2.set_title(f'{symbol_b} with Cointegrated Periods Highlighted')
ax2.legend()
ax2.grid(True, alpha=0.3)

green_patch = mpatches.Patch(color='green', alpha=0.3, label='Cointegrated')
fig.legend(handles=[green_patch], loc='upper right')

plt.tight_layout()
plt.show() 