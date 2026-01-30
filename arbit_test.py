import statsmodels.tsa.stattools as ts
import yfinance as yf

def check_for_leash(symbol_a, symbol_b):
    # 1. Get 1 year of historical data
    data = yf.download([symbol_a, symbol_b], period="1y")['Close'].dropna()
    
    # 2. Run the Cointegration Test (The "Leash" Test)
    # This returns: (test_stat, p-value, critical_values)
    score, p_value, _ = ts.coint(data[symbol_a], data[symbol_b])
    
    if p_value < 0.05:
        print(f"MATCH: {symbol_a} and {symbol_b} are cointegrated (p={p_value:.4f})")
        return True
    else:
        print(f"SKIP: No statistical leash found. The p-value is {p_value:.4f}.")
        return False
    
def find_leashed_periods(symbol_a,symbol_b, window_size=500):
    # 1. Get 5 years of historical data
    data = yf.download([symbol_a, symbol_b], period="8d", interval="1m")['Close'].dropna()
    
    leashed_periods = []
    
    # 2. Check for cointegration in rolling windows of window_size days
    for start in range(0, len(data) - window_size):
        end = start + window_size
        window_data = data.iloc[start:end]
        
        score, p_value, _ = ts.coint(window_data[symbol_a], window_data[symbol_b])
        
        if p_value < 0.05:
            leashed_periods.append((window_data.index[0], window_data.index[-1], p_value))

        progress = (start + 1) / (len(data) - window_size) * 100
        print(f"Progress: {progress:.1f}%", end="\r")
        
    
    if leashed_periods:
        print(f"Found {len(leashed_periods)} leashed periods between {symbol_a} and {symbol_b}:")
        for start_date, end_date, p_val in leashed_periods:
            print(f" - From {start_date} to {end_date} (p={p_val:.4f})")
    else:
        print(f"No leashed periods found between {symbol_a} and {symbol_b}.")
    
    return leashed_periods


# Test it on a classic pair
# check_for_leash("PEP", "KO")
find_leashed_periods("PEP", "KO", window_size=500)