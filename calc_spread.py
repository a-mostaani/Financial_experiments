import yfinance as yf
from typing import Optional, Dict

def get_bid_ask_spread(symbol: str) -> Optional[Dict[str, float]]:
    """
    Fetches bid and ask from Yahoo Finance using yfinance and computes:
      - absolute spread (ask - bid)
      - percent spread relative to mid ( (ask - bid) / ((ask + bid)/2) )
    
    Returns dict with bid, ask, mid, spread, spread_pct if available;
    otherwise returns None.
    """
    t = yf.Ticker(symbol)

    # Try .info (primary) then .fast_info (fallback) — field presence varies by ticker
    info = {}
    try:
        info = t.info or {}
    except Exception:
        pass

    bid = info.get("bid")
    ask = info.get("ask")

    # Fallback to fast_info if needed
    if bid is None or ask is None:
        try:
            fi = t.fast_info
            # Some yfinance versions expose bid/ask here; if not present, leave as-is
            bid = bid if bid is not None else getattr(fi, "bid", None) or fi.get("bid", None)
            ask = ask if ask is not None else getattr(fi, "ask", None) or fi.get("ask", None)
        except Exception:
            pass

    # If still missing, we can't compute spread robustly
    if bid is None or ask is None or bid == 0 or ask == 0:
        return None

    mid = (bid + ask) / 2.0
    spread = ask - bid
    spread_pct = spread / mid if mid else float("nan")

    return {
        "bid": float(bid),
        "ask": float(ask),
        "mid": float(mid),
        "spread": float(spread),
        "spread_pct": float(spread_pct),
    }

# Example:
# print(get_bid_ask_spread("AAPL"))