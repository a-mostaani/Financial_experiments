


#massive streamer:
import os
import time
import json
import logging
import signal
import sys
from typing import Dict, Iterable, Tuple, Optional
from datetime import datetime, timezone
import requests
import redis
from dateutil import parser as dtparse

# ------------------ Config ------------------
API_KEY =   os.getenv("MASSIVE_API_KEY")
if not API_KEY:
    raise RuntimeError("Set MASSIVE_API_KEY env variable")

# Try real-time Benzinga first; fallback to hourly Reference News if permission denied.
BENZINGA_URL = "https://api.massive.com/benzinga/v2/news"
REF_NEWS_URL = "https://api.massive.com/v2/reference/news"

TICKERS = [t.strip().upper() for t in os.getenv("NEWS_TICKERS", "").split(",") if t.strip()]
POLL_INTERVAL = float(os.getenv("POLL_INTERVAL", "2"))

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_QUEUE = os.getenv("REDIS_QUEUE", "news_queue")
REDIS_SEEN_SET = os.getenv("REDIS_SEEN_SET", "news_seen")

# ------------------ Logging ------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("massive-news")

# ------------------ Redis ------------------
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)

# ------------------ Helpers ------------------
def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def parse_time(s: str) -> datetime:
    # Accept both ISO-8601 and RFC-3339; dtparse handles 'Z'
    return dtparse.parse(s).astimezone(timezone.utc)

def select_id(payload: Dict) -> Tuple[str, Optional[str]]:
    """
    Returns (stable_id, published_iso) for either endpoint format.
    Benzinga fields: benzinga_id (int), published (ISO)
    Reference News fields: id (hash string), published_utc (ISO)
    """
    if "benzinga_id" in payload:
        return f"bz:{payload['benzinga_id']}", payload.get("published")
    # reference news
    return f"ref:{payload['id']}", payload.get("published_utc")

def build_benzinga_params(since: Optional[datetime]) -> Dict:
    params = {
        "limit": 200,   # tune as needed; max allowed is much higher
        "sort": "published.desc",
        "apiKey": API_KEY,
    }
    if TICKERS:
        params["tickers"] = ",".join(TICKERS)
    if since:
        # Massive accepts RFC3339, yyyy-mm-dd, or epoch seconds for 'published' filter
        params["published.gt"] = since.isoformat()
    return params

def build_ref_params(since: Optional[datetime]) -> Dict:
    params = {
        "order": "desc",
        "limit": 100,
        "sort": "published_utc",
        "apiKey": API_KEY,
    }
    if TICKERS:
        params["ticker"] = ",".join(TICKERS)
    if since:
        params["published_utc.gte"] = since.isoformat()
    return params

def push_to_redis(item: Dict):
    r.lpush(REDIS_QUEUE, json.dumps(item))

def seen_add(stable_id: str) -> bool:
    """
    Returns True if added (i.e., new), False if already present.
    """
    return r.sadd(REDIS_SEEN_SET, stable_id) == 1

def handle_batch(items: Iterable[Dict]) -> int:
    added = 0
    latest_ts: Optional[datetime] = None

    for it in items:
        stable_id, published = select_id(it)
        if not seen_add(stable_id):
            continue

        push_to_redis(it)
        added += 1

        # track high-watermark
        if published:
            try:
                pdt = parse_time(published)
                if latest_ts is None or pdt > latest_ts:
                    latest_ts = pdt
            except Exception:
                pass

    if added:
        log.info("Enqueued %d new articles", added)
    return added, latest_ts

# ------------------ Poll Loops ------------------
def poll_benzinga(last_seen: Optional[datetime]) -> Tuple[int, Optional[datetime]]:
    params = build_benzinga_params(last_seen)
    resp = requests.get(BENZINGA_URL, params=params, timeout=15)
    if resp.status_code == 403:
        raise PermissionError("Benzinga endpoint not enabled on this API key")
    resp.raise_for_status()
    data = resp.json()
    items = data.get("results", [])
    return handle_batch(items)

def poll_reference(last_seen: Optional[datetime]) -> Tuple[int, Optional[datetime]]:
    params = build_ref_params(last_seen)
    resp = requests.get(REF_NEWS_URL, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    items = data.get("results", [])
    return handle_batch(items)

def run():
    stop = False
    def _sigterm(_a, _b):
        nonlocal stop
        stop = True
        log.info("Shutting down...")

    signal.signal(signal.SIGINT, _sigterm)
    signal.signal(signal.SIGTERM, _sigterm)

    log.info("Starting Massive news streamer (tickers=%s, interval=%ss)", TICKERS or "ALL", POLL_INTERVAL)

    use_benzinga = True
    last_seen: Optional[datetime] = None

    # On cold start, you can set last_seen to 'now' to avoid backfill:
    # last_seen = datetime.now(timezone.utc)

    while not stop:
        try:
            if use_benzinga:
                count, high_water = poll_benzinga(last_seen)
            else:
                count, high_water = poll_reference(last_seen)

            if high_water and (not last_seen or high_water > last_seen):
                last_seen = high_water

            time.sleep(POLL_INTERVAL)

        except PermissionError:
            if use_benzinga:
                log.warning("Benzinga not enabled for this key. Falling back to /v2/reference/news (hourly).")
                use_benzinga = False
                continue
        except requests.HTTPError as e:
            log.error("HTTP error: %s | response=%s", e, getattr(e.response, "text", ""))
            time.sleep(min(30, POLL_INTERVAL * 4))
        except Exception as e:
            log.exception("Unexpected error: %s", e)
            time.sleep(min(30, POLL_INTERVAL * 4))

    log.info("Stopped Massive news streamer.")

if __name__ == "__main__":
    run()