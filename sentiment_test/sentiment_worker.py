

# sentiment_worker.py for alpaca_streamer.py
import json
import redis
import psycopg2
from vllm import LLM, SamplingParams

def analyze_sentiment(llm, params, text):
    prompt = f"Classify this financial news headline as positive, neutral, or negative:\n\n{text}"
    out = llm.generate(prompt, params)
    return out[0].outputs[0].text.strip()

def main():
    llm = LLM("mistralai/Mistral-7B-Instruct-v0.3")
    params = SamplingParams(max_tokens=50)

    redis_client = redis.Redis(host="localhost", port=6379, db=0)

    conn = psycopg2.connect(
        dbname="newsdb",
        user="newsuser",
        password="news123",
        host="localhost",
        port=5432
    )
    cursor = conn.cursor()

    while True:
        item = redis_client.rpop("news_queue")
        if not item:
            continue

        data = json.loads(item)

        headline = data.get("headline", "")
        summary = data.get("summary", "")
        text = headline or summary

        url = data.get("url", "")

        symbols = data.get("symbols", [])
        ticker = ",".join(symbols) if symbols else None

        sentiment = analyze_sentiment(llm, params, text)

        cursor.execute(
            """INSERT INTO sentiment (ticker, headline, url, sentiment, confidence)
               VALUES (%s, %s, %s, %s, %s)""",
            (ticker, headline, url, sentiment, 1.0)
        )
        conn.commit()

        print(f"Stored: {headline[:80]}  →  {sentiment}")

if __name__ == "__main__":
    main()




#sentiment_worker.py for massive_streamer.py
# sentiment_worker.py adapted for Massive (Benzinga + Reference News)
import json
import redis
import psycopg2
from vllm import LLM, SamplingParams
from dateutil import parser as dtparse


def analyze_sentiment(llm, params, text):
    prompt = f"Classify this financial news headline as positive, neutral, or negative:\n\n{text}"
    out = llm.generate(prompt, params)
    return out[0].outputs[0].text.strip()


# ----------------------------
# Unified normalizer for:
# - Massive Benzinga real-time feed
# - Massive Reference News feed
# - Alpaca fallback
# ----------------------------
def normalize_news(data: dict):
    # Massive Benzinga (real-time)
    if "benzinga_id" in data:
        headline = data.get("title") or ""
        summary = data.get("teaser") or ""
        body    = data.get("body") or ""
        url     = data.get("url") or ""
        tickers = data.get("tickers") or []

        return {
            "headline": headline,
            "summary": summary,
            "body": body,
            "text_for_sentiment": headline or summary or body,
            "url": url,
            "tickers": tickers
        }

    # Massive Reference News (hourly fallback)
    if "article_url" in data or "published_utc" in data:
        headline = data.get("title") or ""
        summary = data.get("description") or ""
        body    = data.get("content") or ""
        url     = data.get("article_url") or ""
        tickers = data.get("tickers") or []

        return {
            "headline": headline,
            "summary": summary,
            "body": body,
            "text_for_sentiment": headline or summary or body,
            "url": url,
            "tickers": tickers
        }

    # Alpaca fallback
    headline = data.get("headline") or data.get("title") or ""
    summary = data.get("summary") or data.get("description") or ""
    body    = data.get("content") or ""
    url     = data.get("url") or ""
    tickers = data.get("symbols") or data.get("tickers") or []

    return {
        "headline": headline,
        "summary": summary,
        "body": body,
        "text_for_sentiment": headline or summary or body,
        "url": url,
        "tickers": tickers
    }


def main():
    llm = LLM("mistralai/Mistral-7B-Instruct-v0.3")
    params = SamplingParams(max_tokens=50)

    redis_client = redis.Redis(host="localhost", port=6379, db=0)

    conn = psycopg2.connect(
        dbname="newsdb",
        user="newsuser",
        password="news123",
        host="localhost",
        port=5432
    )
    cursor = conn.cursor()

    while True:
        item = redis_client.rpop("news_queue")
        if not item:
            continue

        data = json.loads(item)
        norm = normalize_news(data)

        headline = norm["headline"]
        url = norm["url"]
        ticker_list = norm["tickers"]
        ticker = ",".join(ticker_list) if ticker_list else None

        text = norm["text_for_sentiment"]
        if not text:
            # skip empty payloads
            continue

        sentiment = analyze_sentiment(llm, params, text)

        cursor.execute(
            """INSERT INTO sentiment (ticker, headline, url, sentiment, confidence)
               VALUES (%s, %s, %s, %s, %s)""",
            (ticker, headline, url, sentiment, 1.0)
        )
        conn.commit()

        print(f"Stored: {headline[:80]}  →  {sentiment}")


if __name__ == "__main__":
    main()