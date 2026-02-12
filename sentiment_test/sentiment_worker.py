import json
import redis
import psycopg2
from vllm import LLM, SamplingParams

def analyze_sentiment(llm, params, headline):
    prompt = f"Classify this financial news headline as positive, neutral, or negative:\n\n{headline}"
    out = llm.generate(prompt, params)
    return out[0].outputs[0].text.strip()

def main():
    # Initialize LLM INSIDE main()
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
        headline = data.get("title", "")
        url = data.get("url", "")
        ticker = ",".join(data.get("tickers", []))

        sentiment = analyze_sentiment(llm, params, headline)

        cursor.execute(
            "INSERT INTO sentiment (ticker, headline, url, sentiment, confidence) VALUES (%s, %s, %s, %s, %s)",
            (ticker, headline, url, sentiment, 1.0)
        )
        conn.commit()

        print(f"Stored: {headline} -> {sentiment}")

if __name__ == "__main__":
    main()