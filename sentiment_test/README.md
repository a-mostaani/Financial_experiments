# Sentiment Pipeline

`start.sh` launches all three services of the sentiment pipeline in one command.

## Prerequisites

Before running, make sure the following are running:

- **Redis** on `localhost:6379`
- **PostgreSQL** on `localhost:5432` (database: `newsdb`, user: `newsuser`)

The script will check both and exit with an error if either is unavailable.

## Usage

```bash
cd sentiment_test
./start.sh
```

This starts:

| Service | Script | Log |
|---|---|---|
| News streamer | `news_streamer.py` | `logs/streamer.log` |
| Sentiment worker | `sentiment_worker.py` | `logs/worker.log` |
| Dashboard | `dashboard.py` | `logs/dashboard.log` |

The dashboard is accessible at **http://localhost:8501** once started.

## Stopping

Press `Ctrl+C` — all three processes are stopped cleanly.

## Troubleshooting

Check the log files in `logs/` if a service misbehaves:

```bash
tail -f logs/streamer.log
tail -f logs/worker.log
tail -f logs/dashboard.log
```
