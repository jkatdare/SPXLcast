#!/bin/sh
# One daily run inside the container: forecast + append to the track record + rescore.
# Everything persistent lives under $SPXLCAST_DATA (an Azure Files share mounted at /data).
set -eu
DATA="${SPXLCAST_DATA:-/data}"
mkdir -p "$DATA/logs" "$DATA/output" "$DATA/.cache" "$DATA/archive"
cd /app
# keep the cache on the share too, so history archives survive between job executions
ln -sfn "$DATA/.cache" /app/.cache
echo "[$(date -u +%FT%TZ)] spxlcast daily run"
python -m spxlcast forecast --price 200 --price 250 \
    --log-file "$DATA/logs/forecast_log.csv" \
    --archive "$DATA/archive" \
    --json "$DATA/output/forecast.json" \
    --plot "$DATA/output/fan.png" > "$DATA/output/report.txt" 2>&1 || { cat "$DATA/output/report.txt"; exit 1; }
python -m spxlcast score --log-file "$DATA/logs/forecast_log.csv" --archive "$DATA/archive" \
    > "$DATA/output/score.txt" 2>&1 || true
tail -n 3 "$DATA/output/report.txt"
head -n 4 "$DATA/output/score.txt"
echo "[$(date -u +%FT%TZ)] done"
