#!/bin/sh
# One daily run inside the container: forecast + append to the track record + rescore.
# Everything persistent lives under $SPXLCAST_DATA (an Azure Files share mounted at /data).
# The page's report and score are written to a temp file and renamed into place only when their step
# succeeds, so a run in progress never blanks them and a failed run never replaces them. The temp files
# are unique (a manual run can overlap the scheduled one) and sit at the root of the share, which the
# page does not serve.
set -eu
DATA="${SPXLCAST_DATA:-/data}"
OUT="$DATA/output"
mkdir -p "$DATA/logs" "$OUT" "$DATA/.cache" "$DATA/archive"
cd /app
# keep the cache on the share too, so history archives survive between job executions
ln -sfn "$DATA/.cache" /app/.cache
echo "[$(date -u +%FT%TZ)] spxlcast daily run"
rtmp=$(mktemp "$DATA/report.txt.XXXXXX")
if python -m spxlcast forecast --price 200 --price 250 \
    --log-file "$DATA/logs/forecast_log.csv" \
    --archive "$DATA/archive" \
    --json "$OUT/forecast.json" \
    --plot "$OUT/fan.png" > "$rtmp" 2>&1; then
    mv -f "$rtmp" "$OUT/report.txt"
else
    # kept outside the served folders: a traceback can quote request URLs
    mv -f "$rtmp" "$DATA/last_error.txt"
    cat "$DATA/last_error.txt"
    exit 1
fi
# A failed score keeps the last good track record and does not fail the job (a retry would re-run the
# forecast); it leaves output/score_error.txt (UTC time, then the output's tail) for the health check.
stmp=$(mktemp "$DATA/score.txt.XXXXXX")
if python -m spxlcast score --log-file "$DATA/logs/forecast_log.csv" --archive "$DATA/archive" \
    > "$stmp" 2>&1; then
    mv -f "$stmp" "$OUT/score.txt"
    rm -f "$OUT/score_error.txt"
else
    { date -u +%FT%TZ; tail -n 20 "$stmp"; } > "$OUT/score_error.txt"
    rm -f "$stmp"
    echo "score step FAILED:"
    tail -n 5 "$OUT/score_error.txt"
fi
grep -h '^warning:' "$OUT/report.txt" || true     # a report that could not be rendered in full
tail -n 3 "$OUT/report.txt"
head -n 4 "$OUT/score.txt" 2>/dev/null || true
echo "[$(date -u +%FT%TZ)] done"
