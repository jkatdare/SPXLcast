FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    SPXLCAST_DATA=/data \
    MPLBACKEND=Agg \n    SPXLCAST_LIVE=1

WORKDIR /app
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt
COPY spxlcast ./spxlcast
COPY scripts ./scripts
COPY infra/daily.sh ./daily.sh
RUN chmod +x ./daily.sh && mkdir -p /data/logs /data/output

# Default command: one daily run (used by the scheduled job). The web app overrides the command
# with `python -m spxlcast serve --root /data --port 8000`; SPXLCAST_LIVE=1 makes it run the
# minute-by-minute live loop as well.
CMD ["./daily.sh"]
