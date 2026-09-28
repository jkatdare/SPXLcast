# One exact Python release on one Debian release, so a rebuild cannot silently change either: the
# plain 3.12-slim tag moves to every new 3.12.x and, as in 2025, to a new Debian release. 3.12.14 on
# Debian 13 (trixie) is what 3.12-slim pointed to when this was pinned. Docker Hub rebuilds this tag
# with Debian security fixes only until the next 3.12.x comes out (the 3.12.13 tag stopped on
# 2026-08-07); after that it never changes. So when a new 3.12.x appears, move up: change the tag
# here and python-version in .github/workflows/deploy.yml, run the tests, commit.
FROM python:3.12.14-slim-trixie

ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    SPXLCAST_DATA=/data \
    MPLBACKEND=Agg \
    SPXLCAST_LIVE=1

WORKDIR /app
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt
COPY spxlcast ./spxlcast
COPY scripts ./scripts
COPY infra/daily.sh ./daily.sh
# Strip CRs: a Windows checkout can hand the build a CRLF daily.sh, which /bin/sh cannot run.
RUN sed -i 's/\r$//' ./daily.sh && chmod +x ./daily.sh && mkdir -p /data/logs /data/output

# The git commit the image is built from (the deploy workflow passes it with --build-arg); it is
# logged with every forecast. Declared last so a new commit does not invalidate the layers above.
ARG SPXLCAST_BUILD=unknown
ENV SPXLCAST_BUILD=${SPXLCAST_BUILD}

# Default command: one daily run (used by the scheduled job). The web app overrides the command
# with `python -m spxlcast serve --root /data --port 8000`; SPXLCAST_LIVE=1 makes it run the
# minute-by-minute live loop as well.
CMD ["./daily.sh"]
