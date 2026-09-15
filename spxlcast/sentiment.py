"""News sentiment for the S&P 500 / SPXL from Yahoo Finance headlines.

Scoring: VADER (lexicon + rules) extended with a small finance vocabulary. Each article is
weighted by recency (exponential half-life) and by how much the ticker matters to the index.
The aggregate score in [-1, 1] nudges the *near-term* drift of the simulation only.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

from .config import Config
from .data import NewsItem

try:  # optional at import time so tests can run without it
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:  # pragma: no cover
    SentimentIntensityAnalyzer = None  # type: ignore

# Finance-flavoured additions to VADER's lexicon (VADER scale is roughly -4 .. +4).
FINANCE_LEXICON: Dict[str, float] = {
    "beat": 1.5, "beats": 1.5, "miss": -1.5, "misses": -1.5, "missed": -1.5,
    "upgrade": 2.0, "upgrades": 2.0, "upgraded": 2.0,
    "downgrade": -2.0, "downgrades": -2.0, "downgraded": -2.0,
    "rally": 1.8, "rallies": 1.8, "rallied": 1.8, "surge": 2.0, "surges": 2.0, "surged": 2.0,
    "soar": 2.2, "soars": 2.2, "soared": 2.2, "jump": 1.5, "jumps": 1.5, "jumped": 1.5,
    "climb": 1.2, "climbs": 1.2, "climbed": 1.2, "rise": 1.2, "rises": 1.2, "rose": 1.2,
    "gain": 1.0, "gains": 1.2, "gained": 1.2, "outperform": 1.5, "outperforms": 1.5,
    "bullish": 2.0, "optimism": 1.5, "optimistic": 1.5, "resilient": 1.2, "robust": 1.2,
    "record": 0.8, "boom": 1.5, "growth": 0.8, "recovery": 1.2, "rebound": 1.5, "rebounds": 1.5,
    "plunge": -2.5, "plunges": -2.5, "plunged": -2.5, "crash": -3.0, "crashes": -3.0,
    "selloff": -2.2, "sell-off": -2.2, "tumble": -2.0, "tumbles": -2.0, "tumbled": -2.0,
    "slump": -2.0, "slumps": -2.0, "slide": -1.5, "slides": -1.5, "slid": -1.5,
    "fall": -1.2, "falls": -1.2, "fell": -1.2, "drop": -1.2, "drops": -1.2, "dropped": -1.2,
    "decline": -1.2, "declines": -1.2, "declined": -1.2, "loss": -1.0, "losses": -1.2,
    "underperform": -1.5, "underperforms": -1.5, "bearish": -2.0, "pessimism": -1.5,
    "recession": -2.2, "stagflation": -2.0, "inflation": -0.8, "tariff": -1.0, "tariffs": -1.0,
    "layoffs": -1.5, "bankruptcy": -3.0, "default": -2.0, "defaults": -2.0,
    "hike": -1.0, "hikes": -1.0, "jitters": -1.5, "rattle": -1.5, "rattles": -1.5, "rattled": -1.5,
    "warns": -1.5, "warning": -1.5, "fear": -1.5, "fears": -1.5, "volatility": -0.8,
    "correction": -1.5, "bubble": -1.5, "slashes": -1.8, "cuts": -0.8, "weak": -1.0, "weakness": -1.2,
    "downturn": -2.0, "contraction": -1.8, "shutdown": -1.5, "sanctions": -1.0,
}

MARKET_TICKER_WEIGHT = 1.0


@dataclass
class ScoredNews:
    item: NewsItem
    score: float      # VADER compound in [-1, 1]
    weight: float     # recency x relevance


@dataclass
class SentimentResult:
    score: float                       # weighted mean compound score, [-1, 1]
    label: str
    n_articles: int
    n_used: int
    by_ticker: Dict[str, float]
    drift_adjustment: float            # annualised drift shift applied to the first N days
    top_positive: List[ScoredNews] = field(default_factory=list)
    top_negative: List[ScoredNews] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


def make_analyzer():
    if SentimentIntensityAnalyzer is None:
        return None
    analyzer = SentimentIntensityAnalyzer()
    analyzer.lexicon.update(FINANCE_LEXICON)
    return analyzer


def score_text(analyzer, text: str) -> float:
    if analyzer is None or not text:
        return 0.0
    return float(analyzer.polarity_scores(text)["compound"])


def score_article(analyzer, item: NewsItem) -> float:
    """Title carries most of the signal; the summary refines it."""
    t = score_text(analyzer, item.title)
    if item.summary:
        s = score_text(analyzer, item.summary)
        return 0.6 * t + 0.4 * s
    return t


def sentiment_label(score: float) -> str:
    if score >= 0.4:
        return "Strongly bullish"
    if score >= 0.15:
        return "Bullish"
    if score <= -0.4:
        return "Strongly bearish"
    if score <= -0.15:
        return "Bearish"
    return "Neutral"


def _norm_title(title: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", title.lower()).strip()


def analyze_news(
    news: List[NewsItem],
    cfg: Config,
    holdings_weights: Optional[Dict[str, float]] = None,
    now: Optional[datetime] = None,
    analyzer=None,
) -> SentimentResult:
    now = now or datetime.now(timezone.utc)
    analyzer = analyzer or make_analyzer()
    holdings_weights = holdings_weights or {}
    notes: List[str] = []
    if analyzer is None:
        notes.append("vaderSentiment not installed: sentiment neutral")

    seen = set()
    scored: List[ScoredNews] = []
    per_ticker: Dict[str, List[tuple]] = {}
    for item in news:
        key = _norm_title(item.title)
        if not key or key in seen:
            continue
        seen.add(key)
        age_days = max(0.0, (now - item.published).total_seconds() / 86400.0)
        if age_days > cfg.news_max_age_days:
            continue
        recency = 0.5 ** (age_days / cfg.news_half_life_days)
        if item.ticker in cfg.news_tickers:
            relevance = MARKET_TICKER_WEIGHT
        else:
            relevance = min(1.0, 6.0 * float(holdings_weights.get(item.ticker, 0.02)))
        w = recency * relevance
        s = score_article(analyzer, item)
        scored.append(ScoredNews(item=item, score=s, weight=w))
        per_ticker.setdefault(item.ticker, []).append((s, w))

    if not scored:
        return SentimentResult(score=0.0, label="Neutral", n_articles=len(news), n_used=0, by_ticker={},
                               drift_adjustment=0.0, notes=notes + ["no recent articles"])

    weights = np.array([x.weight for x in scored])
    scores = np.array([x.score for x in scored])
    agg = float(np.sum(weights * scores) / np.sum(weights)) if weights.sum() > 0 else 0.0
    by_ticker = {}
    for t, pairs in per_ticker.items():
        ws = np.array([p[1] for p in pairs])
        ss = np.array([p[0] for p in pairs])
        by_ticker[t] = float(np.sum(ws * ss) / np.sum(ws)) if ws.sum() > 0 else 0.0

    ranked = sorted(scored, key=lambda x: x.score)
    top_negative = [x for x in ranked[:5] if x.score < 0]
    top_positive = [x for x in ranked[::-1][:5] if x.score > 0]
    drift = cfg.sentiment_drift_scale * agg
    notes.append(f"{len(scored)} unique articles within {cfg.news_max_age_days:.0f} days, "
                 f"half-life {cfg.news_half_life_days:.0f} days")
    return SentimentResult(score=agg, label=sentiment_label(agg), n_articles=len(news), n_used=len(scored),
                           by_ticker=by_ticker, drift_adjustment=drift, top_positive=top_positive,
                           top_negative=top_negative, notes=notes)
