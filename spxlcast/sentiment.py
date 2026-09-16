"""News sentiment for the S&P 500 / SPXL from Yahoo Finance headlines.

Scoring: VADER (lexicon + rules) with a finance vocabulary and a subject-aware phrase layer
("Fed cuts rates" is bullish although "cuts" is negative; "yields jump" is bearish although
"jump" is positive; a subject only binds to a verb inside its own clause, so "stocks rise as
oil falls" is bullish). Each unique story is weighted by recency (exponential half-life) and by
relevance judged from its headline, not from which ticker feed it arrived on: market-wide
stories count fully on any feed, single-company stories count in proportion to the company's
index weight, and stories about neither are scaled down further.

The aggregate score in [-1, 1] tilts only the near-term drift of the displayed distribution.
It is excluded from the rating (see pipeline.run_forecast) because the mapping from news tone
to returns is not calibrated.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import Config
from .data import NewsItem

try:  # optional at import time so tests can run without it
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:  # pragma: no cover
    SentimentIntensityAnalyzer = None  # type: ignore

# Finance vocabulary (VADER scale, roughly -4 .. +4). Entries only override VADER's own value when
# they are at least as strong, change the sign, or deliberately neutralise a word (0.0).
FINANCE_LEXICON: Dict[str, float] = {
    "beat": 1.5, "beats": 1.5, "miss": -1.5, "misses": -1.5, "missed": -1.5,
    "upgrade": 2.0, "upgrades": 2.0, "upgraded": 2.0,
    "downgrade": -2.0, "downgrades": -2.0, "downgraded": -2.0,
    "rally": 1.8, "rallies": 1.8, "rallied": 1.8, "surge": 2.0, "surges": 2.0, "surged": 2.0,
    "soar": 2.2, "soars": 2.2, "soared": 2.2, "jump": 1.5, "jumps": 1.5, "jumped": 1.5,
    "climb": 1.2, "climbs": 1.2, "climbed": 1.2, "rise": 1.2, "rises": 1.2, "rose": 1.2,
    "gain": 1.0, "gains": 1.2, "gained": 1.2, "outperform": 1.5, "outperforms": 1.5,
    "bullish": 2.0, "optimism": 1.5, "optimistic": 1.5, "resilient": 1.2, "robust": 1.2,
    "boom": 1.5, "growth": 0.8, "recovery": 1.2, "rebound": 1.5, "rebounds": 1.5, "rebounded": 1.5,
    "plunge": -2.5, "plunges": -2.5, "plunged": -2.5, "crash": -3.0, "crashes": -3.0,
    "selloff": -2.2, "sell-off": -2.2, "tumble": -2.0, "tumbles": -2.0, "tumbled": -2.0,
    "slump": -2.0, "slumps": -2.0, "slide": -1.5, "slides": -1.5, "slid": -1.5,
    "fall": -1.2, "falls": -1.2, "fell": -1.2, "drop": -1.2, "drops": -1.2, "dropped": -1.2,
    "decline": -1.2, "declines": -1.2, "declined": -1.2, "loss": -1.0, "losses": -1.2,
    "slip": -1.2, "slips": -1.2, "slipped": -1.2, "dive": -2.0, "dives": -2.0, "dived": -2.0,
    "sink": -1.8, "sinks": -1.8, "sank": -1.8, "tank": -2.2, "tanks": -2.2, "tanked": -2.2,
    "retreat": -1.2, "retreats": -1.2, "retreated": -1.2, "lower": -1.0, "spike": -0.5, "spikes": -0.5,
    "stumble": -1.5, "stumbles": -1.5, "stumbled": -1.5, "wobble": -1.2, "wobbles": -1.2,
    "slows": -1.2, "slowing": -1.0, "slowdown": -1.5, "stall": -1.2, "stalls": -1.5, "elevated": -0.5,
    "underperform": -1.5, "underperforms": -1.5, "bearish": -2.0, "pessimism": -1.5,
    "recession": -2.2, "stagflation": -2.0, "inflation": -0.8, "tariff": -1.0, "tariffs": -1.0,
    "layoffs": -1.5, "bankruptcy": -3.0, "disorderly": -1.8,
    "hike": -1.0, "hikes": -1.0, "jitters": -1.5, "rattle": -1.5, "rattles": -1.5, "rattled": -1.5,
    "warns": -1.5, "warning": -1.5, "fear": -1.5, "fears": -1.5, "volatility": -0.8,
    "correction": -1.5, "bubble": -1.5, "slashes": -1.8, "cuts": -0.8, "weak": -1.0, "weakness": -1.2,
    "downturn": -2.0, "contraction": -1.8, "shutdown": -1.5, "sanctions": -1.0,
    # finance nouns VADER reads as positive but which carry no tone in headlines
    "interest": 0.0, "interests": 0.0, "asset": 0.0, "assets": 0.0, "share": 0.0, "shares": 0.0,
    "treasury": 0.0, "treasuries": 0.0, "intelligence": 0.0, "securities": 0.0, "capital": 0.0,
    "credit": 0.0, "free": 0.0, "play": 0.0, "plays": 0.0, "record": 0.0,
}

# Subject-aware phrase rules, applied to lower-cased text before VADER. Each match is replaced by
# a sentinel token that carries the intended polarity for the stock market. A subject can only
# bind to a verb inside its own clause (the gap stops at conjunctions and punctuation).
_BREAK = r"(?:as|while|after|but|despite|amid|even as|before|because|since|though|although|whereas|when)"
_GAP = r"(?:(?!\b" + _BREAK + r"\b)[^.,;:!?]){0,30}?"
_SUBJ_MARKET = r"(?:stocks?|shares|equities|wall street|dow|nasdaq|s&p(?: 500)?|futures|indexes|indices|markets?)"
_SUBJ_VIX = r"(?:vix|fear (?:index|gauge)|volatility(?: index)?|wall street's fear gauge)"
_DOWN = r"(?:fall|falls|fell|drop|drops|dropped|slide|slides|slid|sink|sinks|sank|dive|dives|dived|tumble|tumbles|tumbled|slump|slumps|slumped|plunge|plunges|plunged|retreat|retreats|slip|slips|slipped|decline|declines|declined|stumble|stumbles|lower|down(?= \d| sharply| again)|sell off|selloff)"
_UP = r"(?:rise|rises|rose|gain|gains|gained|climb|climbs|climbed|rally|rallies|rallied|jump|jumps|jumped|surge|surges|surged|advance|advances|advanced|higher|up(?= \d| sharply| again)|soar|soars|soared|rebound|rebounds|rebounded)"
_YIELD_UP = _UP[3:-1] + r"|spike|spikes|spiked|touch(?:es|ing)?(?: the)? highest|cross(?:es|ing)? above|scale|hits? \d|push(?:es|ing)? past|tops? \d|topped \d|above \d|stays? elevated|remains? elevated|elevated|disorderly rise"
PHRASE_RULES: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\bfall (?:season|semester|fashion|collection|schedule|lineup)\b"), "autumn"),
    (re.compile(r"\brate[- ]cuts?\b|\bcuts? (?:its |interest |benchmark |key |policy )?rates?\b|\b(?:cut|cutting|lower|lowers|lowered|lowering|slash|slashes|slashed) (?:interest |benchmark |key |policy )?rates?\b"), "xratecut"),
    (re.compile(r"\brate[- ]hikes?\b|\b(?:hike|hikes|hiked|raise|raises|raised|raising|lift|lifts|lifted) (?:interest |benchmark |key |policy )?rates?\b"), "xratehike"),
    (re.compile(r"\b(?:jobless|unemployment) claims\b" + _GAP + r"\b(?:fall|falls|fell|drop|drops|dropped|decline|declines|declined|record low|lowest|multi-year low)[^.,;:!?]{0,15}"), "xclaimsdown"),
    (re.compile(r"\b(?:recession|inflation|tariff|trade|rate|growth|war|default|debt|credit|banking)?\s*(?:fears?|worries|concerns?|jitters|anxiety)\b" + _GAP + r"\b(?:ease|eases|eased|fade|fades|faded|recede|recedes|receded|subside|subsides|subsided)\b"), "xfearsease"),
    (re.compile(r"\byields?\b" + _GAP + r"\b(?:" + _YIELD_UP + r")\b"), "xyieldsup"),
    (re.compile(r"\b(?:" + _YIELD_UP.replace("|elevated", "") + r")\b" + _GAP + r"\byields?\b"), "xyieldsup"),
    (re.compile(r"\byields?\b" + _GAP + r"\b(?:" + _DOWN[3:-1] + r"|ease|eases|eased|dip|dips|dipped)\b"), "xyieldsdown"),
    (re.compile(r"\b(?:treasur(?:y|ies)|bonds?)\b" + _GAP + r"\b(?:cross(?:es|ing)? above \d|tops? \d|topped \d|hits? \d|above \d|highest yield)"), "xyieldsup"),
    (re.compile(r"\binflation\b" + _GAP + r"\b(?:falls?|fell|cools?|cooled|cooling|eases?|eased|easing|slows?|slowed|slowing|drops?|dropped|declines?|declined)\b"), "xinflationcool"),
    (re.compile(r"\binflation\b" + _GAP + r"\b(?:rises?|rose|jumps?|jumped|heats?|heated|accelerat\w*|surges?|surged|hotter|sticky|stubborn|re-?accelerat\w*)\b"), "xinflationhot"),
    (re.compile(r"\brecord highs?\b|\ball[- ]time highs?\b|\bfresh highs?\b|\bnew highs?\b"), "xrecordhigh"),
    (re.compile(r"\b(?:oil|crude)\b" + _GAP + r"\b(?:surges?|surged|jumps?|jumped|spikes?|spiked|soars?|soared|rall(?:y|ies|ied)|climbs?|climbed)\b"), "xoilup"),
    (re.compile(r"\b(?:growth|economy|hiring|spending|sales|demand|manufacturing|output)\b" + _GAP + r"\b(?:slows?|slowed|slowing|stalls?|stalled|weakens?|weakened|contracts?|contracted|shrinks?|shrank|cools?|cooled|falters?|faltered)\b"), "xgrowthslows"),
    (re.compile(r"\b(?:growth|economy|hiring|spending|sales|demand|manufacturing|output)\b" + _GAP + r"\b(?:accelerates?|accelerated|picks? up|picked up|rebounds?|rebounded|strengthens?|strengthened|expands?|expanded)\b"), "xgrowthup"),
    (re.compile(r"\bafter (?:a |the |last week's |yesterday's )?(?:selloff|sell-off|rout|slump|slide|drop|losses|plunge)\b"), "xafterselloff"),
    (re.compile(r"\b" + _SUBJ_VIX + r"\b" + _GAP + r"\b(?:" + _UP[3:-1] + r"|spike|spikes|spiked)\b"), "xvixup"),
    (re.compile(r"\b" + _SUBJ_VIX + r"\b" + _GAP + r"\b" + _DOWN + r"\b"), "xvixdown"),
    (re.compile(r"\b" + _SUBJ_MARKET + r"\b" + _GAP + r"\b" + _DOWN + r"\b"), "xstocksdown"),
    (re.compile(r"\b" + _SUBJ_MARKET + r"\b" + _GAP + r"\b" + _UP + r"\b"), "xstocksup"),
]
PHRASE_TOKENS: Dict[str, float] = {
    "xratecut": 2.0, "xratehike": -2.0, "xyieldsup": -1.5, "xyieldsdown": 1.0,
    "xinflationcool": 1.5, "xinflationhot": -1.5, "xrecordhigh": 1.5, "xoilup": -1.0,
    "xfearsease": 1.5, "xclaimsdown": 1.0, "xstocksdown": -2.0, "xstocksup": 2.0,
    "xvixup": -1.5, "xvixdown": 1.0, "autumn": 0.0, "xgrowthslows": -1.5, "xgrowthup": 1.5, "xafterselloff": 0.3,
}

# A story is market-wide (relevance 1.0 on any feed) if its title names a market-level subject
# and does not look like a single-stock piece.
MARKET_RE = re.compile(
    r"\b(?:stock market|markets|broad market|the market|market (?:rally|selloff|sell-off|rout|slump|falls?|rises?|drops?)|"
    r"stocks|equities|wall street|dow|nasdaq|s&p(?: 500)?|spx|russell|futures|fed|fomc|"
    r"federal reserve|powell|treasur(?:y|ies)|yields?|bond market|bonds|rate (?:hikes?|cuts?)|interest rates?|"
    r"inflation|cpi|pce|payrolls|jobs report|jobless|unemployment|gdp|recession|tariffs?|trade war|oil prices|"
    r"crude|the dollar|dollar index|vix|volatility|economy|economic|risk-?off|risk-?on|sell-?off|correction|"
    r"bear market|bull market|indexes|indices)\b", re.I)
_SINGLE_LOWER = re.compile(
    r"\(\w{1,5}\)|\bvs\.?\b|\b\d+ (?:reasons|stocks|top)\b|better (?:buy|investment)|stocks? to (?:buy|watch|own|sell)|"
    r"\bis (?:it )?a buy\b|price target|upgraded|downgraded|earnings (?:beat|miss|call|report|preview)|"
    r"\bstock (?:tops|jumps|falls|rises|drops|soars|plunges|climbs|slides|gains|sinks|rallies|surges|slips|tumbles)\b", re.I)
_SINGLE_CAP = re.compile(r"\b[A-Z][A-Za-z&.'-]+ (?:Stock|Shares) (?:Tops|Jumps|Falls|Rises|Drops|Soars|Plunges|Climbs|"
                         r"Slides|Gains|Sinks|Rallies|Surges|Slips|Tumbles)\b")
BOILERPLATE_RE = re.compile(r"\binvestor letter\b|can be downloaded|click here|\bsubscribe\b|\bsign up\b|\bread more\b", re.I)


def is_market_wide(title: str) -> bool:
    return bool(MARKET_RE.search(title)) and not (_SINGLE_LOWER.search(title) or _SINGLE_CAP.search(title))


@dataclass
class ScoredNews:
    item: NewsItem
    score: float      # VADER compound in [-1, 1]
    weight: float     # recency x relevance
    feeds: List[str] = field(default_factory=list)


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
    lex = analyzer.lexicon
    for word, val in FINANCE_LEXICON.items():
        old = lex.get(word)
        if old is None or val == 0.0 or abs(val) >= abs(old) or (old > 0) != (val > 0):
            lex[word] = val
    lex.update(PHRASE_TOKENS)
    return analyzer


def apply_phrase_rules(text: str) -> str:
    text = text.lower()
    for pattern, token in PHRASE_RULES:
        text = pattern.sub(f" {token} ", text)
    return text


def score_text(analyzer, text: str) -> float:
    if analyzer is None or not text:
        return 0.0
    return float(analyzer.polarity_scores(apply_phrase_rules(text))["compound"])


def score_article(analyzer, item: NewsItem, summary_weight: float = 0.2, use_summary: bool = True) -> float:
    """Title carries the signal; the teaser summary refines it slightly (it is often boilerplate)."""
    t = score_text(analyzer, item.title)
    if use_summary and item.summary and not BOILERPLATE_RE.search(item.summary):
        s = score_text(analyzer, item.summary)
        return (1.0 - summary_weight) * t + summary_weight * s
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


def _company_aliases(holdings_names: Dict[str, str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """(symbol -> canonical symbol for share classes, symbol -> lower-case company keyword)."""
    canon: Dict[str, str] = {}
    keyword: Dict[str, str] = {}
    by_root: Dict[str, str] = {}
    for sym, name in holdings_names.items():
        cleaned = re.sub(r"\b(inc|corp|corporation|co|ltd|plc|class [abc]|holdings?)\b\.?", "", str(name).lower())
        words = [w for w in re.split(r"[^a-z0-9]+", cleaned) if w]
        root = " ".join(words)
        canon[sym] = by_root.setdefault(root, sym) if root else sym
        keyword[sym] = (words or [sym.lower()])[0]
    return canon, keyword


def analyze_news(
    news: List[NewsItem],
    cfg: Config,
    holdings_weights: Optional[Dict[str, float]] = None,
    now: Optional[datetime] = None,
    analyzer=None,
    holdings_names: Optional[Dict[str, str]] = None,
) -> SentimentResult:
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    analyzer = analyzer or make_analyzer()
    holdings_weights = holdings_weights or {}
    canon, keyword = _company_aliases(holdings_names or {})
    market_feeds = set(cfg.news_tickers)
    notes: List[str] = []
    if analyzer is None:
        notes.append("vaderSentiment not installed: sentiment neutral")

    # Pass 1: keep only fresh items (drop stale and future-dated); count the distinct stories that
    # share a summary so recycled boilerplate can be ignored.
    fresh = []
    for it in news:
        age = (now - it.published).total_seconds() / 86400.0
        if -1.0 <= age <= cfg.news_max_age_days:
            fresh.append((it, max(0.0, age)))
    titles_by_summary: Dict[str, set] = {}
    for it, _ in fresh:
        if it.summary:
            titles_by_summary.setdefault(it.summary, set()).add(_norm_title(it.title))

    def about_company(it: NewsItem) -> bool:
        title = it.title.lower()
        kw = keyword.get(it.ticker)
        if kw and re.search(rf"\b{re.escape(kw)}\b", title):
            return True
        return len(it.ticker) >= 3 and re.search(rf"\b{re.escape(it.ticker)}\b", it.title) is not None

    def relevance(it: NewsItem) -> float:
        if is_market_wide(it.title):
            return 1.0
        if it.ticker in market_feeds:
            return cfg.offtopic_weight
        sym = canon.get(it.ticker, it.ticker)
        base = min(1.0, 6.0 * float(holdings_weights.get(sym, holdings_weights.get(it.ticker, 0.02))))
        return base if about_company(it) else cfg.offtopic_weight * base

    # Pass 2: score every occurrence (for the per-ticker view), dedupe by title for the aggregate.
    per_ticker: Dict[str, List[Tuple[float, float]]] = {}
    unique: Dict[str, ScoredNews] = {}
    for it, age in fresh:
        key = _norm_title(it.title)
        if not key:
            continue
        recency = 0.5 ** (age / cfg.news_half_life_days)
        w = recency * relevance(it)
        use_summary = len(titles_by_summary.get(it.summary, ())) <= 1
        s = score_article(analyzer, it, cfg.summary_weight, use_summary)
        tick = canon.get(it.ticker, it.ticker)
        per_ticker.setdefault(tick, []).append((s, w))
        if key in unique:
            best = unique[key]
            best.weight = max(best.weight, w)          # relevance = max over the feeds it appeared in
            best.feeds.append(it.ticker)
            if it.published > best.item.published:     # keep the freshest copy's text and score
                best.item, best.score = it, s
        else:
            unique[key] = ScoredNews(item=it, score=s, weight=w, feeds=[it.ticker])

    scored = list(unique.values())
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

    ranked = sorted(scored, key=lambda x: x.score * x.weight)
    top_negative = [x for x in ranked[:5] if x.score < 0]
    top_positive = [x for x in ranked[::-1][:5] if x.score > 0]
    drift = cfg.sentiment_drift_scale * agg
    n_market = sum(1 for x in scored if x.weight >= 0.5)
    notes.append(f"{len(scored)} unique stories within {cfg.news_max_age_days:.0f} days "
                 f"({n_market} weighted as market-relevant), half-life {cfg.news_half_life_days:.0f} days")
    return SentimentResult(score=agg, label=sentiment_label(agg), n_articles=len(news), n_used=len(scored),
                           by_ticker=by_ticker, drift_adjustment=drift, top_positive=top_positive,
                           top_negative=top_negative, notes=notes)
