from datetime import datetime, timedelta, timezone

from spxlcast.config import Config
from spxlcast.data import NewsItem
from spxlcast.sentiment import analyze_news, score_article, make_analyzer, sentiment_label

NOW = datetime(2026, 9, 15, 12, 0, tzinfo=timezone.utc)


def item(ticker, title, summary="", hours_ago=1.0):
    return NewsItem(ticker=ticker, title=title, summary=summary,
                    published=NOW - timedelta(hours=hours_ago), provider="test", url="")


def test_finance_lexicon_shapes_scores():
    an = make_analyzer()
    good = score_article(an, item("SPY", "Stocks rally to record high as earnings beat estimates"))
    bad = score_article(an, item("SPY", "Stocks plunge as recession fears and tariffs rattle markets"))
    assert good > 0.3
    assert bad < -0.3


def test_labels():
    assert sentiment_label(0.5) == "Strongly bullish"
    assert sentiment_label(0.2) == "Bullish"
    assert sentiment_label(0.0) == "Neutral"
    assert sentiment_label(-0.2) == "Bearish"
    assert sentiment_label(-0.6) == "Strongly bearish"


def test_aggregate_weights_recency_and_dedupes():
    cfg = Config()
    news = [
        item("SPY", "Markets surge on strong growth and upgrades", hours_ago=1),
        item("^GSPC", "Markets surge on strong growth and upgrades", hours_ago=1),  # duplicate title
        item("SPY", "Markets crash in worst selloff since the recession", hours_ago=24 * 30),  # too old
        item("NVDA", "Chipmaker downgraded on weak demand warning", hours_ago=2),
    ]
    res = analyze_news(news, cfg, holdings_weights={"NVDA": 0.08}, now=NOW)
    assert res.n_articles == 4
    assert res.n_used == 2  # duplicate removed, stale article dropped
    assert "SPY" in res.by_ticker and "NVDA" in res.by_ticker
    assert res.by_ticker["SPY"] > 0 > res.by_ticker["NVDA"]
    # SPY carries full weight, NVDA 0.48 -> aggregate stays positive
    assert res.score > 0
    assert res.drift_adjustment == cfg.sentiment_drift_scale * res.score
    assert res.top_positive and res.top_negative


def test_no_news_is_neutral():
    res = analyze_news([], Config(), now=NOW)
    assert res.score == 0.0 and res.label == "Neutral" and res.drift_adjustment == 0.0
