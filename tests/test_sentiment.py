from datetime import datetime, timedelta, timezone

from spxlcast.config import Config
from spxlcast.data import NewsItem
from spxlcast.sentiment import (analyze_news, apply_phrase_rules, is_market_wide, make_analyzer, score_article,
                                score_text, sentiment_label)

NOW = datetime(2026, 9, 15, 12, 0, tzinfo=timezone.utc)
AN = make_analyzer()


def item(ticker, title, summary="", hours_ago=1.0):
    return NewsItem(ticker=ticker, title=title, summary=summary,
                    published=NOW - timedelta(hours=hours_ago), provider="test", url="")


def test_finance_lexicon_shapes_scores():
    good = score_article(AN, item("SPY", "Stocks rally to record high as earnings beat estimates"))
    bad = score_article(AN, item("SPY", "Stocks plunge as recession fears and tariffs rattle markets"))
    assert good > 0.3
    assert bad < -0.3


def test_subject_aware_phrase_rules():
    assert "xratecut" in apply_phrase_rules("Fed cuts rates by a quarter point")
    assert "xyieldsup" in apply_phrase_rules("Treasury yields jump to 5%")
    assert score_text(AN, "Fed cuts rates by a quarter point") > 0
    assert score_text(AN, "Fed hikes rates again") < 0
    assert score_text(AN, "Treasury yields jump to 5%") < 0
    assert score_text(AN, "Inflation falls to 2%") > 0
    assert score_text(AN, "Dow dives as yields jump") < 0
    assert score_text(AN, "Recession fears ease as jobless claims drop") > 0
    assert score_text(AN, "Global shares fall as Treasury yields scale fresh peaks") < 0
    assert score_text(AN, "Stocks rally as Fed cuts rates") > 0.3


def test_rules_are_clause_bounded_and_cover_the_audit_cases():
    bullish = [
        "Stocks rise as oil falls", "Stocks climb as oil drops", "Stocks gain as dollar slips",
        "Stocks rally as volatility drops", "Yields drop as stocks rally", "Stocks rebound after selloff",
        "Recession fears ease", "Jobless claims drop to record low", "Inflation cools more than expected",
    ]
    bearish = [
        "Growth slows sharply", "Dow Falls 500 Points After 10-Year Yield Tops 5%",
        "Chip stocks fall as oil prices gain, Treasury yields stay elevated",
        "Investors See Disorderly Rise in Bond Yields as the Main Risk", "VIX spikes as stocks tumble",
        "Wall Street's fear gauge jumps to a six-month high", "Stocks stumble after Labor Department data",
    ]
    for h in bullish:
        assert score_text(AN, h) > 0, h
    for h in bearish:
        assert score_text(AN, h) < 0, h
    assert score_text(AN, "Retailers gear up for the fall season") >= 0


def test_market_wide_detection():
    assert is_market_wide("Dow Jones falls as Treasury yields climb")
    assert is_market_wide("Stocks stumble after Labor Department data")
    assert not is_market_wide("Skyworks Solutions Stock Tops S&P 500. The Apple Supplier Picks Up Where It Left Off.")
    assert not is_market_wide("Is Nvidia (NVDA) a better buy than AMD?")
    assert not is_market_wide("Which solar stock is a better buy in 2026")


def test_vader_strong_words_are_not_weakened():
    lex = AN.lexicon
    assert lex["gain"] >= 2.0        # VADER's stronger value kept
    assert lex["crash"] == -3.0      # finance override kept
    assert lex["interest"] == 0.0    # neutralised finance noun


def test_labels():
    assert sentiment_label(0.5) == "Strongly bullish"
    assert sentiment_label(0.2) == "Bullish"
    assert sentiment_label(0.0) == "Neutral"
    assert sentiment_label(-0.2) == "Bearish"
    assert sentiment_label(-0.6) == "Strongly bearish"


def test_aggregate_weights_recency_relevance_and_dedupes():
    cfg = Config()
    news = [
        item("SPY", "Markets surge on strong growth and upgrades", hours_ago=1),
        item("^GSPC", "Markets surge on strong growth and upgrades", hours_ago=1),  # duplicate title
        item("SPY", "Markets crash in worst selloff since the recession", hours_ago=24 * 30),  # too old
        item("NVDA", "Nvidia downgraded on weak demand warning", hours_ago=2),
    ]
    res = analyze_news(news, cfg, holdings_weights={"NVDA": 0.08}, now=NOW,
                       holdings_names={"NVDA": "NVIDIA Corp"})
    assert res.n_articles == 4
    assert res.n_used == 2  # duplicate merged, stale article dropped
    assert res.by_ticker["SPY"] > 0 > res.by_ticker["NVDA"]
    assert res.by_ticker["^GSPC"] > 0     # the duplicated story still counts for its second feed
    assert res.score > 0                  # SPY carries weight 1.0, NVDA 0.48
    assert res.drift_adjustment == cfg.sentiment_drift_scale * res.score
    assert res.top_positive and res.top_negative


def test_stale_copy_does_not_shadow_fresh_copy():
    cfg = Config()
    title = "Stocks plunge as recession fears grow"
    res = analyze_news([item("SPY", title, hours_ago=24 * 30), item("^GSPC", title, hours_ago=1)], cfg, now=NOW)
    assert res.n_used == 1 and res.score < 0


def test_offtopic_stories_get_low_weight_and_share_classes_merge():
    cfg = Config()
    names = {"GOOGL": "Alphabet Inc Class A", "GOOG": "Alphabet Inc Class C"}
    news = [
        item("SPY", "Which solar stock is a better buy in 2026", hours_ago=1),       # single-stock, market feed
        item("GOOG", "Quantum start-up soars after funding round", hours_ago=1),      # off-topic for Alphabet
        item("GOOGL", "Alphabet shares jump on cloud growth", hours_ago=1),
    ]
    res = analyze_news(news, cfg, holdings_weights={"GOOGL": 0.03, "GOOG": 0.024}, now=NOW, holdings_names=names)
    assert "GOOG" not in res.by_ticker and "GOOGL" in res.by_ticker
    weights = {x.item.title: x.weight for x in res.top_positive + res.top_negative}
    assert weights["Quantum start-up soars after funding round"] < weights["Alphabet shares jump on cloud growth"]


def test_company_keyword_handles_punctuated_names_and_market_stories_on_company_feeds():
    cfg = Config()
    names = {"AMZN": "Amazon.com Inc", "MU": "Micron Technology Inc"}
    news = [   # hours_ago=0 so the recency factor is exactly 1
        item("AMZN", "Amazon slips as AWS outage hits retailers", hours_ago=0),
        item("MU", "Musk's new venture soars in its debut", hours_ago=0),      # "MU" must not match "Musk"
        item("MU", "Stocks fall as Treasury yields climb", hours_ago=0),        # market-wide on a company feed
    ]
    res = analyze_news(news, cfg, holdings_weights={"AMZN": 0.038, "MU": 0.016}, now=NOW, holdings_names=names)
    w = {x.item.title: x.weight for x in res.top_positive + res.top_negative}
    assert abs(w["Amazon slips as AWS outage hits retailers"] - min(1.0, 6 * 0.038)) < 1e-9
    assert abs(w["Musk's new venture soars in its debut"] - cfg.offtopic_weight * min(1.0, 6 * 0.016)) < 1e-9
    assert w["Stocks fall as Treasury yields climb"] == 1.0


def test_boilerplate_summary_is_ignored():
    cfg = Config()
    bad_summary = "The Q2 investor letter can be downloaded here. Great returns for all shareholders."
    plain = item("SPY", "Stocks fall on rate worries")
    with_boiler = item("SPY", "Stocks fall on rate worries", summary=bad_summary)
    assert score_article(AN, with_boiler, cfg.summary_weight) == score_article(AN, plain, cfg.summary_weight)


def test_no_news_is_neutral():
    res = analyze_news([], Config(), now=NOW)
    assert res.score == 0.0 and res.label == "Neutral" and res.drift_adjustment == 0.0


def test_naive_now_is_treated_as_utc():
    res = analyze_news([item("SPY", "Stocks fall as yields jump")], Config(), now=NOW.replace(tzinfo=None))
    assert res.n_used == 1 and res.score < 0
