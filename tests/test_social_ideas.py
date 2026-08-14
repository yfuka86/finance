import json
from datetime import datetime, timezone

from trading.social_ideas import XClient, analyze, collect


class Response:
    status_code = 200
    headers = {}
    def raise_for_status(self): pass
    def json(self):
        return {"data": [{"id": "1", "author_id": "u1", "created_at": "2026-08-04T01:00:00Z",
                          "text": "TOPIX リバランスと自社株買い", "public_metrics": {"like_count": 5}}],
                "includes": {"users": [{"id": "u1", "username": "researcher",
                                          "public_metrics": {"followers_count": 100}}]},
                "meta": {"result_count": 1}}


class Session:
    def __init__(self): self.headers = {}
    def get(self, *args, **kwargs): return Response()


def config():
    return {"max_pages_per_query": 1, "recent_hours": 24, "history_days": 7,
            "queries": [{"id": "q", "query": "TOPIX lang:ja -is:retweet"}],
            "hype_terms": ["絶対"],
            "themes": [{"id": "rebalance", "label": "rebalance", "keywords": ["TOPIX"],
                        "mechanism": "flow", "proxy": "official data", "falsification": "no OOS edge"}]}


def test_collect_is_globally_deduplicated_and_secret_is_not_saved(tmp_path):
    client = XClient("super-secret", session=Session())
    now = datetime(2026, 8, 4, 2, tzinfo=timezone.utc)
    assert collect(config(), tmp_path, client, now)["new_posts"] == 1
    assert collect(config(), tmp_path, client, now)["new_posts"] == 0
    text = (tmp_path / "posts.jsonl").read_text()
    assert "super-secret" not in text
    assert len(text.splitlines()) == 1


def test_analysis_makes_review_only_card(tmp_path):
    collect(config(), tmp_path, XClient("token", session=Session()),
            datetime(2026, 8, 4, 2, tzinfo=timezone.utc))
    result = analyze(config(), tmp_path, datetime(2026, 8, 4, 3, tzinfo=timezone.utc))
    report = json.loads((tmp_path / "reports" / result["report"].split("/")[-1]).read_text())
    card = report["cards"][0]
    assert card["status"] == "REVIEW_REQUIRED"
    assert card["executable"] is False
    assert card["recent_unique_authors"] == 1
