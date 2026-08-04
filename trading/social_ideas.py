"""PIT-safe X post collector and deterministic trading-idea triage.

X posts are discovery material, never executable signals.  The output is an
append-only evidence log plus review-required hypothesis cards.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from urllib.parse import quote_plus
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import requests

X_RECENT_SEARCH = "https://api.x.com/2/tweets/search/recent"
TOKEN_ENV = "X_BEARER_TOKEN"
_CASHTAG = re.compile(r"\$([A-Za-z][A-Za-z0-9._-]{0,14})")
_SPACE = re.compile(r"\s+")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_z(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def load_dotenv(path: Path) -> None:
    """Load the repository's untracked .env without logging secret values."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def _append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return len(rows)


@dataclass
class XClient:
    bearer_token: str
    session: requests.Session | None = None
    timeout: float = 30.0

    def __post_init__(self) -> None:
        if not self.bearer_token:
            raise ValueError(f"{TOKEN_ENV} is empty")
        if self.session is None:
            self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {self.bearer_token}"})

    def recent_search(self, query: str, max_pages: int = 2) -> list[dict[str, Any]]:
        """Fetch up to ``max_pages`` while respecting 429 reset headers."""
        params: dict[str, Any] = {
            "query": query,
            "max_results": 100,
            "sort_order": "recency",
            "tweet.fields": "id,text,author_id,created_at,lang,public_metrics,entities,conversation_id,referenced_tweets",
            "expansions": "author_id",
            "user.fields": "id,username,name,verified,public_metrics,description",
        }
        rows: list[dict[str, Any]] = []
        for _ in range(max_pages):
            response = self.session.get(X_RECENT_SEARCH, params=params, timeout=self.timeout)
            if response.status_code == 429:
                reset = int(response.headers.get("x-rate-limit-reset", "0") or 0)
                wait = max(1, min(60, reset - int(time.time()) + 1))
                time.sleep(wait)
                response = self.session.get(X_RECENT_SEARCH, params=params, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
            users = {x["id"]: x for x in payload.get("includes", {}).get("users", [])}
            for post in payload.get("data", []):
                rows.append({"post": post, "author": users.get(post.get("author_id"), {})})
            token = payload.get("meta", {}).get("next_token")
            if not token:
                break
            params["next_token"] = token
        return rows


def collect(config: dict[str, Any], out_dir: Path, client: XClient,
            received_at: datetime | None = None) -> dict[str, Any]:
    received_at = received_at or utc_now()
    raw_path = out_dir / "posts.jsonl"
    known = {str(x["post_id"]) for x in iter_jsonl(raw_path)}
    fresh: list[dict[str, Any]] = []
    per_query: dict[str, int] = {}
    for item in config["queries"]:
        query = item["query"]
        if len(query) > 512:
            raise ValueError(f"X recent-search query exceeds 512 chars: {item['id']}")
        found = client.recent_search(query, int(config.get("max_pages_per_query", 2)))
        count = 0
        for hit in found:
            post, author = hit["post"], hit["author"]
            post_id = str(post["id"])
            if post_id in known:
                continue
            metrics = post.get("public_metrics", {})
            author_metrics = author.get("public_metrics", {})
            fresh.append({
                "post_id": post_id,
                "query_ids": [item["id"]],
                "received_at": iso_z(received_at),
                "created_at": post.get("created_at"),
                "author_id": str(post.get("author_id", "")),
                "username": author.get("username", ""),
                "author_verified": bool(author.get("verified", False)),
                "author_followers": int(author_metrics.get("followers_count", 0) or 0),
                "text": post.get("text", ""),
                "lang": post.get("lang"),
                "public_metrics": metrics,
                "conversation_id": post.get("conversation_id"),
                "source_url": f"https://x.com/{author.get('username', 'i')}/status/{post_id}",
            })
            known.add(post_id)
            count += 1
        per_query[item["id"]] = count
    fresh.sort(key=lambda x: (x.get("created_at") or "", x["post_id"]))
    added = _append_jsonl(raw_path, fresh)
    manifest = {
        "received_at": iso_z(received_at), "new_posts": added,
        "per_query": per_query, "raw_path": str(raw_path),
        "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
    }
    _append_jsonl(out_dir / "runs.jsonl", [manifest])
    return manifest


def _load_browser_cookies(path: Path) -> list[dict[str, Any]]:
    """Return only the two session cookies required by x.com; never log values."""
    data = load_json(path)
    cookies = []
    for name in ("auth_token", "ct0"):
        value = str(data.get(name, "")).strip()
        if not value:
            raise ValueError(f"cookie file has no {name}")
        cookies.append({"name": name, "value": value, "domain": ".x.com", "path": "/",
                        "secure": True, "httpOnly": name == "auth_token", "sameSite": "Lax"})
    return cookies


def collect_browser(config: dict[str, Any], out_dir: Path, cookie_file: Path,
                    received_at: datetime | None = None, headless: bool = True) -> dict[str, Any]:
    """Collect rendered public search results using the user's logged-in X session."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:  # pragma: no cover - depends on local optional package
        raise RuntimeError("playwright is required for browser collection") from exc

    received_at = received_at or utc_now()
    raw_path = out_dir / "posts.jsonl"
    known = {str(x["post_id"]) for x in iter_jsonl(raw_path)}
    fresh_by_id: dict[str, dict[str, Any]] = {}
    per_query: dict[str, int] = {}
    cookies = _load_browser_cookies(cookie_file)
    scrolls = int(config.get("browser_scrolls", 8))

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=headless)
        context = browser.new_context(locale="ja-JP", timezone_id="Asia/Tokyo")
        context.add_cookies(cookies)
        page = context.new_page()
        for item in config["queries"]:
            url = f"https://x.com/search?q={quote_plus(item['query'])}&src=typed_query&f=live"
            page.goto(url, wait_until="domcontentloaded", timeout=60_000)
            try:
                page.wait_for_selector('article[data-testid="tweet"]', timeout=20_000)
            except Exception:
                # Authentication failures surface without exposing cookie values.
                if "login" in page.url:
                    raise RuntimeError("X cookie session expired or login was rejected")
            for _ in range(scrolls):
                page.mouse.wheel(0, 1400)
                page.wait_for_timeout(900)
            rendered = page.locator('article[data-testid="tweet"]').evaluate_all(r"""
              articles => articles.map(a => {
                const time = a.querySelector('time');
                const status = time && time.closest('a[href*="/status/"]');
                const match = status && status.getAttribute('href').match(/^\/([^/]+)\/status\/(\d+)/);
                const text = a.querySelector('[data-testid="tweetText"]');
                const metric = name => {
                  const el = a.querySelector(`[data-testid="${name}"]`);
                  const label = el && el.getAttribute('aria-label') || '';
                  const m = label.replace(/,/g, '').match(/(\d+)/);
                  return m ? Number(m[1]) : 0;
                };
                return match ? {post_id: match[2], username: match[1], created_at: time.dateTime,
                  text: text ? text.innerText : '', reply_count: metric('reply'),
                  retweet_count: metric('retweet'), like_count: metric('like')} : null;
              }).filter(Boolean)
            """)
            count = 0
            for post in rendered:
                post_id = str(post["post_id"])
                if post_id in known:
                    continue
                if post_id in fresh_by_id:
                    if item["id"] not in fresh_by_id[post_id]["query_ids"]:
                        fresh_by_id[post_id]["query_ids"].append(item["id"])
                    continue
                row = {
                    "post_id": post_id, "query_ids": [item["id"]],
                    "received_at": iso_z(received_at), "created_at": post.get("created_at"),
                    "author_id": post.get("username", "").casefold(),
                    "username": post.get("username", ""), "author_verified": False,
                    "author_followers": 0, "text": post.get("text", ""), "lang": None,
                    "public_metrics": {"reply_count": post.get("reply_count", 0),
                                       "retweet_count": post.get("retweet_count", 0),
                                       "like_count": post.get("like_count", 0), "quote_count": 0},
                    "conversation_id": None,
                    "source_url": f"https://x.com/{post.get('username', 'i')}/status/{post_id}",
                }
                fresh_by_id[post_id] = row; count += 1
            per_query[item["id"]] = count
        context.close(); browser.close()

    fresh = sorted(fresh_by_id.values(), key=lambda x: (x.get("created_at") or "", x["post_id"]))
    added = _append_jsonl(raw_path, fresh)
    manifest = {
        "received_at": iso_z(received_at), "backend": "browser", "new_posts": added,
        "per_query": per_query, "raw_path": str(raw_path),
        "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
    }
    _append_jsonl(out_dir / "runs.jsonl", [manifest])
    return manifest


def _theme_matches(text: str, theme: dict[str, Any]) -> bool:
    lower = text.casefold()
    return any(word.casefold() in lower for word in theme["keywords"])


def _engagement(row: dict[str, Any]) -> float:
    m = row.get("public_metrics", {})
    weighted = (int(m.get("like_count", 0)) + 2 * int(m.get("retweet_count", 0))
                + 2 * int(m.get("quote_count", 0)) + int(m.get("reply_count", 0)))
    return math.log1p(weighted) / max(1.0, math.log1p(int(row.get("author_followers", 0)) + 10))


def analyze(config: dict[str, Any], out_dir: Path, asof: datetime | None = None) -> dict[str, Any]:
    """Rank themes and emit non-executable, review-required hypothesis cards."""
    asof = asof or utc_now()
    recent_cut = asof - timedelta(hours=int(config.get("recent_hours", 24)))
    history_cut = asof - timedelta(days=int(config.get("history_days", 7)))
    posts = []
    for row in iter_jsonl(out_dir / "posts.jsonl"):
        try:
            created = datetime.fromisoformat(row["created_at"].replace("Z", "+00:00"))
        except (KeyError, TypeError, ValueError):
            continue
        if history_cut <= created <= asof:
            row = dict(row); row["_created"] = created; posts.append(row)

    hype = [x.casefold() for x in config.get("hype_terms", [])]
    cards = []
    for theme in config["themes"]:
        matched = [p for p in posts if _theme_matches(p["text"], theme)]
        newest = [p for p in matched if p["_created"] >= recent_cut]
        older = [p for p in matched if p["_created"] < recent_cut]
        recent_authors = {p["author_id"] for p in newest}
        older_days = max(1.0, (int(config.get("history_days", 7)) * 24 - int(config.get("recent_hours", 24))) / 24)
        baseline = len(older) / older_days
        acceleration = (len(newest) + 1) / (baseline + 1)
        engagement = sum(_engagement(p) for p in newest) / max(1, len(newest))
        hype_share = sum(any(h in p["text"].casefold() for h in hype) for p in newest) / max(1, len(newest))
        score = math.log1p(len(newest)) + math.log1p(len(recent_authors)) + math.log(acceleration) + engagement - 1.5 * hype_share
        examples = sorted(newest, key=_engagement, reverse=True)[:3]
        cashtags = Counter(tag.upper() for p in newest for tag in _CASHTAG.findall(p["text"]))
        cards.append({
            "hypothesis_id": f"X-{asof:%Y%m%d}-{theme['id']}",
            "status": "REVIEW_REQUIRED", "executable": False,
            "theme_id": theme["id"], "label": theme["label"],
            "discovery_score": round(score, 4), "recent_posts": len(newest),
            "recent_unique_authors": len(recent_authors), "baseline_posts_per_day": round(baseline, 3),
            "acceleration": round(acceleration, 3), "hype_share": round(hype_share, 3),
            "top_cashtags": cashtags.most_common(10),
            "mechanism_to_verify": theme["mechanism"], "measurable_proxy": theme["proxy"],
            "falsification": theme["falsification"],
            "evidence": [{"post_id": p["post_id"], "created_at": p["created_at"],
                          "url": p["source_url"]} for p in examples],
            "research_guardrails": [
                "X投稿時刻より前の価格・開示だけでPIT特徴量を構築する",
                "候補数は制限しないが、各候補を人間が独立仮説として承認する",
                "パラメータ選択期間と未使用OOS/forward評価期間を分離する",
                "投稿人気や投稿後リターンを売買シグナルとして直接使わない",
            ],
        })
    cards.sort(key=lambda x: x["discovery_score"], reverse=True)
    report = {"asof": iso_z(asof), "posts_in_window": len(posts), "cards": cards}
    report_dir = out_dir / "reports"; report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"ideas_{asof:%Y%m%dT%H%M%SZ}.json"
    if path.exists():
        raise FileExistsError(f"append-only report exists: {path}")
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"report": str(path), "posts_in_window": len(posts), "cards": len(cards),
            "top": cards[:5]}
