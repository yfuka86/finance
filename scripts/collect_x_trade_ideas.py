#!/usr/bin/env python3
"""Collect X posts and turn trends into review-only research hypotheses."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trading.social_ideas import (TOKEN_ENV, XClient, analyze, collect, collect_browser,
                                  load_dotenv, load_json)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("collect", "analyze", "run"), nargs="?", default="run")
    parser.add_argument("--config", type=Path, default=ROOT / "config/x_trade_ideas.json")
    parser.add_argument("--out", type=Path, default=ROOT / "data/x_trade_ideas")
    parser.add_argument("--backend", choices=("browser", "api"), default="browser")
    parser.add_argument("--cookie-file", type=Path,
                        default=Path("/Users/yutafukazawa/work/tc/secrets/x_cookies.json"))
    parser.add_argument("--show-browser", action="store_true")
    args = parser.parse_args()
    config = load_json(args.config)
    result = {}
    if args.command in ("collect", "run"):
        if args.backend == "browser":
            if not args.cookie_file.exists():
                raise SystemExit(f"Cookieファイルがありません: {args.cookie_file}")
            result["collection"] = collect_browser(
                config, args.out, args.cookie_file, headless=not args.show_browser)
        else:
            load_dotenv(ROOT / ".env")
            token = os.environ.get(TOKEN_ENV, "")
            if not token:
                raise SystemExit(f".env に {TOKEN_ENV}=... を設定してください")
            result["collection"] = collect(config, args.out, XClient(token))
    if args.command in ("analyze", "run"):
        result["analysis"] = analyze(config, args.out)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
