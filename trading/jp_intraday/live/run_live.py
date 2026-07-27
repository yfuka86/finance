"""CLI for the live intraday flat-overnight trader (auカブコム / kabuステーションAPI).

Run ON WINDOWS with kabuステーション running & logged in (env=test/prod). The
`preflight` action needs neither Windows nor kabu — it drives the whole flow from
on-disk historical data via MockKabuClient, so you can validate logic anywhere.

  python -m trading.jp_intraday.live.run_live preflight  # どこでもOK: モックで全フロー検証
  python -m trading.jp_intraday.live.run_live plan       # 08:55 立案（発注しない・確認）
  python -m trading.jp_intraday.live.run_live entry      # 08:59 寄成 新規建て
  python -m trading.jp_intraday.live.run_live exit       # 14:55 引成 返済（全フラット）
  python -m trading.jp_intraday.live.run_live state      # 建玉/資産を管理画面へ送信

Safe by default (KABU_ENV=mock, dry-run). Real orders need in .env:
  KABU_ENV=prod  KABU_DRY_RUN=0  KABU_LIVE_CONFIRMED=1   (検証は KABU_ENV=test)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json

import pandas as pd

from trading.jp_intraday.daily_gap import load_existing_daily
from trading.jp_intraday.daily_model import build_daily_features
from . import reporter
from .config import LiveConfig
from .executor import enter, exit_all, generate_plan
from .kabu_client import KabuClient, to_kabu_symbol
from .mock_client import MockKabuClient


def _now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def _build_client(cfg: LiveConfig):
    if cfg.env == "mock":
        panel = build_daily_features(load_existing_daily(), min_value_yen=cfg.min_value_yen)
        last = panel[panel["date"].eq(panel["date"].max())]
        opens = (last["raw_open"].fillna(last["open"]))
        prices = dict(zip(last["symbol"], opens))
        prev = dict(zip(last["symbol"], last["raw_close"].fillna(last["close"])))
        return MockKabuClient(prices, capital_yen=cfg.capital_yen, prev_close=prev)
    client = KabuClient(cfg.api_password, cfg.order_password, env=cfg.env)
    client.authenticate()
    return client


def _plan_symbols_today() -> set:
    """kabu symbols from today's persisted entry marker (to restrict exit)."""
    from .executor import _marker
    m = _marker("entry")
    if not m.exists():
        return set()
    data = json.loads(m.read_text())
    return {o.get("symbol") for o in data.get("orders", []) if o.get("symbol")}


def main() -> None:
    ap = argparse.ArgumentParser(description="kabuステーション 場中フラット トレーダー")
    ap.add_argument("action", choices=["train", "preflight", "plan", "entry", "exit", "state"])
    ap.add_argument("--force", action="store_true", help="entry: allow same-day re-run")
    args = ap.parse_args()

    if args.action == "train":  # 年次1回: 当年より前の全データでridgeを学習・保存
        from trading.jp_intraday.strategies import STRATEGIES
        from . import models
        panel = build_daily_features(load_existing_daily(),
                                     min_value_yen=LiveConfig.from_env().min_value_yen)
        feats = STRATEGIES["ml_mag_adaptive"]["features"]  # 本番MLと同一特徴量で学習
        path = models.train_and_save(panel, dt.date.today().year, features=feats)
        print(f"saved: {path} (features={len(feats)})")
        return

    cfg = LiveConfig.from_env()
    if args.action == "preflight":
        cfg = LiveConfig(**{**cfg.__dict__, "env": "mock"})  # offline mock (harmless in-memory)
    cfg.validate()
    print("CONFIG:", cfg.summary(), flush=True)
    client = _build_client(cfg)

    if args.action in ("preflight", "plan"):
        plan, meta = generate_plan(client, cfg)
        print("META:", meta)
        print(plan.to_string(index=False) if len(plan) else "(no positions today)")
        reporter.report(cfg, "plan", {"meta": meta, "plan": plan.to_dict("records")}, _now())
        if args.action == "preflight":  # rehearse entry+exit offline via mock
            e = enter(client, cfg, plan, force=True)
            x = exit_all(client, cfg, only_kabu_symbols={r["kabu_symbol"] for _, r in plan.iterrows()})
            print(f"\nPREFLIGHT: entered={len(e)} exit_orders={len(x)} "
                  f"positions_after={len(client.positions(product=2))} (should be 0)")

    elif args.action == "entry":
        plan, meta = generate_plan(client, cfg)
        res = enter(client, cfg, plan, force=args.force)
        print(f"entered {len(res)} (will_send={cfg.will_send_orders})")
        print(json.dumps(res, ensure_ascii=False, default=str, indent=2)[:2000])
        reporter.report(cfg, "entry", {"meta": meta, "orders": res}, _now())

    elif args.action == "exit":
        only = _plan_symbols_today() or None
        res = exit_all(client, cfg, only_kabu_symbols=only)
        print(f"exit orders: {len(res)} (will_send={cfg.will_send_orders}, restrict={bool(only)})")
        print(json.dumps(res, ensure_ascii=False, default=str, indent=2)[:2000])
        reporter.report(cfg, "exit", {"orders": res}, _now())

    elif args.action == "state":
        state = {"positions": client.positions(product=2), "margin": client.wallet_margin()}
        reporter.report(cfg, "state", state, _now())
        print(json.dumps(state, ensure_ascii=False, default=str, indent=2)[:2000])


if __name__ == "__main__":
    main()
