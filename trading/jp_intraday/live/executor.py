"""Daily flat-overnight execution against a KabuClientProtocol (real or mock).

Flow (JST): ~08:55 generate_plan → ~08:59 enter (寄成 新規) → ~14:55 exit_all
(引成 返済, flat) → report. Signal is point-in-time (today's indicative open vs
yesterday's close); positions are held only intraday.

Safety: orders are sent only when cfg.will_send_orders (paper in 検証, or the
prod triple-lock). enter() is idempotent (skips names already held / working, and
refuses a same-day re-run without force). exit_all() closes LeavesQty−HoldQty so
re-runs never over-close.
"""
from __future__ import annotations

import datetime as dt
import json
import time

import numpy as np
import pandas as pd

from trading.jp_intraday.daily_gap import load_existing_daily
from trading.jp_intraday.daily_model import build_daily_features
from trading.jp_intraday.strategies import STRATEGIES
from .config import PROJECT_ROOT, LiveConfig
from .kabu_client import (
    FRONT_CLOSE, FRONT_OPEN, SIDE_BUY, SIDE_SELL, KabuAPIError,
    KabuClientProtocol, to_kabu_symbol,
)

_STATE_DIR = PROJECT_ROOT / "data" / "live_reports"


def _today() -> str:
    return dt.date.today().isoformat()


# ── signal (pure) ───────────────────────────────────────────────────
def open_prices(client: KabuClientProtocol, symbols) -> dict:
    """Pre-open indicative price per symbol (CalcPrice preferred, then CurrentPrice)."""
    out = {}
    for s in symbols:
        try:
            b = client.board(s)
        except Exception:
            continue
        px = b.get("CalcPrice") or b.get("CurrentPrice") or 0
        if px and float(px) > 0:
            out[s] = float(px)
    return out


def compute_today_signals(last: pd.DataFrame, opens: dict, strategy: str) -> pd.DataFrame:
    """Today's scored frame from yesterday's features + today's indicative opens (PIT)."""
    df = last[last["symbol"].isin(opens)].copy()
    df["today_open"] = df["symbol"].map(opens)
    prev_close = df["raw_close"].fillna(df["close"])
    ok = (df["today_open"] > 0) & (prev_close > 0)
    df, prev_close = df[ok].copy(), prev_close[ok]
    df["overnight_gap"] = df["today_open"] / prev_close - 1
    df["residual_gap"] = df["overnight_gap"] - df["overnight_gap"].mean()
    df["sector_resid_gap"] = df["residual_gap"] - df.groupby("sector")["residual_gap"].transform("mean")
    df["gap_z"] = df["residual_gap"] / df["gap_vol60"].clip(lower=0.005)
    # 指数分解のライブ近似: 寄付き前は業種指数の当日Oが未形成のため、業種内平均ギャップを
    # セクター成分のプロキシとする（研究側は指数ベース。乖離は小さく方向は同一）。
    df["sector_index_gap"] = df.groupby("sector")["overnight_gap"].transform("mean")
    df["idio_gap2"] = df["overnight_gap"] - df["sector_index_gap"]
    # prev_resid_gap / prev_intraday / ivol / vol20_floor / liq_rank / shortable are
    # already yesterday's values carried in `last` — the correct "prev" inputs.
    spec = STRATEGIES[strategy]
    missing = [c for c in spec["need"] if c not in df.columns]
    if missing:
        raise ValueError(f"strategy '{strategy}' needs columns not available live: {missing}")
    df["_s"] = pd.to_numeric(pd.Series(spec["score"](df), index=df.index), errors="coerce")
    return df[np.isfinite(df["_s"])].copy()


def _assert_fresh(panel: pd.DataFrame, cfg: LiveConfig) -> str:
    last_date = pd.to_datetime(panel["date"]).max()
    stale_days = (dt.date.today() - last_date.date()).days
    if cfg.env == "prod" and stale_days > 4:
        raise RuntimeError(f"daily data is stale (latest {last_date.date()}, {stale_days}d old). "
                           "Refresh with scripts.collect_jp_daily_history before trading.")
    return str(last_date.date())


def _score_today(last: pd.DataFrame, opens: dict, strategy: str) -> pd.DataFrame:
    """Score today's cross-section for one strategy (xs rule or persisted-ML)."""
    spec = STRATEGIES[strategy]
    if spec["kind"] == "xs":
        return compute_today_signals(last, opens, strategy)
    if spec["kind"] == "ml":
        from . import models
        df = compute_today_signals(last, opens, "gap_reversal")  # builds today's gap features
        model = models.load_model(dt.date.today().year)
        df["_s"] = models.predict(model, df)
        return df[np.isfinite(df["_s"])].copy()
    raise NotImplementedError(f"live scoring unsupported for kind={spec['kind']}")


def _sleeve_rows(scored: pd.DataFrame, cfg: LiveConfig, side_cap: float,
                 magnitude: bool) -> list[dict]:
    """Select + size one sleeve's book (equal-yen or |score|-proportional, cap 3x)."""
    n = cfg.names_per_side
    longs = scored.nlargest(n, "_s")
    short_pool = scored[scored["shortable"] & (scored["prev_value"] >= 1e9)]
    shorts = short_pool.nsmallest(n, "_s")
    shorts = shorts[~shorts["symbol"].isin(set(longs["symbol"]))]
    MAX_SHORT_LOTS = 50
    rows = []
    for df, side, label in ((longs, SIDE_BUY, "LONG"), (shorts, SIDE_SELL, "SHORT")):
        if df.empty:
            continue
        if magnitude:  # 検証済みの|予測|比例（上限3×等金額）
            mag = df["_s"].abs()
            raw = (mag / mag.sum()).clip(upper=3.0 / len(df))
            budgets = side_cap * raw / raw.sum()
        else:
            budgets = pd.Series(side_cap / len(df), index=df.index)
        for (_, r), budget in zip(df.iterrows(), budgets):
            lots = int(np.floor(budget / (r["today_open"] * 100)))
            if label == "SHORT":
                # 価格規制トリガー銘柄のみ50単元キャップ（非トリガー銘柄は規制対象外）。
                # 判定: 前日中にトリガー(前日安値≤前日基準値×0.9) or 当日寄りで-10%。
                base_prev = r.get("prev_close")
                trig = bool(
                    (pd.notna(r.get("low")) and pd.notna(base_prev)
                     and r["low"] <= base_prev * 0.9)
                    or (pd.notna(r.get("close"))
                        and r["today_open"] <= r["close"] * 0.9))
                if trig:
                    lots = min(lots, MAX_SHORT_LOTS)
            if lots >= 1:
                rows.append({"symbol": r["symbol"], "kabu_symbol": to_kabu_symbol(r["symbol"]),
                             "name": r.get("name", r["symbol"]), "sector": r.get("sector", ""),
                             "side": side, "side_label": label, "lots": lots, "qty": lots * 100,
                             "est_price": float(r["today_open"]),
                             "residual_gap": float(r.get("residual_gap", 0.0)),
                             "est_yen": lots * 100 * float(r["today_open"])})
    return rows


def verify_shortable(client: KabuClientProtocol, kabu_symbols: list[str],
                     cache: dict) -> set:
    """kabu APIの銘柄情報で売建可否（制度 MarginSell / 一般 KCMarginSell）を実チェック。

    デイトレ信用の売建可能銘柄は日々変動するため、貸借フィルタ（近似）に加えて
    発注前にAPIで確認する。フラグが取得できない場合は保守的に不可扱いにせず
    「不明=可」とし、発注時エラーで最終捕捉する（enterはKabuAPIErrorを握って記録する）。
    Returns: 売建不可と確認された銘柄集合。
    """
    banned = set()
    for k in kabu_symbols:
        if k in cache:
            info = cache[k]
        else:
            try:
                info = client.symbol_info(k)
            except Exception:
                info = {}
            cache[k] = info
        ms, kc = info.get("MarginSell"), info.get("KCMarginSell")
        if ms is None and kc is None:
            continue  # 不明 → 発注時エラーで捕捉
        if not (bool(ms) or bool(kc)):
            banned.add(k)
    return banned


def generate_plan(client: KabuClientProtocol, cfg: LiveConfig) -> tuple[pd.DataFrame, dict]:
    spec = STRATEGIES[cfg.strategy]
    members = spec.get("members", [(cfg.strategy, 1.0)])
    for m, _ in members:
        if STRATEGIES[m].get("holding", "intraday") != "intraday":
            raise NotImplementedError(
                f"ライブ執行は場中フラット(intraday)のみ対応。'{m}' は "
                f"{STRATEGIES[m].get('holding')} 保有のためライブ不可")
    panel = build_daily_features(load_existing_daily(), min_value_yen=cfg.min_value_yen)
    data_date = _assert_fresh(panel, cfg)
    last = panel[panel["date"].eq(panel["date"].max())].copy()

    opens = open_prices(client, list(last["symbol"]))
    coverage = len(opens) / max(len(last), 1)
    if coverage < 0.8 and cfg.env == "prod":
        raise RuntimeError(f"board coverage {coverage:.0%} < 80% — aborting (partial cross-section)")

    def _build(last_df: pd.DataFrame) -> pd.DataFrame:
        if spec["kind"] == "ensemble":
            rows = []
            for member, w in spec["members"]:
                mspec = STRATEGIES[member]
                scored_m = _score_today(last_df, opens, member)
                magnitude = mspec.get("construction", "").startswith("magnitude")
                rows += _sleeve_rows(scored_m, cfg, cfg.capital_yen * w / 2.0, magnitude)
            p = pd.DataFrame(rows)
            # 同一銘柄が複数スリーブに出たら統合（同方向合算・逆方向は差し引き）
            if not p.empty:
                p["signed_qty"] = np.where(p["side"] == SIDE_BUY, p["qty"], -p["qty"])
                agg = p.groupby(["symbol", "kabu_symbol", "name", "sector"], as_index=False).agg(
                    signed_qty=("signed_qty", "sum"), est_price=("est_price", "first"),
                    residual_gap=("residual_gap", "first"))
                agg = agg[agg["signed_qty"] != 0]
                agg["side"] = np.where(agg["signed_qty"] > 0, SIDE_BUY, SIDE_SELL)
                agg["side_label"] = np.where(agg["signed_qty"] > 0, "LONG", "SHORT")
                agg["qty"] = agg["signed_qty"].abs().astype(int)
                agg["lots"] = agg["qty"] // 100
                agg["est_yen"] = agg["qty"] * agg["est_price"]
                p = agg.drop(columns="signed_qty")
            return p
        scored = _score_today(last_df, opens, cfg.strategy)
        magnitude = spec.get("construction", "").startswith("magnitude")
        return pd.DataFrame(_sleeve_rows(scored, cfg, cfg.capital_yen / 2.0, magnitude))

    # 売建可否の実チェック（デイトレ信用の在庫は日々変動）: 不可銘柄を除外して
    # 次点候補で再選択（最大3周）。チェック済みはキャッシュしAPI呼数を最小化。
    info_cache: dict = {}
    banned_symbols: set = set()
    shorts_banned = 0
    work = last
    for _ in range(3):
        plan = _build(work)
        if plan.empty:
            break
        shorts = plan[plan["side_label"] == "SHORT"]
        newly = verify_shortable(client, [s for s in shorts["kabu_symbol"]], info_cache)
        newly -= {to_kabu_symbol(s) for s in banned_symbols}
        if not newly:
            break
        shorts_banned += len(newly)
        banned_symbols |= {s for s in work["symbol"] if to_kabu_symbol(s) in newly}
        work = work[~work["symbol"].isin(banned_symbols)]

    # Balance-preserving gross cap: scale BOTH sides by one factor, then re-floor.
    if not plan.empty and plan["est_yen"].sum() > cfg.max_gross_yen:
        scale = cfg.max_gross_yen / plan["est_yen"].sum()
        plan["lots"] = np.floor(plan["lots"] * scale).astype(int)
        plan = plan[plan["lots"] >= 1].copy()
        plan["qty"] = plan["lots"] * 100
        plan["est_yen"] = plan["qty"] * plan["est_price"]
    meta = {"data_date": data_date, "coverage": round(coverage, 3),
            "n_long": int((plan["side_label"] == "LONG").sum()) if len(plan) else 0,
            "n_short": int((plan["side_label"] == "SHORT").sum()) if len(plan) else 0,
            "gross_yen": float(plan["est_yen"].sum()) if len(plan) else 0.0,
            "shorts_banned": shorts_banned}
    return plan.reset_index(drop=True), meta


# ── execution ───────────────────────────────────────────────────────
def _marker(action: str):
    _STATE_DIR.mkdir(parents=True, exist_ok=True)
    return _STATE_DIR / f"{action}_{_today()}.json"


def enter(client: KabuClientProtocol, cfg: LiveConfig, plan: pd.DataFrame, force: bool = False) -> list[dict]:
    """Place 寄成 信用新規 orders. Idempotent: skips held/working names and same-day re-runs."""
    marker = _marker("entry")
    if marker.exists() and not force:
        raise RuntimeError(f"entry already ran today ({marker.name}). Re-run with force=True only if sure.")
    held = {to_kabu_symbol(p.get("Symbol")) for p in client.positions(product=2)}
    held |= {to_kabu_symbol(o.get("Symbol")) for o in client.orders(product=2)}
    results = []
    for _, r in plan.iterrows():
        ksym = r["kabu_symbol"]
        if ksym in held:
            results.append({"action": "OPEN", "symbol": ksym, "skipped": "already held/working"})
            continue
        intent = {"action": "OPEN", "symbol": ksym, "side": r["side"], "qty": int(r["qty"]),
                  "front": FRONT_OPEN, "est_price": r["est_price"]}
        if cfg.will_send_orders:
            try:
                intent["response"] = client.send_margin_open(
                    r["symbol"], r["side"], int(r["qty"]), front_order_type=FRONT_OPEN,
                    margin_type=cfg.margin_type, account_type=cfg.account_type)
            except KabuAPIError as exc:
                intent["error"] = str(exc)
        else:
            intent["response"] = {"dry_run": True}
        results.append(intent)
    marker.write_text(json.dumps({"time": _today(), "orders": results}, ensure_ascii=False, default=str))
    return results


def exit_all(client: KabuClientProtocol, cfg: LiveConfig, only_kabu_symbols: set | None = None,
             retries: int = 2) -> list[dict]:
    """Flatten with 引成 返済. Closes LeavesQty−HoldQty (idempotent), retries on failure."""
    results = []
    positions = client.positions(product=2)  # read-only — always fetch, even in dry-run
    for pos in positions:
        leaves = float(pos.get("LeavesQty") or 0)
        hold = float(pos.get("HoldQty") or 0)
        closeable = int(leaves - hold)
        ksym = to_kabu_symbol(pos.get("Symbol"))
        if closeable <= 0:
            continue
        if only_kabu_symbols is not None and ksym not in only_kabu_symbols:
            continue
        close_side = SIDE_SELL if str(pos.get("Side")) == SIDE_BUY else SIDE_BUY
        hold_id = pos.get("ExecutionID") or pos.get("HoldID")
        intent = {"action": "CLOSE", "symbol": ksym, "hold_id": hold_id, "qty": closeable,
                  "close_side": close_side, "front": FRONT_CLOSE}
        if not hold_id:
            intent["error"] = "missing HoldID/ExecutionID"
            results.append(intent)
            continue
        if cfg.will_send_orders:
            for attempt in range(retries + 1):
                try:
                    intent["response"] = client.send_margin_close(
                        pos.get("Symbol"), close_side, closeable, hold_id,
                        front_order_type=FRONT_CLOSE, margin_type=cfg.margin_type,
                        account_type=cfg.account_type)
                    break
                except KabuAPIError as exc:
                    intent["error"] = str(exc)
                    if attempt < retries:
                        time.sleep(1.5)
        else:
            intent["response"] = {"dry_run": True}
        results.append(intent)
    return results
