"""Live-trading configuration. Safe by default: 検証環境 + dry-run (no orders).

Three environments:
  mock  … no kabuステーション at all — drives the whole flow from historical data
          (for off-Windows development / preflight self-test)
  test  … kabuステーション 検証環境 (port 18081). Orders are PAPER (harmless) unless dry_run.
  prod  … kabuステーション 本番 (port 18080). REAL money — needs the triple lock.

Order gating:
  paper_orders_enabled = env=='test' and not dry_run           # rehearse safely
  orders_enabled       = env=='prod' and not dry_run and live_confirmed   # real money

Set in the repo-root .env (never commit real values):
  KABU_ENV=mock|test|prod   KABU_API_PASSWORD=...   KABU_ORDER_PASSWORD=...
  KABU_DRY_RUN=1  KABU_LIVE_CONFIRMED=0
  LIVE_STRATEGY=sector_vol_double_neutral  LIVE_CAPITAL_YEN=20000000
  LIVE_NAMES_PER_SIDE=8  LIVE_MARGIN_RATIO=2.0  LIVE_MARGIN_TYPE=3  LIVE_ACCOUNT_TYPE=4
  LIVE_MIN_VALUE_YEN=500000000  LIVE_MAX_GROSS_YEN=20000000
  LIVE_COST_BPS_SIDE=7
  REPORT_URL=https://trade.a-tokyo.jp/api/report  REPORT_TOKEN=...
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from data.collectors.config import _load_local_env

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _b(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes")


@dataclass(frozen=True)
class LiveConfig:
    env: str = "mock"
    api_password: str = ""
    order_password: str = ""
    # 板データだけ別環境から取る (検証環境は板がnullのため "prod" を指定して
    # リハーサルする。発注側は env のまま。空文字=無効)
    data_env: str = ""
    data_api_password: str = ""
    dry_run: bool = True
    live_confirmed: bool = False
    strategy: str = "ensemble_core"
    capital_yen: float = 20_000_000
    names_per_side: int = 8
    margin_ratio: float = 2.0          # 信用倍率: グロス建玉目標 = 保証金 × これ
    margin_type: int = 3
    account_type: int = 4
    min_value_yen: float = 5e8
    short_min_mktcap_yen: float = 10_000_000_000   # ショートは時価総額≥¥100億（規制常連の小型を回避）
    max_gross_yen: float = 40_000_000  # 既定 = capital × margin_ratio（from_envで自動整合）
    cost_bps_side: float = 7.0
    report_url: str = ""
    report_token: str = ""

    @classmethod
    def from_env(cls) -> "LiveConfig":
        """Read the environment freshly (so tests / re-runs see current values)."""
        _load_local_env()
        capital = float(os.environ.get("LIVE_CAPITAL_YEN", "20000000"))
        margin_ratio = float(os.environ.get("LIVE_MARGIN_RATIO", "2.0"))
        return cls(
            env=os.environ.get("KABU_ENV", "mock"),
            api_password=os.environ.get("KABU_API_PASSWORD", ""),
            order_password=os.environ.get("KABU_ORDER_PASSWORD", ""),
            data_env=os.environ.get("KABU_DATA_ENV", ""),
            data_api_password=os.environ.get("KABU_DATA_API_PASSWORD", ""),
            dry_run=_b("KABU_DRY_RUN", "1"),
            live_confirmed=_b("KABU_LIVE_CONFIRMED", "0"),
            strategy=os.environ.get("LIVE_STRATEGY", "ensemble_core"),
            capital_yen=capital,
            names_per_side=int(os.environ.get("LIVE_NAMES_PER_SIDE", "8")),
            margin_ratio=margin_ratio,
            margin_type=int(os.environ.get("LIVE_MARGIN_TYPE", "3")),
            account_type=int(os.environ.get("LIVE_ACCOUNT_TYPE", "4")),
            min_value_yen=float(os.environ.get("LIVE_MIN_VALUE_YEN", "500000000")),
            short_min_mktcap_yen=float(os.environ.get("LIVE_MIN_MKTCAP_SHORT_YEN", "10000000000")),
            max_gross_yen=float(os.environ.get("LIVE_MAX_GROSS_YEN", str(capital * margin_ratio))),
            cost_bps_side=float(os.environ.get("LIVE_COST_BPS_SIDE", "7")),
            report_url=os.environ.get("REPORT_URL", ""),
            report_token=os.environ.get("REPORT_TOKEN", ""),
        )

    @property
    def orders_enabled(self) -> bool:
        """Real money: only when all three prod locks agree."""
        return self.env == "prod" and (not self.dry_run) and self.live_confirmed

    @property
    def paper_orders_enabled(self) -> bool:
        """検証環境 paper orders — harmless, lets the flow be rehearsed end to end."""
        return self.env == "test" and (not self.dry_run)

    @property
    def will_send_orders(self) -> bool:
        # mock always "sends" to the in-memory client (harmless) so the flow is exercised.
        return self.orders_enabled or self.paper_orders_enabled or self.env == "mock"

    def validate(self) -> None:
        errs = []
        if self.env not in ("mock", "test", "prod"):
            errs.append(f"KABU_ENV must be mock/test/prod (got {self.env})")
        if self.data_env:
            if self.data_env not in ("test", "prod"):
                errs.append(f"KABU_DATA_ENV must be test/prod (got {self.data_env})")
            if self.env != "test":
                errs.append("KABU_DATA_ENV is only allowed with KABU_ENV=test "
                            "(板を別環境から取るのはリハーサル専用)")
            if not self.data_api_password:
                errs.append("KABU_DATA_API_PASSWORD is required when KABU_DATA_ENV is set")
        if self.env in ("test", "prod") and not self.api_password:
            errs.append("KABU_API_PASSWORD is required for test/prod")
        if (self.orders_enabled or self.paper_orders_enabled) and not self.order_password:
            errs.append("KABU_ORDER_PASSWORD is required when sending real/paper orders")
        if self.margin_type not in (1, 3):
            errs.append("LIVE_MARGIN_TYPE must be 1(制度) or 3(一日信用)")
        if self.names_per_side < 1:
            errs.append("LIVE_NAMES_PER_SIDE must be >= 1")
        if not (1.0 <= self.margin_ratio <= 3.3):
            errs.append("LIVE_MARGIN_RATIO must be in [1.0, 3.3] (保証金率30%)")
        if self.capital_yen <= 0 or self.max_gross_yen <= 0:
            errs.append("capital / max_gross must be > 0")
        if errs:
            raise ValueError("LiveConfig invalid:\n  - " + "\n  - ".join(errs))

    def summary(self) -> str:
        if self.orders_enabled:
            mode = "🔴 実発注(本番)"
        elif self.paper_orders_enabled:
            mode = "🟠 ペーパー発注(検証)"
        elif self.env == "mock":
            mode = "⚪ モック(履歴データ)"
        else:
            mode = "🟢 ドライラン(発注なし)"
        return (f"env={self.env} strategy={self.strategy} capital=¥{self.capital_yen/1e6:.0f}M "
                f"names/side={self.names_per_side} margin={self.margin_type} mode={mode}")
