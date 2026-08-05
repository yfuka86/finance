"""kabuステーションの PUSH配信（WebSocket）で、登録銘柄の板を常時メモリに保持する.

なぜ必要か（2026-08-03 実測で確定）:
  RESTの `GET /board` は **未登録銘柄で中央値~900ms**（stationが取引所へ取りに行く）。
  686銘柄を1周すると十数分かかり、最初と最後で気配が十数分ずれる＝**スメア**。
  「気配が寄値を予測しない」という実測はこのスメアと交絡していて分離できていない。

  一方 **登録済み銘柄の応答は中央値1.4ms**（700倍）。さらに PUSH を使えば
  ポーリング自体が不要になり、参照系10req/sの制限も効かない。
  **登録上限50銘柄までなら、全銘柄の同時スナップショットが取れる**。

制約:
  * 登録銘柄は **最大50**（PUSH/RESTで共有・株式と先物も共有）
  * PUSH は**更新があったときだけ**飛ぶ（板が動かない銘柄・場が閉じている間は無音）
  * 400ms程度の間引きがある

使い方:
    with PushBoardFeed(client, symbols) as feed:
        feed.wait_ready(timeout=30)      # 全銘柄の初値が入るまで待つ（任意）
        snap = feed.snapshot()           # {symbol: {"board":…, "at":…, "age_s":…}}
"""
from __future__ import annotations

import json
import threading
import time
from typing import Iterable

import websocket

from .kabu_client import to_kabu_symbol

MAX_REGISTERED = 50


class PushBoardFeed:
    """Register ≤50 symbols and keep their latest board in memory via WebSocket."""

    def __init__(self, client, symbols: Iterable[str], exchange: int = 1,
                 seed_via_rest: bool = True, log=print):
        syms = [str(s) for s in symbols]
        if len(syms) > MAX_REGISTERED:
            raise ValueError(f"登録銘柄は最大{MAX_REGISTERED}件（{len(syms)}件が指定された）")
        self._client = client
        self._symbols = syms
        self._kabu = {to_kabu_symbol(s): s for s in syms}   # 4桁 -> 入力表記
        self._exchange = exchange
        self._seed_via_rest = seed_via_rest
        self._log = log
        self._latest: dict[str, dict] = {}
        self._at: dict[str, float] = {}
        # 値の出どころ（"seed"=起動時のREST / "push"=配信で更新済み）と更新回数。
        # **分析ではPUSHで更新された銘柄だけを使う**ため必ず記録する。
        self._source: dict[str, str] = {}
        self._updates: dict[str, int] = {}
        # 受信履歴（record_history=True のとき）。板寄せの成立が配信でいつ観測されるか
        # ＝下り遅延の実測に使う。1メッセージ= (受信epoch, 銘柄, 現値時刻, 現値, 寄値/引値)
        self.record_history = False
        self.history: list[tuple] = []
        self._history_cap = 200_000
        self._lock = threading.Lock()
        self._ws: websocket.WebSocketApp | None = None
        self._thread: threading.Thread | None = None
        self._connected = threading.Event()
        self.messages = 0

    # ── lifecycle ───────────────────────────────────────────────────
    def __enter__(self) -> "PushBoardFeed":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def start(self) -> None:
        self._register()
        url = self._client.base.replace("http://", "ws://") + "/websocket"
        self._ws = websocket.WebSocketApp(
            url, on_message=self._on_message, on_open=self._on_open,
            on_error=lambda _ws, e: self._log(f"  push: WebSocketエラー {e}"))
        self._thread = threading.Thread(target=self._ws.run_forever, daemon=True)
        self._thread.start()
        self._connected.wait(timeout=10)
        if self._seed_via_rest:
            # PUSHは「更新があったときだけ」飛ぶ。初期値はRESTで1回だけ埋める
            # （登録済みなので1銘柄~1.4ms）。
            #
            # ★client.board() を使ってはいけない: あれは全ユニバース走査用に
            #   **45件ごとに unregister_all() を呼ぶ**ので、シード中に自分の登録を
            #   消してPUSHが止まる（2026-08-03 実障害: 913件で配信断・age が
            #   41秒→386秒に伸びた）。登録管理を通さない生の取得を使う。
            seed = getattr(self._client, "_board_or_none", None)
            for s in self._symbols:
                try:
                    b = seed(s) if seed else self._client._request(
                        "GET", f"/board/{to_kabu_symbol(s)}@{self._exchange}")
                    if b:
                        self._store(b, fallback_symbol=s, source="seed")
                except Exception:  # noqa: BLE001
                    pass

    def stop(self) -> None:
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:  # noqa: BLE001
                pass
        if self._thread is not None:
            self._thread.join(timeout=5)

    # ── internals ───────────────────────────────────────────────────
    def _register(self) -> None:
        # **先に全消去する**。GET /board は照会した銘柄を自動登録するので、
        # 直前に全ユニバースを1周していると枠(50)が他銘柄で埋まっており、
        # ここで追加登録しても上限超過で黙って弾かれる（実障害・2026-08-03）。
        try:
            self._client.unregister_all()
        except Exception:  # noqa: BLE001
            self._log("  push: unregister/all に失敗（登録枠が埋まっている可能性）")
        body = {"Symbols": [{"Symbol": to_kabu_symbol(s), "Exchange": self._exchange}
                            for s in self._symbols]}
        res = self._client._request("PUT", "/register", json=body)
        n = len(res.get("RegistList", []) or [])
        self._log(f"  push: {len(self._symbols)}銘柄を登録（station側の登録数={n}）")
        if n < len(self._symbols):
            self._log(f"  push: ⚠️登録できたのは{n}件だけ（上限50・他の処理と枠を共有）")

    def _on_open(self, _ws) -> None:
        self._connected.set()
        self._log("  push: WebSocket接続")

    def _on_message(self, _ws, message: str) -> None:
        try:
            board = json.loads(message)
        except Exception:  # noqa: BLE001
            return
        self.messages += 1
        if self.record_history and len(self.history) < self._history_cap:
            # bid/ask も残す: 寄前の気配は bid/ask にしか無い（CalcPrice/CurrentPriceは
            # 前日終値のまま）。無いと履歴から「時刻tの気配」を再構成できない
            self.history.append((time.time(), str(board.get("Symbol") or ""),
                                 board.get("CurrentPriceTime"), board.get("CurrentPrice"),
                                 board.get("OpeningPrice"), board.get("ClosingPrice"),
                                 board.get("BidPrice"), board.get("AskPrice")))
        self._store(board)

    def _store(self, board: dict, fallback_symbol: str | None = None,
               source: str = "push") -> None:
        sym = str(board.get("Symbol") or "")
        key = self._kabu.get(sym, fallback_symbol)
        if key is None:
            return
        with self._lock:
            self._latest[key] = board
            self._at[key] = time.time()
            self._source[key] = source
            if source == "push":
                self._updates[key] = self._updates.get(key, 0) + 1

    # ── public ──────────────────────────────────────────────────────
    @property
    def connected(self) -> bool:
        return self._connected.is_set()

    def wait_ready(self, timeout: float = 30.0) -> bool:
        """全銘柄に値が入るまで待つ（入らなくても False を返すだけ）。"""
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if len(self._latest) >= len(self._symbols):
                    return True
            time.sleep(0.2)
        return False

    def snapshot(self) -> dict:
        """{symbol: {"board":…, "at": epoch, "age_s": 取得からの経過秒}} を返す。

        age_s の**最大値が「スナップショットのスメア」**そのもの。RESTの1周では
        これが数百秒になるが、PUSHなら数秒以内に収まる（＝同時性が担保される）。
        """
        now = time.time()
        with self._lock:
            return {k: {"board": v, "at": self._at[k], "age_s": now - self._at[k],
                        "source": self._source.get(k, "seed"),
                        "updates": self._updates.get(k, 0)}
                    for k, v in self._latest.items()}

    def smear_seconds(self, push_only: bool = False) -> float:
        """最終更新時刻のばらつき（最古 − 最新）。

        ⚠️**これは測定誤差ではない**（2026-08-04 に解釈を訂正）。PUSHは値が動いた
        瞬間に届くので、「最終更新が10分前」の銘柄は情報が古いのではなく
        **10分間その気配が動いていない**だけ。スナップショットは時刻Tにおける
        全銘柄の正しい現在値であり、同時性は担保されている。
        RESTの1周（07:50に読んだ銘柄が08:22までに動いても分からない）とは別物。

        この値は「気配がどれくらい静かか」の目安として見る。真の情報鮮度は
        health() で担保する（接続・全銘柄が配信を受けているか・欠落が無いか）。
        push_only=True なら PUSHで更新された銘柄だけで測る。
        """
        with self._lock:
            ats = [t for k, t in self._at.items()
                   if not push_only or self._updates.get(k, 0) > 0]
            return (max(ats) - min(ats)) if len(ats) >= 2 else 0.0

    def push_updated(self) -> int:
        """PUSHで1回以上更新された銘柄数。"""
        with self._lock:
            return sum(1 for v in self._updates.values() if v > 0)

    def health(self) -> dict:
        """スナップショットが信用できるかの健全性指標。

        PUSHの同時性が崩れるのは次の3つだけなので、それを直接見る:
          * WebSocketが切れている
          * 一度もPUSHが来ていない銘柄がある（シード値のまま＝動いても気付けない）
          * 配信が止まっている（messages が増えない）
        """
        with self._lock:
            never = [k for k in self._symbols if self._updates.get(k, 0) == 0]
        return {"connected": self.connected, "messages": self.messages,
                "symbols": len(self._symbols), "never_pushed": never,
                "ok": self.connected and not never}
