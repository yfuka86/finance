# AGENTS.md — finance (自動トレード研究リポジトリ)

クロスエージェント共通の作業規約。Claude / Codex はこれを正本として読む。

## このリポジトリの目的
自動トレード戦略の研究・バックテスト。主な構成:
- `trading/jp_intraday/` … 日本株イントラデイ（リーク安全な）ロング/ショート研究。
  - `data`(バー入出力/セッション定義) `features`(特徴量) `strategy`(重み付け)
    `backtest`(約定・コスト・指標) `model`(ridge ウォークフォワード)
    `walkforward` `universe`(サバイバーシップ安全) `collector`(J-Quants 収集)
    `experiments`/`tick_experiments`(需給OFI含む戦略実験) `reference`(TOPIX/株数)
  - `daily_gap`(オーバーナイトギャップ反転の日次検証+既存データ統合ローダ)
    `daily_model`(クロスセクションridge学習+特徴量) `strategies`(戦略レジストリ+
    ¥単元バックテスト+構築方式) `futures_context`/`us_context`(先物/US夜間)
    `dashboard`(Streamlit管理画面: 戦略選択/理想・¥1000万単元/日次明細)
- **場中フラット戦略の大前提**: 寄付き建て・引け手仕舞い（オーバーナイト保有なし）、
  **個別株のみ**（master MktNm≠プライム/スタンダード/グロース=ETF/ETN/REIT/投信を除外。
  ただし現行masterに無い＝上場廃止銘柄は個別株として残しサバイバーシップ回避）。
  ショートは制度信用貸借(MrgnNm=='貸借')のみ。単元=100株の整数ロット。学習は過去年→翌年OOS。
  - 保有区分: `STRATEGIES[k]["holding"]` = intraday(既定)/overnight(引→翌寄)/cc1(引→翌引)。
    非intradayは `ret_on_fwd`/`ret_cc_fwd`（パネル列）を `score_frame` がエイリアス。
    **スコア計算はエイリアス前に行う**（当日intraday_retをシグナルに使う戦略を壊さないため）。
    ライブ執行はintradayのみ（executorが他区分をNotImplementedErrorで拒否）。
    パネル列を追加したら `daily_model._PANEL_SCHEMA_VERSION` をインクリメント（キャッシュ無効化）。
- `trading/bybit/` … Bybit 無期限先物の戦略群・バックテスト。
- `screener/` … 日本株ファンダ・スクリーナ + ダッシュボード。
- `backtest/`, `scripts/` … 米日セクターのリードラグ等の実験・最適化スクリプト。
- `data/collectors/` … J-Quants(日足/財務/分足) と Stooq(日足) のコレクタ。

## データ源の方針
- **Yahoo Finance は使用禁止**（規約）。日本株・財務は **J-Quants API**、
  海外・指数の日足は **Stooq** を使う。
- API キーは **`.env`**（Git 管理外）に置く。`data/collectors/config.py` が
  起動時に読み込む（`JQUANTS_API_KEY`, `EDINET_API_KEY`）。ソースに鍵を書かない。
- **既存データを再利用し、同じデータを重複して収集・保存しない。** 日次バーは
  `daily_gap.load_existing_daily()` が既存ファイル（`data/cache/bars_day_*`,
  `data/jp_intraday_reference/`, `data/jp_daily_history/`）を (date,symbol) で
  重複排除して読む。年ごとキャッシュを正本にし、結合ファイルの二重保存はしない。
- J-Quants は **Premium**（20年・分足/ティックadd-on・先物/指数/需給）。収集は
  **単独実行**（並列は429でデータ欠損）。コレクタは年ごとキャッシュで再開可能。

## 高速化（シミュレーション）
- パネルは `daily_model.load_panel_cached(...)` を使う（入力mtime指紋のディスクキャッシュ、
  cold 13.7s → **warm 0.11s**。ユニバース制約もキー化済み・直近6件のみ保持）。
- transform(lambda rolling) は禁止 → `groupby.rolling`（Cython）に統一。β/cov/varはrolling和で
  ベクトル化（ゴールデン比較で全列一致確認済み）。unit_lotのlambda集計もベクトル化済み。

## 戦略研究の確定知見（再検証不要のベースライン）
- 本命: `ensemble_core`（ml_mag_adaptive+svdn_concentrated 50/50, OOS 7.81@3bps/6.24@7bps）。
  執行は**寄成→引成のみ**。日中±バリアは約定現実性検証で**棄却**（ブリーチ足終値約定は
  実装不能な好条件。exec_overlay.py の警告参照）。遅延エントリーも全滅＝寄付き参加が本体。
- サイジング＞シグナル: |予測|比例ウェイト+適応絞込で+1.3〜2.3 Sh。IV加重はEWに劣後。
- kabu API制約: OCO/W指値なし・注文訂正なし(取消→再発注のみ)・PUSH銘柄上限50・
  発注5件/秒・HoldQty拘束で同一建玉に複数返済不可。
- kabu API制約・Windows本番VPSでの実測 (2026-07-29確定・すべて本番口座で検証済み):
  - **信用注文は Exchange=9(SOR) 必須**。東証直指定(1)は 100368「信用新規注文は
    抑止されております」で全拒否 (エラー表: 市場に1を指定した信用新規が該当)。
    指値・寄指(板寄せ系)とも SOR で受理→取消を確認。kabu_client は SOR 既定+東証フォールバック
  - **GET /board は照会銘柄を自動登録し上限50** (51件目から4002006)。45件ごとに
    PUT /unregister/all で回避 (無いと plan の板coverage が4%に落ちる)
  - 参照系レート制限10req/s (11件目から429)・**Session再利用必須** (無いと1req/秒)
  - kabuステーションは夜間セッション切れで**毎朝ログインが必要** → 自動ログイン設定必須
    (2026-07-29朝: 未設定で 401、entry不発の実障害)
  - タスクスケジューラ実行は stdout=cp932 → **PYTHONUTF8=1 必須** (¥/絵文字printで
    UnicodeEncodeError クラッシュの実障害)
  - 検証環境(18081)は板が全null・注文はスタブ (受理のみ・建玉生成なし)。板=本番読取+
    発注=検証の HybridKabuClient (KABU_DATA_ENV=prod) で通しリハーサルする
  - 大引けは**15:30** (2024-11-05以降、15:25からプレ・クロージング)。exit 14:55発注→
    15:30板寄せ約定・state 15:40 の根拠
  - 口座区分: 特定(4)は約諾書未確認(100203)で発注不可のため当面 LIVE_ACCOUNT_TYPE=2(一般)。
    約諾解消後に4へ戻す (一般口座の約定は確定申告で自己計算が必要)
- ML再学習は**年次で十分**（四半期/月次/3年窓は本番構成で年次に勝てず・検証済み）。
- 本番推奨構成(R6条件付きショートキャップ後): ensemble_core ¥20M・8銘柄/側・信用2倍・EW。
  7bps: 全期間 53.6%/Sh2.20/DD-21%、OOS24+ 101%/Sh3.21/DD-18%。
  **OOS起点の定義に注意**: 本番換算のOOS24+は2024-01-01（暦年境界）、リサーチOOSは
  2024-08-01（IS前60%/OOS後40%分割点）。再現は `scripts/verify_baseline.py`（4項目）。
  **キャパ天井~¥100M**（¥100Mでも51%/Sh2.00。旧一律50単元キャップが唯一の減衰要因だった。
  価格規制はトリガー銘柄(全行の2.2%)のみ対象のため条件適用に変更。参加率p95<1%は~¥90-110M）。
  R6棄却: スリーブ別銘柄数変更・ML傾斜(60/40,70/30=IS過学習)・レバ水増し(2.4x=質の低いレバ)・
  適応絞込の単元移植(8/側は既に極集中)。
- R5棄却の反対パターン: ボラターゲティング(ボラ高=稼ぎ時なので逆効果)・決算日除外(ギャップは
  アルファ源)・月曜除外(ISアーティファクト)・流動性1e9フロア(シグナル消失)。
- 実験プロトコル: IS(前60%)選択→OOS(後40%)確認→敵対的再計算。破ったクレームは全て崩れた
  （GBM・交互作用・バリア等）。約定モデルの保守性チェックを必ず含めること。

## デプロイ / 公私分離（重要）
- **個人トレード関連は aim-research 組織の GCP に一切置かない**（公私混同禁止）。
  Web管理画面は https://trade.a-tokyo.jp ＝ GCPプロジェクト `atokyo-trade`
  （yuta@a-tokyo.jp 個人）。DNS は Cloudflare（a-tokyo.jp ゾーン、cloudflared
  cert.pem.a-tokyo.backup 内のAPIトークンで操作可能）。
- Cloud Run のヘルスチェックは `/api/health`（`/healthz` はGFE予約パスで不達）。

## ブランチ / PR
- base ブランチは **main**。原則 PR 経由（直接 main に入れない）。
- 変更後は自分で検証（下記）を通してから完了とする。可逆なら自己マージ可。
- **本番稼働に関わる変更（実発注ボット等）はマージ/デプロイ前にユーザー確認。**

## 検証（必須）
```bash
PYTHONPATH=. python -m pytest tests/ -q      # 全テスト
PYTHONPATH=. python -m py_compile trading/jp_intraday/*.py
```
- `trading/jp_intraday` は**研究コード**。リファクタは原則 **数値挙動を変えない**こと。
  公開関数の出力をサンプルデータでスナップショットし、変更前後で一致を確認する
  （特性テスト）。意図的に数値を変える場合は理由を明記し個別に検証する。
- 実データのイントラデイ期間は現状 **2026-06-01〜07-24（約39営業日）と短い**。
  多特徴量モデルは過学習に注意（ホールドアウト/ウォークフォワードを厳守）。

## 実行例
```bash
# 価格ベースのイントラデイ戦略実験
PYTHONPATH=. python scripts/run_jp_strategy_experiments.py
# 需給(OFI)戦略実験
PYTHONPATH=. python scripts/run_jp_tick_experiments.py
# ウォークフォワード（ridge モデル）
PYTHONPATH=. python -m trading.jp_intraday.research <bars> <memberships> <shares>
```

## 言語
応答・コミット・PR は日本語。
