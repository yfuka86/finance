# finance — 日本株 場中フラット自動トレード

日本株の**場中フラット戦略**（寄付きで建て・引けで手仕舞い・オーバーナイト保有なし）の
リサーチ〜ライブ執行までの一式。個別株のみ・ドルニュートラルL/S・auカブコム(kabuステーションAPI)執行。

> エージェント向けの作業規約・確定知見は **[AGENTS.md](AGENTS.md)**（正本）。
> ライブ運用の詳細は **[trading/jp_intraday/live/README.md](trading/jp_intraday/live/README.md)**。

## 現在の到達点（2026-07時点・すべて敵対的検証済み）

| 指標 | 値 |
|---|---|
| 本命戦略 | `ensemble_core`（ML予測強度ウェイト × セクターボラ集中版、資本50/50） |
| リサーチOOS (2024-08+) | ネットSharpe **8.77 @3bps / 7.14 @7bps** |
| 本番換算（¥20M・単元・信用2倍・7bps） | 全期間 **年率53.6% / Sh 2.20 / DD−21%**、OOS24+ 101%/Sh3.21 |
| 資金キャパシティ | **〜¥100M**（制約は価格規制の条件付き50単元キャップ→解消済み。次は参加率） |
| 実行 | 寄成(13)建て → 引成(16)返済のみ（日中バリア等は約定現実性検証で棄却） |
| ML | ridge・年次ウォークフォワード（過去年学習→翌年OOS、リーク防止テストで保証） |

## リポジトリ構成

```
trading/jp_intraday/        本体（リサーチ+ライブ）
  daily_gap.py                日次データ統合ローダ（重複排除）・ギャップ検証
  daily_model.py              特徴量パネル・ridgeウォークフォワード・load_panel_cached(高速キャッシュ)
  strategies.py               戦略レジストリ(20+)・構築方式・¥単元/信用忠実バックテスト
  exec_overlay.py             執行感応度分析（リサーチ専用・本番使用禁止の警告参照）
  dashboard.py                Streamlit管理画面（下記）
  live/                       kabuステーションAPI ライブ執行（README必読）
  webapp/                     Web報告先 https://trade.a-tokyo.jp（Cloud Run）
scripts/                    データ収集（冪等）・実験ランナー・GCS同期
tests/                      33テスト（リーク防止・単元計算・ライブ安全機構）
data/                       ほぼ.gitignore（下記「データの入手」参照）
```

## セットアップ

```bash
pip install -r requirements.txt
# .env（Git管理外）: JQUANTS_API_KEY, EDINET_API_KEY,（ライブ用）KABU_*, REPORT_*
PYTHONPATH=. python -m pytest tests/ -q          # 33 passed を確認
```

### データの入手（2通り）

**A. GCSから一括DL（Windows検証用・推奨）** — 大容量データはGitに入れていない。
```powershell
# 署名付きURLはMac側で `bash scripts/upload_data_gcs.sh` を実行すると発行される（7日有効）
powershell -ExecutionPolicy Bypass -File scripts\download_data_win.ps1 -Url "<署名URL>"
```
中身: 日次調整済みバー2018-2026 / 先物・指数・空売り比率 / TOPIX参照・株数 / 学習済みMLモデル。
ティック集計・5分足2年分は別オブジェクト（`gs://atokyo-trade-data/` 配下、同スクリプトの要領でURL発行）。

**B. J-Quantsから収集（Premiumプラン・単独実行必須）**
```bash
PYTHONPATH=. python scripts/collect_jp_daily_history.py   # 日次（冪等・不足日のみ）
PYTHONPATH=. python scripts/collect_jp_derivatives.py     # 先物/指数/空売り
PYTHONPATH=. python scripts/collect_jp_minutes_2y.py      # 分足2年（任意・5GB）
```

## 管理画面（ローカル）

```bash
PYTHONPATH=. streamlit run trading/jp_intraday/dashboard.py   # → http://localhost:8501
```
- **一覧(index)**: 全戦略の成績リスト＋概要 → 「詳細▶」で個別画面(show)
- **モード**: 💰単元取引（予算¥1000万単位・信用倍率・保証金=ストップ高×30%忠実）/ 理想バックテスト
- **ユニバース制約**: 流動性・市場区分（プライムのみ等）・時価総額バンド
- **日次トレード明細**: 任意の日の銘柄・単元数・建玉¥・損益¥

## ライブ実行（要点だけ・詳細は live/README.md）

```bash
python -m trading.jp_intraday.live.run_live preflight  # どのOSでも: モックで全フロー検証
python -m trading.jp_intraday.live.run_live train      # 年1回: MLモデル更新
# Windows+kabuステーション: plan(08:55) → entry(08:59寄成) → exit(14:55引成) → state
```
- 安全設計: **既定は発注しない**（mock）。実発注は `KABU_ENV=prod × KABU_DRY_RUN=0 × KABU_LIVE_CONFIRMED=1` の3重ロック
- 売建可否はAPI実チェック＋繰り上げ補充。価格規制はトリガー銘柄のみ50単元キャップ
- 結果は https://trade.a-tokyo.jp に自動報告（Bearer認証）

## リサーチの規律（must）

1. **PIT厳守**: 当日終値・当日出来高は寄付き時点で未知。特徴量は必ずラグ
2. **IS(前60%)で選択 → OOS(後40%)で確認 → 敵対的再計算**（独立再実装で±10%再現）
3. **約定モデルの保守性チェック**（ブリーチ足終値約定のような実装不能な好条件を信じない）
4. 高速化規約: パネルは `load_panel_cached()`（warm 0.11s）、`transform(lambda rolling)` 禁止

確定知見・棄却済み手法（ボラ目標・決算除外・GBM・バリア執行 等）の全リストは **AGENTS.md** 参照。
