# 場中フラット・ライブトレード（auカブコム kabuステーションAPI）

日本株の**場中フラット（寄付き建て・引け手仕舞い・オーバーナイトなし）**L/S を
auカブコム証券の **kabuステーションAPI** で執行し、結果を **a-tokyo.jp** の Web 管理画面へ送る構成。

```
[Windows PC] kabuステーション(常駐/ログイン)
   └ localhost:18081(検証)/18080(本番) REST/WS
   └ trading/jp_intraday/live/  … クライアント（立案→寄成→引成→報告）
        └─ HTTPS POST 結果 ─▶ [GCP Cloud Run] trade.a-tokyo.jp  … Web管理画面
```

## いま Windows でなくても検証できる（重要）
`preflight` は **kabuステーション/Windows 不要**。ディスク上の履歴データから MockKabuClient で
`plan→entry→exit→report` を丸ごと実行し、ロジック・銘柄コード変換・板取得・寄成/引成・返済・
建玉フラットを検証します（どのOSでも可）。
```
python -m trading.jp_intraday.live.run_live preflight
# → PREFLIGHT: entered=N exit_orders=N positions_after=0 (should be 0) を確認
pytest tests/test_jp_intraday_live.py -q     # オフライン単体テスト
```

## 3環境と安全設計（既定は発注しない）
- `KABU_ENV=mock` … kabu不要。履歴データでモック実行（既定・開発/検証用）。
- `KABU_ENV=test` … kabuステーション検証環境(18081)。`KABU_DRY_RUN=0` で**ペーパー発注**（無害）。
- `KABU_ENV=prod` … 本番(18080)。**実発注は3ロック全一致時のみ**：`env=prod` かつ `KABU_DRY_RUN=0`
  かつ `KABU_LIVE_CONFIRMED=1`。
手順：`preflight`(どこでも) → `test`でペーパー発注 → `prod`少額 → 段階増資。

### 空売り可否・価格規制の取り扱い（重要）
- **単元=100株**。空売り価格規制の適用除外は「1注文50単元(=5,000株)以下」。当戦略のショートは
  上ギャップ銘柄でトリガー(基準値比-10%)該当は全行の~2.2%のみ → **キャップはトリガー銘柄に条件適用**
  （非トリガー銘柄はそもそも規制対象外・サイズ制限なし）。この変更でキャパ天井が~¥100Mに拡大。
- **売建可否の実チェック**: プラン生成時にショート候補を kabu API `/symbol` の
  MarginSell/KCMarginSell で確認し、不可銘柄は除外して次点候補を繰り上げ（最大3周・キャッシュ付き）。
  デイトレ信用の在庫切れは日々変動するため、発注時エラーも enter が捕捉して記録する。

### レビューで修正済みの重要点
- **銘柄コード変換**：内部はJ-Quants 5桁、kabuへは自動で4桁に変換（板・発注の整合）。
- **entry冪等**：既存建玉/発注中はスキップ＋当日マーカーで二重発注防止（`--force`で明示解除）。
- **exit過剰返済防止**：返済数量＝LeavesQty−HoldQty。再実行しても積み増さない。
- **発注レスポンス検証**：HTTP200でも `Result≠0`（建余力/空売り規制等）は失敗として捕捉。
- **データ鮮度チェック**：prodで日次データが古い場合は中断（古いprev_closeで発注しない）。
- **ドルニュートラル維持**：グロス上限超過時は両サイド比例縮小（L/S均衡を崩さない）。

## Windows セットアップ
1. auカブコム証券の口座開設・**信用取引（できれば一日信用）を有効化**。
2. **kabuステーション**（Windowsアプリ）をインストールしログイン。設定→APIを有効化し、
   **APIパスワード**を設定。検証環境も同アプリから起動可。
3. Python 3.11 を入れ、リポジトリを配置：
   ```
   py -m venv .venv & .venv\Scripts\activate
   pip install -r requirements.txt
   ```
4. リポジトリ直下 `.env` に設定（値はコミットしない）：
   ```
   KABU_ENV=mock              # mock(履歴)/test(検証18081)/prod(本番18080)
   KABU_API_PASSWORD=＜APIパスワード＞
   KABU_ORDER_PASSWORD=＜注文発注パスワード＞
   KABU_DRY_RUN=1
   KABU_LIVE_CONFIRMED=0
   LIVE_STRATEGY=ensemble_core   # 本命（ML+svdn集中の50/50）
   LIVE_CAPITAL_YEN=20000000
   LIVE_NAMES_PER_SIDE=10
   LIVE_MARGIN_TYPE=3          # 一日信用(手数料0/金利~0)
   LIVE_COST_BPS_SIDE=7        # 一日信用の実勢スリッページ目安
   REPORT_URL=https://trade.a-tokyo.jp/api/report
   REPORT_TOKEN=＜共有トークン＞
   ```
5. 毎営業日、寄付き前に日次データを更新（前日終値・ユニバース）：
   `python -m scripts.collect_jp_daily_history`（既存はスキップ／不足のみ）。

## 実行（JST）
```
python -m trading.jp_intraday.live.run_live preflight # どこでも: モックで全フロー検証
python -m trading.jp_intraday.live.run_live plan      # 08:55 立案（発注しない・確認）
python -m trading.jp_intraday.live.run_live entry     # 08:59 寄成 新規建て（冪等）
python -m trading.jp_intraday.live.run_live exit      # 14:55 引成 返済（全フラット・再実行安全）
python -m trading.jp_intraday.live.run_live state     # 建玉/資産を管理画面へ送信
```
タスクスケジューラ（Windows）で上記を平日にスケジュールすれば全自動。半自動運用なら
`plan`で内容を確認 → 手動で`entry`。

## Web管理画面（デプロイ済み: https://trade.a-tokyo.jp）
- **GCPプロジェクト**: `atokyo-trade`（**yuta@a-tokyo.jp 個人アカウント**。公私分離のため
  aim-research 組織には一切置かない）。リージョン asia-northeast1、Firestore永続化。
- DNS: Cloudflare `trade CNAME ghs.googlehosted.com`（DNS only）。
- 認証: `/api/report` は Bearer REPORT_TOKEN（.env と Cloud Run 環境変数の両方に設定済み）。
- 再デプロイ:
```
gcloud run deploy atokyo-trade --source trading/jp_intraday/webapp \
  --project atokyo-trade --region asia-northeast1 --allow-unauthenticated \
  --account=yuta@a-tokyo.jp
```
※ ヘルスチェックは `/api/health`（`/healthz` はCloud Run予約パスでアプリに届かない）。

## 本番移行チェックリスト（実弾前に必須）
- [ ] 検証環境(18081)で `plan`→`entry`→`exit` が期待通り（板取得・寄成/引成・返済HoldID）
- [ ] 一日信用の建余力・貸借銘柄・空売り規制（50単元/価格規制）をハンドリング
- [ ] 寄成の想定始値 vs 実約定のズレ（板の予想始値精度）を検証
- [ ] `LIVE_MAX_GROSS_YEN` 安全弁・1銘柄上限・全体上限
- [ ] 障害時の未決済ポジション手仕舞い手順（`exit`再実行・手動）
- [ ] 少額（例 ¥100万・数銘柄）で本番を数日 → 段階増資

## 未検証・要注意（この環境では実接続テスト不可）
- kabuステーションは Windows 常駐が必須のため、Linux/CIでは動作確認不可。仕様準拠で実装済みだが
  **必ず検証環境で実接続テスト**してから本番へ。
- 板の「予想始値」フィールド名は環境差があり得るため、`executor._est_open` を実データで要確認。
- ML戦略は年次学習済みモデル（data/live_models/ridge_YYYY.json）で稼働。年初に
  `python -m trading.jp_intraday.live.run_live train` を1回実行して更新する（年次で十分・R4検証済み）。
