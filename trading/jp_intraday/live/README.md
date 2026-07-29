# 場中フラット・ライブトレード（auカブコム kabuステーションAPI）

日本株の**場中フラット（寄付き建て・引け手仕舞い・オーバーナイトなし）**L/S を
auカブコム証券の **kabuステーションAPI** で執行し、結果を **a-tokyo.jp** の Web 管理画面へ送る構成。

## 🎯 本番推奨構成（これを使う・2026-07確定）

**使うモデル: `LIVE_STRATEGY=ensemble_core`**（唯一の本命。変更しない）
- 中身: **ml_mag_adaptive**（ridge ML・|予測|比例ウェイト）50% ＋ **svdn_concentrated**
  （セクターボラ集中・ルール系）50% の資本分割。クラスタ間相関0.31で分散が効く。
  56組合せ全探索でIS両コスト帯首位、敵対的検証済み。
- **MLモデルファイル**: `data/live_models/ridge_2026.json`（2018〜2025年末で学習済み）。
  **2026年中はこのファイルのまま使う**。再学習は**年1回、2027年年初**に
  `python -m trading.jp_intraday.live.run_live train`（四半期/月次再学習は年次に勝てない・検証済み）。
- **サイズ: 元本¥20M・信用倍率2.0倍（グロス目標¥40M）・8銘柄/側/スリーブ・一日信用**
  → `.env`: `LIVE_CAPITAL_YEN=20000000` `LIVE_MARGIN_RATIO=2.0` `LIVE_NAMES_PER_SIDE=8`
  （執行では2スリーブ統合で実質L/S各13〜16銘柄・平均実効レバ~1.5x になる。正常）
- **執行: 寄成(FrontOrderType=13)建て → 引成(16)返済のみ**。日中の指値・バリア・
  遅延エントリーは全て検証で棄却済み。**勝手に足さない**。

### この構成の精緻シミュレーション（成績はOOSのみ表示・IS成績の使用禁止）

| 期間（すべてOOS） | 年率 | Sharpe | 日次勝率 | 最大DD |
|---|---|---|---|---|
| OOS 2024+ | +100.8% | 3.21 | 58.9% | −17.9% |
| 2025年 | +86.5% | 2.74 | 59.1% | −17.9% |
| 2026年(〜7/24) | +106.4% | 3.18 | 55.2% | −12.7% |

- OOS内は全年プラス。リスク分位（下記の停止基準・最悪日）のみ全期間2,067日の
  実分布から取っている（保守側のため許容・成績の主張ではない）。
- **日次¥損益の肌感（¥20M・信用2倍）**: 平均 +¥4.3万/日、5%タイル −¥42万、1%タイル −¥69万、
  **過去最悪日 −¥115万**（元本の−5.8%）。この規模の損失日は普通に来る前提で臨む。
- 平均拘束保証金 ¥10.7M（元本の54%・ストップ高×30%基準）→ 建余力に余裕あり。
- コスト感応度（OOS24+）: 5bps=年率118%/Sh3.8、7bps=101%/Sh3.2、**10bpsでも75%/Sh2.4**。
  損益分岐は保守側（全期間分布・リスク文脈）で~14bps。
  寄成・引成の板中心約定なら実勢2〜5bpsを想定（**最初の数週間で実測すること**）。
- リスクを抑えるなら信用1.5倍（OOS24+）: 年率77%/Sh3.29（倍率だけ下げれば良い）。

### 明日から始める手順（Day 1・検証済み 2026-07-28）

> 実取引可能性は4方向で検証済み: ①執行パイプライン（preflight/plan実出力・全38テスト）
> ②ショート実在性（直近1年のショートの98.4%が現行貸借銘柄・参加率p95 0.11%・キャップ実効1.15%）
> ③コスト（7bps=中心推定4〜9bpsの中央。10bpsでもOOS Sh2.4で経済性維持・損益分岐~14bps）
> ④戦略トーナメント（ライブ実行可能な全戦略×6/8/10銘柄をIS前60%選択→OOS後40%確認 →
> **ensemble_core/8銘柄が勝者**。単体MLはIS首位でもOOSで逆転されDD2倍のため棄却）。

1. **朝(〜08:30)**: `git pull` →
   `PYTHONPATH=. python scripts/collect_jp_daily_history.py`（前営業日分。当日分が無いのは正常）
2. `python -m trading.jp_intraday.live.run_live preflight` → `positions_after=0` と
   **data_dateが前営業日**であることを確認（preflightは本番entryをブロックしない・修正済み）
3. **08:56** `run_live entry` … 板取得(~80秒)+寄成発注（発注0.25秒間隔・自動リトライ・冪等）。
   **08:59開始だと寄付きに間に合わないリスクがあるため08:56厳守**
4. **14:55** `run_live exit` … 引成返済（15:20までに送信完了をログで確認）
5. **15:30** `run_live state` … 結果を https://trade.a-tokyo.jp へ送信

**段階導入（必須）**: Day1〜5 は `KABU_ENV=test`＋`KABU_DRY_RUN=0`（検証環境ペーパー）。
**prod移行条件（全て揃ってから）**: (a) 実効コスト5日平均≤10bps (b) 寄成/引成の全量約定を確認
(c) CalcPrice（予想始値）・HoldID返済・MarginSell/KCMarginSellフィールドの実接続動作確認。

### 日次の実効コスト実測（初日から必ず記録）

実効コスト[片道bps] = （実現PnL − 公式寄値建て・引値返済のシミュレーションPnL）÷（売買代金×2）
- 5日移動平均 **≤7bps**: 継続（想定どおり）
- **7〜10bps**: 継続するが増額凍結（10bpsでも全期間Sh1.25/OOS Sh2.4）
- **>10bpsが5日継続**: サイズ半減（¥10M）
- **>13bps**: 停止して原因分析（全期間損益分岐≈14bps）
あわせて記録: 寄成の不成立/部分約定の有無・ショートのプレミアム料/在庫切れ（銘柄別）。

### 停止・縮小基準（過去2,067日の実分布p5/p1から逆算・¥20M基準）

| トリガー | 水準 | アクション |
|---|---|---|
| 単日損失 | 〜−¥120万 | **停止しない**（p01=−¥88万・過去最悪−¥115万＝経験内） |
| 連敗 | 3連敗 | コスト実測の点検のみ（ほぼ毎月起きる正常事象） |
| 連敗 | 6連敗 | サイズ半減（年1.2回水準） |
| 連敗 | 8連敗 | 停止・点検（8年で2回未満の異常水準） |
| 週次 | −¥100万(−5%) | 注意・コスト検証 |
| 週次 | −¥160万(−8%) | サイズ半減（p01水準） |
| 週次 | −¥240万(−12%) | 停止（過去最悪−10.2%超え） |
| 累積DD | −¥300万(−15%) | サイズ半減 |
| 累積DD | −¥400万(−20%) | 停止・全面見直し（過去最大DD到達） |

**期待値の目安（直近1年・¥20M）**: 日次平均+¥5.5万・中央値+¥2.3万・典型レンジ±¥30万・
p05=−¥58万・p01=−¥88万。勝率54〜59%。負け月は約3割・最悪月−9%は経験内。

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
   LIVE_MARGIN_RATIO=2.0       # 信用倍率（グロス目標=元本×2.0=¥40M。上限3.3）
   LIVE_NAMES_PER_SIDE=8       # R6検証済みの本番値
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

## 本番移行チェックリスト（2026-07-29 Windows VPSで実施済みの記録）
- [x] 検証環境(18081)は**板が全null・注文はスタブ**（受理のみ・OrderId=null・建玉生成なし）と判明。
      板=本番読取＋発注=検証の `HybridKabuClient`（`KABU_ENV=test`＋`KABU_DATA_ENV=prod`）で
      plan coverage 96.5%・ペーパー発注19件受理まで確認
- [x] **信用注文は Exchange=9(SOR) 必須**。東証直指定(1)は 100368 で全拒否（実測）。
      指値・寄指（板寄せ系）とも SOR で受理→取消確認。`kabu_client` は SOR 既定＋東証フォールバック
- [x] 発注パスワード・一日信用(MarginTradeType=3)・一般口座(AccountType=2) の実受理を確認。
      特定口座(4)は 100203「特定口座設定約諾書」未確認で発注不可 → 約諾解消まで
      `LIVE_ACCOUNT_TYPE=2` で運用（一般口座分の損益は確定申告で自己計算）
- [x] `LIVE_MAX_GROSS_YEN` 安全弁・比例縮小・売建可否チェックは preflight/リハーサルで動作確認
- [x] 毎営業日 08:50 に発注経路プローブ（約定不能指値→即取消、`scripts/preflight_order_probe.py`）。
      entry(08:59) 前に口座抑止・認証切れをダッシュボードで検知できる
- [ ] 寄成の想定始値 vs 実約定のズレ（板の予想始値精度）— 実弾初日以降に計測
- [ ] 少額（¥5M・5銘柄/側から）で本番を数日 → 段階増資

## Windows VPS 運用（タスクスケジューラ登録済み・平日JST）
| 07:40 collect | 08:50 probe | 08:55 plan | 08:59 entry | 14:55 exit | 15:40 state |
|---|---|---|---|---|---|

- 大引けは **15:30**（2024-11-05以降・15:25からプレ・クロージング）。exit 14:55 発注→15:30 板寄せ約定
- **kabuステーションの自動ログイン設定が必須**。夜間セッション切れで /token が 401 になり
  entry が不発になる（2026-07-29 朝に実発生）

### 朝の自動ログイン（07:25 `TradeBot-00-KabuLogin`）

`scripts\ensure_kabu_login.ps1` が冪等に実行される（API が 200 なら何もしない）。
未ログインなら KabuS.exe を再起動し、WebView2 の多段フォームを UIAutomation で進める:

| 段 | 画面 | 自動化 |
|---|---|---|
| [1] | 口座番号（AutoId=`username`）→「次へ」 | DPAPI 保存の資格情報 |
| [2] | パスワード（IsPassword）→「ログイン」 | 同上 |
| [3] | **ワンタイム認証コード**（AutoId=`code`）→「続ける」 | Gmail(IMAP) から自動取得 |

- kabuステーションは**毎回**メール認証コードを要求する（「この端末を信頼する」は無い）。
  `scripts\fetch_otp.ps1` → `trading/jp_intraday/live/otp_mail.py` が Gmail を IMAP で読む。
- **古いコードを使い回さない**のが安全弁: パスワード投入時刻より後に届いたメールしか見ない。
  メールボックスは readonly で開く（既読化も削除もしない）。試行は既定2回まで。
- 資格情報は 2ファイルとも DPAPI 暗号化（このユーザー・このマシンでしか復号できない）:
  - `data\live_reports\.kabu_creds.xml` … `scripts\setup_kabu_credentials.ps1`
  - `data\live_reports\.gmail_otp.xml` … `scripts\setup_gmail_otp.ps1`
    （Gmail の**アプリパスワード**16桁。2段階認証をONにすると発行できる。通常のGoogleパスワード不可）
- 設定確認: `powershell -File scripts\fetch_otp.ps1 -Probe`（直近7日の該当メールを一覧）。
  差出人が既定パターンに合わなければ `.env` の `OTP_MAIL_FROM` で上書きする。
- 手動で通したいとき: `powershell -File scripts\ensure_kabu_login.ps1 -Force`
- **UI自動化なので対話セッションが必要**。RDP はログオフせず**切断**すること。
- 終了コード: 0=ログイン済/成功, 1=タイムアウト, 2=設定不備, 3=認証コードが通らない（要手動）
- タスクは**失敗時に5分間隔で2回まで自動リトライ**（`RestartCount=2` / `RestartInterval=PT5M`）。
  それでも 08:44 の plan ログが 401 なら手動介入（`-Force` を実行）。
- ランナー `scripts/run_live_task.ps1` は `PYTHONUTF8=1` を強制（タスクスケジューラの stdout は
  cp932 で、CONFIG 行の ¥ / 絵文字印字が UnicodeEncodeError で落ちる実障害があった）

## 未検証・要注意
- 板の「予想始値」フィールド名は環境差があり得るため、`executor._est_open` を実データで要確認。
- 返済 (CashMargin=3) の SOR 可否は建玉が無いと実測できないため、SOR→東証の順で自動フォールバック
  実装とした。実弾初日の exit ログで確定させること。
- ML戦略は年次学習済みモデル（data/live_models/ridge_YYYY.json）で稼働。年初に
  `python -m trading.jp_intraday.live.run_live train` を1回実行して更新する（年次で十分・R4検証済み）。
