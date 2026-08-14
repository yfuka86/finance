# Xトレードアイデア収集 — 運用手順

## 目的と境界

Xは仮説の発見源にだけ使う。投稿人気、センチメント、投稿後リターンをそのまま売買シグナルにせず、
公式/PITデータで再定義できる仮説だけを人間がレビューする。候補数は制限せず、既存研究枠とは
別のアイデア・バックログへ継続蓄積する。

## 実行方式

既定はPlaywrightでXの検索画面を描画するブラウザ方式。既存のログインCookieをメモリ内だけで使い、
投稿データやログへCookie値を保存しない。

```bash
python scripts/collect_x_trade_ideas.py run
```

既定Cookie: `/Users/yutafukazawa/work/tc/secrets/x_cookies.json`

画面を表示して診断する場合は `--show-browser`、別Cookieなら `--cookie-file PATH` を指定する。

## 公式API方式（任意）

X Developer ConsoleでAppのBearer Tokenを発行し、Git管理外の `.env` に次を追加する。

```dotenv
X_BEARER_TOKEN=...
```

検索語と仮説テンプレートは `config/x_trade_ideas.json` で管理する。recent searchは直近7日なので、
欠測を避けるには最低でも毎日1回実行する。

```bash
python scripts/collect_x_trade_ideas.py run --backend api
```

収集だけ、再分析だけならそれぞれ以下を使う。

```bash
python scripts/collect_x_trade_ideas.py collect --backend browser
python scripts/collect_x_trade_ideas.py analyze
```

macOSの例（毎日07:15。cron/launchdの環境に合わせてPythonの絶対パスを指定すること）:

```cron
15 7 * * * cd /Users/yutafukazawa/work/finance && .venv/bin/python scripts/collect_x_trade_ideas.py run >> data/x_trade_ideas/collector.log 2>&1
```

Windowsではタスクスケジューラで同じコマンドを日次実行し、既存運用と同様に
`PYTHONUTF8=1` を設定する。

## 出力と判断

- `data/x_trade_ideas/posts.jsonl`: post ID、投稿時刻、初回取得時刻、公開指標のappend-only原票
- `data/x_trade_ideas/runs.jsonl`: 実行記録と設定hash
- `data/x_trade_ideas/reports/ideas_*.json`: トレンド順位と反証条件付きhypothesis card

`REVIEW_REQUIRED` は戦略採用を意味しない。候補を採る場合は、既検証テーマとの重複を確認し、
データを見る前に主仕様・感度・Kill条件・未使用評価日を別文書へ事前登録する。複数候補を並行して
保持・検証してよいが、同じ未使用期間を見ながら派生案を選び直してはならない。

X APIのプランや利用量課金、保存・削除義務は変更され得る。Developer Consoleの契約と利用状況を
定期確認し、削除要請等がある場合はXの現行ポリシーを優先する。
