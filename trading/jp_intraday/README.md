# 日本株・分足ロングショート検証

未来情報を使わないウォークフォワード検証です。各foldでパラメータとエントリー閾値を過去の学習期間だけから決め、直後のテスト期間を一度だけ評価します。バー `t` の終値でシグナルを確定し、`t+1` の始値で約定する前提です。

入力は `timestamp,symbol,open,high,low,close,volume` 列を持つ1分足CSVまたはParquetです。タイムスタンプはJST推奨（タイムゾーンなしならJSTとして扱います）。複数銘柄が同じ時刻に必要です。

```bash
python -m trading.jp_intraday data/jp_1m.csv --interval 1 --train-days 40 --test-days 10
python -m trading.jp_intraday data/jp_1m.csv --interval 5 --train-days 80 --test-days 20
```

結果の `folds_*.csv` は学習・テスト期間と、学習期間だけで選ばれた設定を監査できます。`returns_*.csv` は未見テスト期間だけのリターンです。

注意: 現行戦略は研究用ベースラインです。実運用前には銘柄ごとの信用売建可否、売買単位、値幅制限、特別気配、注文遅延、部分約定、空売り価格規制、貸株料・逆日歩を追加してください。

## TOPIX・1,000億円ユニバースとモデル

`research` はTOPIX採用期間と、その時点までに開示済みの発行済株式数、前営業日終値から日次ユニバースを作ります。現在のTOPIX一覧を昔の期間へ適用してはいけません。長期検証ではJ-Quants DataCube等から月末時点の過去構成銘柄を取得し、`symbol,effective_from,effective_to` に変換してください。

時価総額は `前営業日終値 × (発行済株式数 - 自己株式数)` です。株式数の `known_at` は決算期末でなく開示日を指定します。

```bash
export JQUANTS_API_KEY='ローテーション後のキー'
python - <<'PY'
from trading.jp_intraday.collector import collect_jquants_minutes
collect_jquants_minutes(['7203', '6758'], '2026-06-01', '2026-06-30', 'data/jp_minutes')
PY

python -m trading.jp_intraday.research \
  data/jp_minutes/jp_1m_2026-06-01_2026-06-30.parquet \
  data/jp_intraday_reference/topix_memberships_current.csv \
  data/jp_intraday_reference/share_snapshots.csv \
  --interval 1 --train-days 60 --test-days 10
```

モデルの特徴量は1/5/15バーリターン、15バー実現ボラティリティ、セッションVWAP乖離、出来高ショック、バー内終値位置、市場リターン、時刻周期です。各fold内の古い80%で学習し、新しい20%で正則化を選択してから、その後の未見テストだけを評価します。
