# 事前登録: 自社株買い実行圧力

登録日: 2026-08-01。`BUYBACK_PRESSURE` 1ファミリーとして管理し、下記3状態を独立報告する。

## 凍結後の一回限り過去確認（2026-08-01）

結果を確認する前に固定した `steady execution` を、エントリー日2026-03-01以降で一度だけ評価した。
20bps往復コスト後は1案件、案件リターン -2.32%、ポートフォリオSharpe -0.53、40bpsでも
マイナスだった。当初は最低20案件を満たさないためNO-GOとしたが、後日
`purchase_month_sessions` の分母不備が判明し、この判定自体を **INVALIDATED** とした。
サンプル不足なので自社株買い一般の
反証とはせず、この仕様を過去データで支持できなかったという判定に限定する。閾値は結果を見て変更せず、
2026-08-03以降のフォワード観測は別枠で継続する。正本は
`results/buyback_steady_oos_20260801.json`。

## データとPIT

- TDnetの決議・状況・完了、EDINETの月内日次取得実績、ToSTNeT-3を使用。
- 公式構造化データの履歴はTDnet 2013年～、EDINET 2019年～、ToSTNeT-3 2008年～。
- `source_published_at`、`first_received_at`、`first_tradable_at`を分離。開示当日の価格へ遡及しない。
- 訂正はrevision追加。過去レコードを上書きしない。ToSTNeT-3は市場内買付圧力と分離する。
- 個人向けTDnet add-on原文または正式な構造化exportを入力にし、PDF検索結果をデータに使わない。

## 固定特徴量

- `remaining_shares = max_shares - cumulative_shares`
- `remaining_yen = max_yen - cumulative_yen`
- `remaining_capacity_shares = min(remaining_shares, remaining_yen / prior_close)`
- `remaining_pressure = remaining_capacity_shares / (remaining_sessions * ADV20_shares)`
- `pace_surprise = cumulative_shares / (max_shares * elapsed_sessions / total_sessions) - 1`
- program規模、期間、取得方法、完了/取消、残日数も保存する。

## 3つの状態仮説

1. `acceleration`: remaining pressureが高くpace surpriseも正なら、次の5～40日も買付需要が続く。
2. `underexecution`: 発表規模は大きいが実績が線形計画を大幅に下回ると発表効果が剥落する。
3. `completion_cliff`: 大規模プログラム完了後に買付需要消失の反動が出る。

閾値・保有期間はデータ取得前に別紙で一度だけ固定する。現段階では分布未確認のため売買ルールを
仮置きしない。比較対象は業種・時価総額・β・valuationを固定マッチし、ショートは貸借・規制・
証券会社在庫を必須とする。非貸借・低位株へ利益が集中したらKill。

## データ取得・閾値凍結（2026-08-01、リターン未確認）

無料EDINET APIの220/230を全走査。公開APIで実際に取得できた期間は2025-08-01～2026-07-07。
ユニーク原本5,142件、解析成功5,046件、1,604プログラム、日別取得45,953行。ToSTNeT-3記載を
除外し、未調整O/C/Vo、ADV20、残営業日まで完全な特徴量は2,558報告・883プログラム・672銘柄。

リターンを見ずに得た分布は remaining pressure p50=3.94%、p90=19.85%、pace surprise
p10=-0.63、p90=+0.57。この分布だけから以下へ丸めて固定する。

- acceleration: remaining pressure>=0.20 かつ pace surprise>=+0.55。20営業日ロング。
- underexecution: remaining pressure>=0.20 かつ pace surprise<=-0.60。20営業日ショート候補。
- completion cliff: 累計取得率が初めて95%以上となり、直前報告のremaining pressure>=0.10。
  20営業日ショート候補。
- ショート候補は貸借・規制なし・証券会社在庫ありの場合だけ発注。満たさなければ観測のみ。
- 1案件NAV 3%、最大10案件、株式のみ、往復20bps。翌営業日寄付きから取引可能。

### 主運用追加状態: steady execution long

日別取得実績を使い、一括取得ではなく継続的に市場内買付を行う企業をロングする独立状態。
2026-08-01、件数・リターン確認前に以下を固定する。

- remaining pressure>=0.10、pace surprise>=0。
- 報告対象月の取引日の50%以上で取得実績がある。
- 最大1日取得株数/月間取得株数<=0.25。
- remaining sessions>=10。
- ToSTNeT-3記載なし、前日売買代金>=10億円、100株単位で1案件NAV 3%以内。
- 翌営業日寄付きでロング。20営業日、完了、取消、残pressure<0.02の最初で退出。
- 同時最大10件。候補超過時はremaining pressure降順、同値はsecurity code昇順。結果で変更しない。

`acceleration` と `steady execution` が同時成立しても1案件として扱う。株式ロングだけで完結し、
指数先物ヘッジは使わない。underexecution/completion cliffは貸借確認前の観測候補であり主運用外。

### 履歴確認プロトコル（2026-08-01、リターン確認前）

- 確認期間はentry date 2026-03-01以降。特徴量分布の計算には使ったが、収益・方向選択には未使用。
- 主判定はsteady execution単独。acceleration単独と両者unionは診断であり採用選択に使わない。
- 前日売買代金>=10億円、1単元価格<=NAV 3%=60万円、100株整数単元。
- 20bps後Sharpe>=1、取引>=20、DD>-10%、最大利益取引寄与<20%、40bpsでも正をGO候補条件。
- 短期間のため基準達成時も `FORWARD CANDIDATE`。2026-08-03以降を変えずに観測する。

評価開始は2026-08-03。過去1年は特徴量分布の固定に使ったため成績評価には使用しない。
6か月時点は完全性監査のみ、主評価は2027-08-02に一度だけ行う。
