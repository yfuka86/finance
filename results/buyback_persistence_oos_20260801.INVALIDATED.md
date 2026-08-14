# INVALIDATED — buyback persistence historical result

The reported 20bps Sharpe 2.37 must not be used for research or production decisions.

After the one-shot result was opened, adversarial review found that
`purchase_month_sessions` counted only sessions between the first and last observed purchase.
The preregistered economic definition requires every exchange session in the reported month,
including zero-purchase days. The bug inflated `realized_daily_pressure` and
`purchase_day_ratio`, changing candidate selection.

The raw EDINET documents and the original result are retained for audit. The derived panel is
corrected for forward signal generation. Because the bug affects returns and selection, the same
historical period cannot be rerun and presented as OOS. Evaluation restarts on a new unused
forward period.
