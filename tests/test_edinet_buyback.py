import io,zipfile
from trading.jp_intraday.edinet_buyback import parse_ixbrl_zip

def _zip(html):
    b=io.BytesIO()
    with zipfile.ZipFile(b,"w") as z: z.writestr("XBRL/PublicDoc/010_honbun_ixbrl.htm",html)
    return b.getvalue()

def test_parse_official_style_acquisition_table():
    html='''<html><head><meta charset="utf-8"></head><body><table><tr><td>2026年7月24日現在</td></tr></table>
    <table><tr><th>区分</th><th>株式数（株）</th><th>株式数（株）</th><th>価額の総額（円）</th></tr>
    <tr><td>取締役会（2026年1月14日）での決議状況 （取得期間 2026年1月15日～2026年10月31日）</td><td>280000</td><td>280000</td><td>210000000</td></tr>
    <tr><td>報告月における取得自己株式（取得日）</td><td>7月1日</td><td>600</td><td>389500</td></tr>
    <tr><td>計</td><td>－</td><td>61100</td><td>60457900</td></tr>
    <tr><td>報告月末現在の累計取得自己株式</td><td>272100</td><td>272100</td><td>209957600</td></tr>
    <tr><td>自己株式取得の進捗状況（％）</td><td>97.17%</td><td>97.17%</td><td>99.97%</td></tr></table></body></html>'''
    x=parse_ixbrl_zip(_zip(html))
    assert x["max_shares"]==280000 and x["cumulative_yen"]==209957600
    assert x["period_end"]=="2026-10-31"
    assert x["daily_acquisitions"]==[{"date":"2026-07-01","shares":600,"yen":389500}]
