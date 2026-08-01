"""Parse EDINET own-share acquisition reports from official inline XBRL HTML."""
from __future__ import annotations
import io,re,zipfile
import pandas as pd

_ZEN=str.maketrans("０１２３４５６７８９", "0123456789")

def _num(v):
    if pd.isna(v): return None
    s=str(v).translate(_ZEN).replace(",","").replace("%","").strip()
    if s in ("","-","－","―","nan"): return None
    try: return float(s) if "." in s else int(s)
    except ValueError: return None

def _date_parts(text):
    m=re.search(r"(\d{4})年\s*(\d{1,2})月\s*(\d{1,2})日",str(text).translate(_ZEN))
    return tuple(map(int,m.groups())) if m else None

def parse_ixbrl_zip(content: bytes) -> dict:
    """Return structured board-resolution and execution facts, preserving daily rows."""
    z=zipfile.ZipFile(io.BytesIO(content))
    names=[n for n in z.namelist() if "honbun" in n and n.endswith(".htm")]
    if not names: raise ValueError("EDINET zip has no honbun ixbrl html")
    html=z.read(names[0]); tables=pd.read_html(io.BytesIO(html))
    report_date=None; acquisition=None
    for t in tables:
        flat=" ".join(map(str,t.columns))+" "+" ".join(map(str,t.astype(str).values.ravel()))
        if report_date is None and "現在" in flat:
            report_date=_date_parts(flat)
        if t.shape[1]>=4 and "取得自己株式" in flat and "取締役会" in flat:
            acquisition=t.copy()
    if acquisition is None: raise ValueError("acquisition table not found")
    t=acquisition.reset_index(drop=True); label=t.iloc[:,0].astype(str); sub=t.iloc[:,1].astype(str)
    board_i=label.str.contains("取締役会",na=False).idxmax()
    board_text=label.loc[board_i]; dates=re.findall(r"\d{4}年\s*\d{1,2}月\s*\d{1,2}日",board_text.translate(_ZEN))
    parsed_dates=[_date_parts(x) for x in dates]
    total_idx=label[label.eq("計")].index
    cum_idx=label[label.str.contains("報告月末現在の累計",na=False)].index
    prog_idx=label[label.str.contains("進捗状況",na=False)].index
    daily=[]
    for i,row in t.iterrows():
        dm=re.search(r"(\d{1,2})月\s*(\d{1,2})日",sub.iloc[i].translate(_ZEN))
        shares=_num(row.iloc[2]); yen=_num(row.iloc[3])
        if dm and shares is not None and yen is not None and report_date:
            daily.append({"date":f"{report_date[0]:04d}-{int(dm.group(1)):02d}-{int(dm.group(2)):02d}",
                          "shares":shares,"yen":yen})
    def vals(index):
        if len(index)==0: return (None,None)
        row=t.loc[index[0]]; return _num(row.iloc[2]),_num(row.iloc[3])
    month_shares,month_yen=vals(total_idx); cum_shares,cum_yen=vals(cum_idx)
    progress_shares,progress_yen=vals(prog_idx)
    return {"report_date":"-".join(map(lambda x:f"{x:02d}",report_date)) if report_date else None,
            "board_meeting_date":"-".join(map(lambda x:f"{x:02d}",parsed_dates[0])) if parsed_dates else None,
            "period_start":"-".join(map(lambda x:f"{x:02d}",parsed_dates[1])) if len(parsed_dates)>1 else None,
            "period_end":"-".join(map(lambda x:f"{x:02d}",parsed_dates[2])) if len(parsed_dates)>2 else None,
            "max_shares":_num(t.loc[board_i].iloc[2]),"max_yen":_num(t.loc[board_i].iloc[3]),
            "month_shares":month_shares,"month_yen":month_yen,"cumulative_shares":cum_shares,
            "cumulative_yen":cum_yen,"progress_shares_pct":progress_shares,
            "progress_yen_pct":progress_yen,"daily_acquisitions":daily,
            "tostnet3_mentioned":b"ToSTNeT-3" in html or b"ToSTNET-3" in html}
