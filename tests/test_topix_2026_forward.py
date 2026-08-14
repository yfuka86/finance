import importlib.util
from pathlib import Path
import pandas as pd

spec=importlib.util.spec_from_file_location("topix_forward",Path("scripts/topix_2026_forward.py"))
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

def test_current_weight_merge_normalizes_jquants_five_digit_code():
    day=pd.DataFrame({"Code":["13010","72030"]})
    w=pd.DataFrame({"Code":["1301"],"topix_weight":[.001],"ニューインデックス区分":["Small"]})
    out=mod.merge_current_weights(day,w)
    assert out.current_topix_member.tolist()==[True,False]
    assert out.loc[0,"topix_weight"]==.001
