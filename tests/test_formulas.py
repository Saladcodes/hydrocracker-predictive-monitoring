import pandas as pd
import numpy as np
from ofm_fg_ofm.features.fg_formulas import compute_fg_mw, compute_fg_lhv

def test_fg_mw_lhv_shapes():
    # minimal DF with composition cols
    cols = {
        "S.LIMS.Q:U-31100:02514::C1:F": [80, 70],
        "S.LIMS.Q:U-31100:02514::C2:F": [5, 10],
        "S.LIMS.Q:U-31100:02514::C2=:F": [1, 1],
        "S.LIMS.Q:U-31100:02514::C3:F": [3, 4],
        "S.LIMS.Q:U-31100:02514::C3=:F": [1, 1],
        "S.LIMS.Q:U-31100:02514::C4-i:F": [1, 2],
        "S.LIMS.Q:U-31100:02514::C4-n:F": [1, 2],
        "S.LIMS.Q:U-31100:02514::C4=i:F": [1, 2],
        "S.LIMS.Q:U-31100:02514::C4=1:F": [1, 1],
        "S.LIMS.Q:U-31100:02514::C4=2c:F": [1, 1],
        "S.LIMS.Q:U-31100:02514::C4=2t:F": [1, 1],
        "S.LIMS.Q:U-31100:02514::C5-i:F": [1, 1],
        "S.LIMS.Q:U-31100:02514::C5-n:F": [1, 1],
        "S.LIMS.Q:U-31100:02514::C6+:F": [1, 2],
        "S.LIMS.Q:U-31100:02514::H2": [2, 1],
    }
    df = pd.DataFrame(cols)
    mw = compute_fg_mw(df)
    lhv = compute_fg_lhv(df)
    assert mw.shape == (2,)
    assert lhv.shape == (2,)
    assert float(mw.iloc[0]) > 0
    assert float(lhv.iloc[0]) > 0
