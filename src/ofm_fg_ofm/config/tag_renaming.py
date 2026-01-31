from __future__ import annotations
from typing import Dict, Tuple
import pandas as pd

TAG_MAP: Dict[str, str] = {
    # Temperatures
    "S.B1.302TI1948": "rx_temp_1",
    "S.B1.302TI1947": "rx_temp_2",
    "S.B1.302TI1940": "rx_temp_3",
    "S.B1.302TC1510": "rx_temp_ctrl",
    "S.B1.302TI2076": "rx_temp_4",
    "S.B1.302TI2051": "rx_temp_5",

    # Recycle / header meters
    "S.B1.302FI0701": "recycle_gas_flow",
    "S.B1.302FI6901": "fh_fg_header_flow",
    "S.B1.306FI6901": "sh_fg_header_flow",

    # FH fuel gas users
    "S.B1.302FC0405": "fh_fg_flow_1",
    "S.B1.302FC0806": "fh_fg_flow_2",
    "S.B1.302FC1505": "fh_fg_flow_3a",
    "S.B1.302FC1506": "fh_fg_flow_3b",
    "S.B1.302FC0910": "fh_fg_flow_4",

    # SH fuel gas users
    "S.B1.306FI0701": "sh_fg_flow_1",
    "S.B1.306FI0106": "sh_fg_flow_2",
    "S.B1.306FI1106": "sh_fg_flow_3",

    # LIMS viscosity tags
    "S.LIMS.Q:U-30200:02271:D-30201:Vis@040C": "fh_vis_40C_cSt",
    "S.LIMS.Q:U-30200:02271:D-30201:Vis@100C": "fh_vis_100C_cSt",
    "S.LIMS.Q:U-30200:02561::Vis@040C": "sh_vis_40C_cSt",
    "S.LIMS.Q:U-30200:02561::Vis@100C": "sh_vis_100C_cSt",

    # OPTIONAL (also client-ish)
    "S.C.302F0102-1": "fh_feed_flow",
    "S.C.302F0101-1": "sh_feed_flow",
}

def build_column_rename_map(columns, tag_map: Dict[str, str]) -> Dict[str, str]:
    rename: Dict[str, str] = {}
    for col in columns:
        for tag, friendly in tag_map.items():
            if col == tag or col.startswith(tag + "_"):
                rename[col] = friendly + col[len(tag):]  # preserves suffix like _lag1/_roc
                break
    return rename

def rename_by_tag_map(df: pd.DataFrame, tag_map: Dict[str, str] = TAG_MAP) -> Tuple[pd.DataFrame, Dict[str, str]]:
    rename = build_column_rename_map(df.columns, tag_map)
    return df.rename(columns=rename), rename
