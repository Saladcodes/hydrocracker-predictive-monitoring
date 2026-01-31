import re
import pandas as pd
from pathlib import Path
from ofm_fg_ofm.config.tag_renaming import TAG_MAP, build_column_rename_map

def suggest_name(col: str) -> str:
    # Suggest a safe anonymized name preserving instrument meaning
    # Examples:
    # S.B1.302TI1948 -> u302_ti_1948
    # S.LIMS...Vis@040C -> lims_vis_40c
    if col.startswith("S.B1."):
        m = re.match(r"S\.B1\.(\d+)([A-Z]+)(\d+)", col)
        if m:
            unit, inst, num = m.groups()
            return f"u{unit}_{inst.lower()}_{num}"
        return "b1_tag_unknown"

    if col.startswith("S.C."):
        return re.sub(r"[^a-zA-Z0-9]+", "_", col).lower().strip("_")

    if col.startswith("S.LIMS."):
        # keep key analytical meaning
        s = col
        s = s.replace("S.LIMS.Q:", "lims_")
        s = re.sub(r"[^a-zA-Z0-9]+", "_", s).lower().strip("_")
        return s[:60]  # prevent crazy long names

    return col

def main():
    df = pd.read_csv("data/processed/dataset.csv", nrows=1)
    cols = list(df.columns)

    rename_map = build_column_rename_map(cols, TAG_MAP)

    rows = []
    for c in cols:
        mapped = rename_map.get(c, "")
        base_suggest = suggest_name(c.split("_")[0]) if "_" in c else suggest_name(c)
        # if suffix exists, preserve it
        suffix = c[len(c.split("_")[0]):] if "_" in c else ""
        if mapped:
            new = mapped
        else:
            new = base_suggest + suffix

        rows.append({
            "old_column": c,
            "new_column": new,
            "meaning": "",     # you fill
            "unit": "",        # you fill
            "source": "DCS" if c.startswith("S.B1") or c.startswith("S.C") else ("LIMS" if c.startswith("S.LIMS") else ""),
        })

    out = Path("docs/tag_mapping_template.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print("Wrote:", out)

if __name__ == "__main__":
    main()
