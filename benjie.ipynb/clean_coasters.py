"""Load coaster_database.db and return a model-ready DataFrame.

Usage:  python clean_coasters.py path/to/coaster_database.db  -> writes coasters_clean.csv
Or:     from clean_coasters import load_clean; df = load_clean("coaster_database.db")
"""
import re
import sqlite3
import sys

import numpy as np
import pandas as pd

STAR_COLS = [f"rating_{s}_stars" for s in
             ["0_5", "1_0", "1_5", "2_0", "2_5", "3_0", "3_5", "4_0", "4_5", "5_0"]]
STAR_VALUES = np.arange(0.5, 5.01, 0.5)


def _parse_opening_year(s):
    """Dates come as '1996' or 'm/d/yy'. Two-digit years are ambiguous;
    yy <= 27 -> 20yy, else 19yy (nothing in the data opens after 2027)."""
    if pd.isna(s):
        return np.nan
    s = s.strip()
    if re.fullmatch(r"\d{4}", s):
        return int(s)
    m = re.fullmatch(r"\d{1,2}/\d{1,2}/(\d{2})", s)
    if m:
        yy = int(m.group(1))
        return 2000 + yy if yy <= 27 else 1900 + yy
    return np.nan


def _split_launch(s):
    """'Chain lift hill\\n    ,   Cable lift hill' -> ['Chain lift hill', 'Cable lift hill']"""
    if pd.isna(s):
        return []
    return [p.strip() for p in re.sub(r"\s+", " ", s).split(",") if p.strip()]


def _collapse_rare(series, min_count, other="Other"):
    counts = series.value_counts()
    keep = counts[counts >= min_count].index
    return series.where(series.isin(keep) | series.isna(), other)


def load_clean(db_path, min_ratings=20, rare_min=15):
    with sqlite3.connect(db_path) as con:
        raw = pd.read_sql("SELECT * FROM coasters", con)

    df = pd.DataFrame({"uid": raw["uid"], "name": raw["name"], "park": raw["park"]})

    # ---- targets (pick ONE per model; never use one as a feature for the other) ----
    counts = raw[STAR_COLS].to_numpy()
    df["n_ratings"] = counts.sum(axis=1)
    df["avg_stars"] = (counts * STAR_VALUES).sum(axis=1) / df["n_ratings"]
    df["duels"] = raw["duels"]

    # ---- numeric features (NaN kept; impute inside the model pipeline, fit on train only) ----
    for col in ["height", "length", "top_speed", "inversions"]:
        df[col] = raw[col]
    df["opening_year"] = raw["opening_date"].map(_parse_opening_year)

    # ---- categorical features ----
    df["type"] = raw["type"].fillna("Unknown")
    df["train"] = _collapse_rare(raw["train"].fillna("Unknown"), rare_min)
    df["restraint"] = raw["restraint"].replace({"Common shoulder harness": "Shoulder harness"})
    df["status"] = raw["status"]
    df["manufacturer"] = _collapse_rare(raw["manufacturer"], rare_min)
    df["country"] = _collapse_rare(raw["country"], rare_min)
    df["model"] = _collapse_rare(raw["model"].fillna("Unknown"), rare_min)

    # ---- launch: multi-valued -> one binary column per mechanism ----
    launches = raw["launch"].map(_split_launch)
    df["launch_missing"] = raw["launch"].isna().astype(int)
    for mech in sorted({m for lst in launches for m in lst}):
        col = "launch_" + re.sub(r"\W+", "_", mech.lower()).strip("_")
        df[col] = launches.map(lambda lst, m=mech: int(m in lst))
    df["is_launched"] = launches.map(lambda lst: int(any("launch" in m.lower() for m in lst)))

    # cost dropped: 93% missing and in 11 currencies from different years.

    # drop rows whose target average rests on too few votes
    df = df[df["n_ratings"] >= min_ratings].reset_index(drop=True)
    return df


if __name__ == "__main__":
    out = load_clean(sys.argv[1])
    out.to_csv("coasters_clean.csv", index=False)
    print(out.shape)
    print(out.isna().sum()[out.isna().sum() > 0])
