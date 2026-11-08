"""Load the six input tables and rename columns using columns_map.py."""
import os
import pandas as pd

import config
from columns_map import FILES, COLUMNS


def _load(name, data_dir):
    path = os.path.join(data_dir, FILES[name])
    df = pd.read_csv(path)
    rename = {theirs: ours for ours, theirs in COLUMNS[name].items() if theirs in df.columns}
    df = df.rename(columns=rename)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def load_all(data_dir=None):
    data_dir = data_dir or config.DATA_DIR
    t = {n: _load(n, data_dir) for n in FILES}
    # The practice generator also writes the "true" chlorine it used. A real utility never
    # has that column, so Raqeeb drops it and learns chlorine only from lab results.
    t["daily"] = t["daily"].drop(columns=["expected_free_cl_mgL"], errors="ignore")
    t["results"] = flag_abnormal(t["results"])
    return t


def flag_abnormal(r):
    L = config.LIMITS
    r = r.copy()
    reasons = []
    for _, x in r.iterrows():
        why = []
        if x.free_cl_mgL < L["free_chlorine_min_mgL"]:
            why.append("low chlorine")
        if x.free_cl_mgL > L["free_chlorine_max_mgL"]:
            why.append("high chlorine")
        if x.turbidity_NTU > L["turbidity_max_NTU"]:
            why.append("high turbidity")
        if not (L["ph_min"] <= x.ph <= L["ph_max"]):
            why.append("pH out of range")
        if x.total_coliform_cfu100mL > L["total_coliform_max_cfu100mL"]:
            why.append("coliform detected")
        if x.ecoli_cfu100mL > L["ecoli_max_cfu100mL"]:
            why.append("E. coli detected")
        reasons.append(", ".join(why))
    r["fail_reasons"] = reasons
    r["is_abnormal"] = r["fail_reasons"] != ""
    r["near_miss"] = (~r["is_abnormal"]) & (
        r["free_cl_mgL"] < config.NEAR_MISS_FACTOR * L["free_chlorine_min_mgL"])
    return r
