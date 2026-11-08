"""Turn the raw tables into the evidence Raqeeb uses for each result."""
import numpy as np
import pandas as pd

import config

QUALITY_COMPLAINTS = {"taste_odour", "discoloured"}

# Features of the site-expectation model (what chlorine SHOULD be here today)
EXPECT_FEATURES = ["water_temp_C", "water_age_h", "month", "rural", "src_desal", "src_well", "src_blend",
                   "base_water_age_h", "site_cl_median"]

# Evidence used by the cause classifier
CAUSE_FEATURES = [
    "cl_surprise",            # how far chlorine is from what this site normally shows today
    "cl_measured", "cl_expected",
    "log_turbidity", "total_coliform", "ecoli",
    "micro_only",             # bacteria found while chlorine and turbidity look normal
    "qc_cl_dev", "qc_turb_dev",  # how far the batch reference material is from 100%
    "qc_out_of_control",
    "batch_abnormal_share",   # share of the batch's samples that are abnormal
    "repairs_recent",         # repairs in the same DMA in the last few days
    "quality_complaints_recent",
    "water_temp_C", "water_age_h", "rural",
]


def site_medians(results, train_months):
    """Each site's typical chlorine from its PASSING results in the training months."""
    ok = results[(results["date"].dt.month <= train_months) & ~results["is_abnormal"]]
    return ok.groupby("site_id")["free_cl_mgL"].median().round(3).to_dict()


def add_site_columns(df, medians):
    df["month"] = df["date"].dt.month
    df["rural"] = (df["zone"] == "rural").astype(int)
    for s, col in [("desalinated", "src_desal"), ("well", "src_well"), ("blend", "src_blend")]:
        df[col] = (df["source"] == s).astype(int)
    fallback = float(np.median(list(medians.values()))) if medians else np.nan
    df["site_cl_median"] = df["site_id"].map(medians).fillna(fallback)
    if "base_water_age_h" not in df.columns:
        df["base_water_age_h"] = np.nan  # LightGBM handles missing values
    return df


def site_context(t, medians):
    """Results joined with site info and that day's temperature and water age."""
    r = t["results"].merge(t["sites"], on="site_id", how="left")
    r = r.merge(t["daily"], on=["site_id", "date"], how="left")
    return add_site_columns(r, medians)


def add_cause_features(r, t, expect_model, resid_sd):
    r = r.copy()
    r["cl_expected"] = expect_model.predict(r[EXPECT_FEATURES])
    r["cl_measured"] = r["free_cl_mgL"]
    r["cl_surprise"] = (r["cl_measured"] - r["cl_expected"]) / resid_sd
    r["log_turbidity"] = np.log10(r["turbidity_NTU"].clip(lower=0.01))
    r["total_coliform"] = r["total_coliform_cfu100mL"]
    r["ecoli"] = r["ecoli_cfu100mL"]
    L = config.LIMITS
    r["micro_only"] = (((r["total_coliform"] > 0) | (r["ecoli"] > 0))
                       & (r["cl_measured"] >= L["free_chlorine_min_mgL"])
                       & (r["turbidity_NTU"] <= L["turbidity_max_NTU"])).astype(int)

    # lab health of the batch
    qc = t["qc"].pivot_table(index="batch_id", columns="parameter", values="crm_recovery_pct")
    qc = qc.rename(columns={"free_chlorine": "qc_cl", "turbidity": "qc_turb", "ph": "qc_ph"})
    r = r.merge(qc[["qc_cl", "qc_turb"]], left_on="batch_id", right_index=True, how="left")
    r["qc_cl_dev"] = (r["qc_cl"] - 100).abs()
    r["qc_turb_dev"] = (r["qc_turb"] - 100).abs()
    r["qc_out_of_control"] = ((r["qc_cl_dev"] > config.QC_TOLERANCE_PCT)
                              | (r["qc_turb_dev"] > config.QC_TOLERANCE_PCT)).astype(int)
    r["batch_abnormal_share"] = r.groupby("batch_id")["is_abnormal"].transform("mean")

    # network and customer context, counted per DMA in a window before the sample date
    r["repairs_recent"] = _window_count(r, t["repairs"], config.REPAIR_WINDOW_DAYS)
    qcompl = t["complaints"][t["complaints"]["category"].isin(QUALITY_COMPLAINTS)]
    r["quality_complaints_recent"] = _window_count(r, qcompl, config.COMPLAINT_WINDOW_DAYS, include_same_day=True)
    return r


def _window_count(r, events, days, include_same_day=False):
    out = np.zeros(len(r), dtype=int)
    if events.empty:
        return out
    ev = events.groupby("dma_id")["date"].apply(lambda s: np.sort(s.values))
    lo_off = pd.Timedelta(days=days)
    for i, (dma, d) in enumerate(zip(r["dma_id"].values, r["date"].values)):
        if dma not in ev.index:
            continue
        dates = ev[dma]
        hi = d if include_same_day else d - np.timedelta64(1, "D")
        lo = d - lo_off
        out[i] = np.searchsorted(dates, hi, side="right") - np.searchsorted(dates, lo, side="left")
    return out
