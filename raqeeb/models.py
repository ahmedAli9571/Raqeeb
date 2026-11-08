"""Train, save, load and apply Raqeeb's two models."""
import json
import math
import os

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_predict

import config
from raqeeb.features import (CAUSE_FEATURES, EXPECT_FEATURES, add_cause_features, add_site_columns,
                             site_context, site_medians)

CLASSES = ["real_event", "sampling_error", "analytical_error"]

READABLE = {
    "cl_surprise": "chlorine far from this site's usual value",
    "cl_measured": "measured chlorine level",
    "cl_expected": "expected chlorine for today's heat and water age",
    "log_turbidity": "turbidity level",
    "total_coliform": "total coliform count",
    "ecoli": "E. coli count",
    "micro_only": "bacteria found while chlorine and turbidity look normal",
    "qc_cl_dev": "chlorine reference material off in this batch",
    "qc_turb_dev": "turbidity reference material off in this batch",
    "qc_out_of_control": "batch QC out of control",
    "batch_abnormal_share": "many abnormal results in the same batch",
    "repairs_recent": "recent pipe repair in this DMA",
    "quality_complaints_recent": "taste/odour/colour complaints in this DMA",
    "water_temp_C": "water temperature",
    "water_age_h": "water age in the tank",
    "rural": "rural, low-demand zone",
}


def explain(feature, value):
    """Turn one piece of evidence into a sentence that states its actual value."""
    v = float(value)
    if feature == "repairs_recent":
        return f"{int(v)} pipe repair(s) in this DMA in the last {config.REPAIR_WINDOW_DAYS} days" if v else \
            f"no pipe repair in this DMA in the last {config.REPAIR_WINDOW_DAYS} days"
    if feature == "quality_complaints_recent":
        return f"{int(v)} taste/odour/colour complaint(s) in this DMA" if v else "no taste/odour/colour complaints in this DMA"
    if feature == "qc_out_of_control":
        return "batch QC out of control" if v else "batch QC in control"
    if feature == "micro_only":
        return "bacteria found while chlorine and turbidity look normal" if v else \
            "bacteria (if any) come with chlorine or turbidity problems"
    if feature == "rural":
        return "rural, low-demand zone" if v else "urban zone"
    if feature == "cl_surprise":
        return f"chlorine {abs(v):.1f} typical errors {'below' if v < 0 else 'above'} this site's expected value"
    if feature == "batch_abnormal_share":
        return f"{v:.0%} of this lab batch is abnormal"
    if feature in ("qc_cl_dev", "qc_turb_dev"):
        what = "chlorine" if feature == "qc_cl_dev" else "turbidity"
        return f"{what} reference material off by {v:.0f}% in this batch"
    if feature == "log_turbidity":
        return f"turbidity {10 ** v:.2f} NTU"
    units = {"cl_measured": "mg/L", "cl_expected": "mg/L", "water_temp_C": "°C", "water_age_h": "h",
             "total_coliform": "/100 mL", "ecoli": "/100 mL"}
    if feature in ("total_coliform", "ecoli"):
        return f"{READABLE[feature]}: {int(v)} /100 mL"
    if feature in ("water_temp_C", "water_age_h"):
        return f"{READABLE[feature]}: {v:.0f} {units[feature]}"
    return f"{READABLE[feature]}: {v:.2f} {units.get(feature, '')}".strip()


def train(t, out_dir=None):
    out_dir = out_dir or config.MODEL_DIR
    os.makedirs(out_dir, exist_ok=True)
    medians = site_medians(t["results"], config.TRAIN_END_MONTH)
    r = site_context(t, medians)
    train_mask = r["date"].dt.month <= config.TRAIN_END_MONTH

    # 1) Site-expectation model: learns normal chlorine from results that passed.
    ok = r[train_mask & ~r["is_abnormal"]]
    expect = LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=15,
                           min_child_samples=10, random_state=config.RANDOM_SEED, verbose=-1)
    oof = cross_val_predict(expect, ok[EXPECT_FEATURES], ok["free_cl_mgL"], cv=5)
    resid_sd = float(np.std(ok["free_cl_mgL"] - oof))
    expect.fit(ok[EXPECT_FEATURES], ok["free_cl_mgL"])

    # 2) Cause classifier: learns from abnormal results whose cause is known.
    f = add_cause_features(r, t, expect, resid_sd)
    labels = f["true_cause"] if "true_cause" in f.columns else pd.Series("", index=f.index)
    lab = f[f["is_abnormal"] & labels.isin(CLASSES)]
    tr, te = lab[lab["date"].dt.month <= config.TRAIN_END_MONTH], lab[lab["date"].dt.month > config.TRAIN_END_MONTH]
    old_cause = os.path.join(out_dir, "cause.joblib")
    kept = {}
    if len(tr) < 10 and os.path.exists(old_cause):
        # Real data usually has no labelled causes yet: keep the classifier trained on practice
        # data, and only refit the site-expectation model on the new data.
        cause = joblib.load(old_cause)
        tr = te = lab.iloc[0:0]
        with open(os.path.join(out_dir, "meta.json"), encoding="utf-8") as fh:
            kept = json.load(fh)  # keep the practice-data scores for the Model report page
    else:
        cause = LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=7, min_child_samples=3,
                               class_weight="balanced", random_state=config.RANDOM_SEED, verbose=-1)
        cause.fit(tr[CAUSE_FEATURES], tr["true_cause"])

    meta = {
        "resid_sd": resid_sd,
        "site_medians": medians,
        "n_expect_train": int(len(ok)),
        "n_cause_train": int(len(tr)),
        "n_cause_test": int(len(te)),
        "train_counts": tr["true_cause"].value_counts().to_dict() if len(tr) else {},
        "test": evaluate(cause, te) if len(te) else kept.get("test", {}),
        "importance": dict(zip(CAUSE_FEATURES, map(int, cause.booster_.feature_importance("gain")))),
    }
    if not len(tr):
        meta["cause_model_note"] = "No labelled causes in this data: classifier kept from practice data."
        for key in ("robustness_seed7", "n_cause_test"):
            if key in kept:
                meta[key] = kept[key]
    joblib.dump(expect, os.path.join(out_dir, "expect.joblib"))
    joblib.dump(cause, os.path.join(out_dir, "cause.joblib"))
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)
    return expect, cause, meta


def evaluate(cause, df):
    if df.empty:
        return {}
    pred = cause.predict(df[CAUSE_FEATURES])
    labels = [c for c in CLASSES if c in set(df["true_cause"]) | set(pred)]
    rep = classification_report(df["true_cause"], pred, labels=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(df["true_cause"], pred, labels=labels)
    return {"report": rep, "labels": labels, "confusion": cm.tolist(),
            "accuracy": float((pred == df["true_cause"]).mean()), "n": int(len(df))}


def load(out_dir=None):
    out_dir = out_dir or config.MODEL_DIR
    expect = joblib.load(os.path.join(out_dir, "expect.joblib"))
    cause = joblib.load(os.path.join(out_dir, "cause.joblib"))
    with open(os.path.join(out_dir, "meta.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    return expect, cause, meta


def triage(t, expect, cause, meta):
    """Score every abnormal result: likely cause, confidence, priority, top reasons."""
    r = site_context(t, meta["site_medians"])
    f = add_cause_features(r, t, expect, meta["resid_sd"])
    ab = f[f["is_abnormal"]].copy()
    if ab.empty:
        return ab
    X = ab[CAUSE_FEATURES]
    proba = cause.predict_proba(X)
    classes = list(cause.classes_)
    for i, c in enumerate(classes):
        ab[f"p_{c}"] = proba[:, i]
    ab["pred_cause"] = [classes[i] for i in proba.argmax(1)]
    ab["confidence"] = np.minimum(proba.max(1), 0.99)  # never show 100%: it is a model, not a fact

    # Reasons: per-feature contributions (SHAP values computed by LightGBM itself)
    contrib = cause.booster_.predict(X, pred_contrib=True)
    k = len(CAUSE_FEATURES) + 1
    reasons = []
    for row, c in zip(contrib, ab["pred_cause"]):
        ci = classes.index(c)
        vals = row[ci * k: ci * k + len(CAUSE_FEATURES)]
        top = np.argsort(vals)[::-1][:3]
        reasons.append([explain(CAUSE_FEATURES[j], X.iloc[len(reasons)][CAUSE_FEATURES[j]])
                        for j in top if vals[j] > 0])
    ab["reasons"] = reasons

    def priority(x):
        if config.SAFETY_OVERRIDE_ECOLI and x["ecoli"] > 0:
            return "1 Urgent (E. coli)"
        p = x.get("p_real_event", 0)
        if p >= 0.6:
            return "2 High: likely real"
        if x["pred_cause"] == "analytical_error":
            return "3 Check lab first"
        if p >= 0.3:
            return "3 Medium"
        return "4 Resample"
    ab["priority"] = ab.apply(priority, axis=1)
    return ab.sort_values(["priority", "date"], ascending=[True, False])


def tomorrow_risk(t, expect, meta, as_of):
    """Rank sites by the chance their chlorine falls below the minimum on the next day."""
    nxt = pd.Timestamp(as_of) + pd.Timedelta(days=1)
    d = t["daily"][t["daily"]["date"] == nxt].merge(t["sites"], on="site_id")
    if d.empty:
        return d
    d = add_site_columns(d, meta["site_medians"])
    d["pred_cl_mgL"] = expect.predict(d[EXPECT_FEATURES])
    lim, sd = config.LIMITS["free_chlorine_min_mgL"], meta["resid_sd"]
    d["p_below_limit"] = [0.5 * (1 + math.erf((lim - p) / (sd * math.sqrt(2)))) for p in d["pred_cl_mgL"]]
    return d.sort_values("p_below_limit", ascending=False)
