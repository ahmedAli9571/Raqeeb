"""
Train Raqeeb and check it is not just copying the data generator.

    python train.py              # train on data/, save to models/
    python train.py --no-check   # skip the robustness check
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import config
from raqeeb.data import load_all
from raqeeb.features import add_cause_features, site_context, site_medians
from raqeeb.models import CLASSES, evaluate, train


def robustness_check(expect, cause, meta):
    """Make a NEW dataset with a different seed and score the already-trained model on it."""
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, "generate_data.py", "--out", tmp, "--seed", "7"],
                       check=True, capture_output=True)
        t2 = load_all(tmp)
        med = site_medians(t2["results"], config.TRAIN_END_MONTH)  # the new sites' own history
        f = add_cause_features(site_context(t2, med), t2, expect, meta["resid_sd"])
        lab = f[f["is_abnormal"] & f["true_cause"].isin(CLASSES)]
        return evaluate(cause, lab)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-check", action="store_true")
    a = ap.parse_args()

    t = load_all()
    expect, cause, meta = train(t)
    print(f"Expectation model: trained on {meta['n_expect_train']} passing results; "
          f"typical error +/- {meta['resid_sd']:.3f} mg/L")
    print(f"Cause classifier: {meta['n_cause_train']} training cases {meta['train_counts']}")
    te = meta["test"]
    if te:
        print(f"Test months (Oct-Dec): accuracy {te['accuracy']:.0%} on {te['n']} abnormal results")
        for c in te["labels"]:
            m = te["report"][c]
            print(f"  {c:17s} precision {m['precision']:.0%}  recall {m['recall']:.0%}  (n={int(m['support'])})")

    if not a.no_check and meta["n_cause_train"] > 0:
        rb = robustness_check(expect, cause, meta)
        meta["robustness_seed7"] = rb
        with open(os.path.join(config.MODEL_DIR, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, default=str)
        print(f"Robustness (new dataset, seed 7): accuracy {rb['accuracy']:.0%} on {rb['n']} abnormal results")


if __name__ == "__main__":
    main()
