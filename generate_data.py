"""
Raqeeb synthetic practice dataset generator.

Creates one year of realistic, labelled drinking-water quality data for ~50 sites:
sites, daily site conditions (temperature, water age), lab results, lab QC per batch,
network repairs, and customer complaints. Every abnormal result carries a hidden
`true_cause` label (real_event / sampling_error / analytical_error / normal) so you
can train and score the triage model.

All numbers are SYNTHETIC. Limits come from config.py (edit them there); the
generator's own constants (decay, event rates) are in CONFIG below.

Run:  python generate_data.py --out data --seed 42
"""
import argparse
import os
import numpy as np
import pandas as pd

import config as raqeeb_config

CONFIG = {
    "year": 2025,
    "n_sites": 50,
    # --- Limits: taken from config.py so the generator and Raqeeb always agree ---
    "limits": raqeeb_config.LIMITS,
    # --- Chlorine decay: C = C0 * exp(-k_T * t), k_T = k20 * theta^(T-20) (assumed) ---
    "theta": 1.07,
    "k20_range_per_h": (0.002, 0.007),
    "c0_range_mgL": (0.8, 1.2),
    # --- Event rates ---
    "repairs_per_site_year": 1.2,
    "intrusion_prob_after_repair": 0.6,
    "sampling_error_rate": 0.012,      # per sample
    "analytical_drift_batches": 14,     # batches per year with QC drift
    "complaint_base_rate_per_site_day": 0.01,
}

GOVS = ["Muscat", "Al Batinah North", "Al Batinah South", "Ad Dakhiliyah",
        "Ash Sharqiyah North", "Ash Sharqiyah South", "Ad Dhahirah", "Al Buraimi"]
GOV_CENTRES = {  # rough lat/lon, for map display only
    "Muscat": (23.59, 58.41), "Al Batinah North": (24.35, 56.73),
    "Al Batinah South": (23.68, 57.88), "Ad Dakhiliyah": (22.93, 57.53),
    "Ash Sharqiyah North": (22.68, 58.53), "Ash Sharqiyah South": (22.57, 59.53),
    "Ad Dhahirah": (23.22, 56.51), "Al Buraimi": (24.25, 55.79),
}


def make_sites(rng, n):
    rows = []
    for i in range(n):
        gov = GOVS[i % len(GOVS)]
        rural = rng.random() < 0.45
        stype = rng.choice(["reservoir", "elevated_tank", "network_tap"], p=[0.3, 0.3, 0.4])
        lat0, lon0 = GOV_CENTRES[gov]
        rows.append({
            "site_id": f"S{i+1:03d}",
            "governorate": gov,
            "dma_id": f"DMA-{gov[:3].upper()}-{(i // 3) + 1:02d}",
            "site_type": stype,
            "zone": "rural" if rural else "urban",
            "source": rng.choice(["desalinated", "well", "blend"], p=[0.6, 0.2, 0.2]),
            # rural, low-demand sites hold water longer
            "base_water_age_h": round(float(rng.uniform(30, 90) if rural else rng.uniform(8, 30)), 1),
            "k20_per_h": round(float(rng.uniform(*CONFIG["k20_range_per_h"])), 4),
            "c0_mgL": round(float(rng.uniform(*CONFIG["c0_range_mgL"])), 2),
            "lat": round(lat0 + rng.normal(0, 0.12), 4),
            "lon": round(lon0 + rng.normal(0, 0.12), 4),
        })
    return pd.DataFrame(rows)


def water_temp(day_of_year, rng):
    # Oman water temperature: ~22 C in Jan, ~36 C in Jul-Aug
    return 29 + 7 * np.sin(2 * np.pi * (day_of_year - 110) / 365) + rng.normal(0, 0.8)


def make_daily(rng, sites, dates):
    rows = []
    for _, s in sites.iterrows():
        for d in dates:
            doy = d.dayofyear
            t = water_temp(doy, rng)
            # summer demand up -> water age down a bit, but heat dominates decay
            age = s.base_water_age_h * (1 - 0.12 * np.sin(2 * np.pi * (doy - 110) / 365)) * rng.lognormal(0, 0.12)
            k = s.k20_per_h * CONFIG["theta"] ** (t - 20)
            cl = s.c0_mgL * np.exp(-k * age)
            rows.append({"date": d, "site_id": s.site_id, "water_temp_C": round(t, 1),
                         "water_age_h": round(age, 1), "expected_free_cl_mgL": round(cl, 3)})
    return pd.DataFrame(rows)


def main(out, seed):
    rng = np.random.default_rng(seed)
    os.makedirs(out, exist_ok=True)
    L = CONFIG["limits"]
    dates = pd.date_range(f"{CONFIG['year']}-01-01", f"{CONFIG['year']}-12-31", freq="D")

    sites = make_sites(rng, CONFIG["n_sites"])
    daily = make_daily(rng, sites, dates)
    dkey = daily.set_index(["site_id", "date"])

    # ---------- repairs (CMMS) and intrusion windows ----------
    repairs, intrusion = [], {}
    for _, s in sites.iterrows():
        for _ in range(rng.poisson(CONFIG["repairs_per_site_year"])):
            d = dates[rng.integers(0, len(dates))]
            rid = f"R{len(repairs)+1:04d}"
            repairs.append({"repair_id": rid, "date": d, "dma_id": s.dma_id, "site_id_nearest": s.site_id,
                            "repair_type": rng.choice(["burst_main", "service_leak", "valve_replacement"], p=[0.4, 0.45, 0.15]),
                            "pipe_diameter_mm": int(rng.choice([63, 110, 160, 225, 315]))})
            if rng.random() < CONFIG["intrusion_prob_after_repair"]:
                for lag in range(1, 6):  # contamination window 1-5 days after repair
                    intrusion[(s.site_id, d + pd.Timedelta(days=lag))] = rid
    repairs = pd.DataFrame(repairs)

    # ---------- sampling schedule: each site ~weekly, plus resamples later ----------
    samples = []
    for _, s in sites.iterrows():
        offset = int(rng.integers(0, 7))
        for d in dates[offset::7]:
            samples.append((s.site_id, d + pd.Timedelta(days=int(rng.integers(-1, 2)))))
        # extra sampling inside intrusion windows (routine schedule catches some, not all)
    samples = [(sid, d) for sid, d in samples if d.year == CONFIG["year"]]

    # ---------- lab batches: one batch per lab day ----------
    sample_df = pd.DataFrame(samples, columns=["site_id", "date"]).drop_duplicates()
    sample_df["batch_id"] = "B" + sample_df["date"].dt.strftime("%Y%m%d")
    batch_ids = sorted(sample_df["batch_id"].unique())
    # spread lab-drift batches across the four quarters so every period has some
    quarters = {q: [b for b in batch_ids if (int(b[5:7]) - 1) // 3 == q] for q in range(4)}
    per_q = max(1, CONFIG["analytical_drift_batches"] // 4)
    drift_batches = set()
    for q, bs in quarters.items():
        drift_batches |= set(rng.choice(bs, min(per_q, len(bs)), replace=False))
    drift_param = {b: rng.choice(["turbidity", "free_chlorine"]) for b in sorted(drift_batches)}

    qc_rows = []
    for b in batch_ids:
        for param in ["free_chlorine", "turbidity", "ph"]:
            rec = rng.normal(100, 2.0)
            if b in drift_batches and drift_param[b] == param:
                # chlorine faults read LOW (e.g. expired DPD reagent); turbidity faults read HIGH
                # (e.g. dirty cuvette / drifting calibration)
                rec = rng.uniform(55, 80) if param == "free_chlorine" else rng.uniform(115, 140)
            qc_rows.append({"batch_id": b, "parameter": param, "crm_recovery_pct": round(rec, 1),
                            "blank_ok": bool(rng.random() > 0.01),
                            "days_since_calibration": int(rng.integers(0, 30)),
                            "qc_in_control": bool(95 <= rec <= 105)})
    qc = pd.DataFrame(qc_rows)
    qc_rec = qc.set_index(["batch_id", "parameter"])["crm_recovery_pct"]

    # ---------- results ----------
    res = []
    for i, r in sample_df.reset_index(drop=True).iterrows():
        sid, d, b = r.site_id, r.date, r.batch_id
        dd = dkey.loc[(sid, d)]
        cause = "normal"
        cl = max(0.0, dd.expected_free_cl_mgL * rng.lognormal(0, 0.10))
        turb = rng.lognormal(np.log(0.25), 0.35)
        ph = rng.normal(7.6, 0.12)
        ec, tc = 0, 0

        if (sid, d) in intrusion:                      # real event: intrusion after repair
            cause = "real_event"
            cl *= rng.uniform(0.05, 0.4)
            turb *= rng.uniform(3, 10)
            tc = int(rng.integers(3, 60))
            ec = int(rng.integers(0, 15))
        elif cl < L["free_chlorine_min_mgL"]:           # real event: heat + water age decay
            cause = "real_event"
        elif rng.random() < CONFIG["sampling_error_rate"]:  # sampling error: isolated micro hit
            cause = "sampling_error"
            tc = int(rng.integers(1, 8))
            ec = int(rng.random() < 0.3)

        # analytical drift applies to the measured value, whatever the true state
        if b in drift_batches:
            p = drift_param[b]
            f = qc_rec[(b, p)] / 100
            if p == "turbidity":
                turb *= f ** 6     # exaggerated so the fault pushes several samples over the limit
            else:
                cl *= f ** 3
            if cause == "normal":
                cause = "analytical_error_candidate"

        row = {"sample_id": f"WQ{i+1:05d}", "site_id": sid, "date": d, "batch_id": b,
               "free_cl_mgL": round(cl, 2), "turbidity_NTU": round(turb, 2), "ph": round(ph, 2),
               "total_coliform_cfu100mL": tc, "ecoli_cfu100mL": ec}
        abnormal = (row["free_cl_mgL"] < L["free_chlorine_min_mgL"] or row["free_cl_mgL"] > L["free_chlorine_max_mgL"]
                    or row["turbidity_NTU"] > L["turbidity_max_NTU"]
                    or not (L["ph_min"] <= row["ph"] <= L["ph_max"])
                    or tc > L["total_coliform_max_cfu100mL"] or ec > L["ecoli_max_cfu100mL"])
        if cause == "analytical_error_candidate":
            cause = "analytical_error" if abnormal else "normal"
        if not abnormal and cause != "real_event":
            cause = "normal"
        row["is_abnormal"] = bool(abnormal)
        near = (not abnormal) and row["free_cl_mgL"] < 1.5 * L["free_chlorine_min_mgL"]
        row["true_cause"] = cause if abnormal else ("near_miss" if near else "normal")
        res.append(row)
    results = pd.DataFrame(res)

    # ---------- complaints (1442) ----------
    comp = []
    site_dma = sites.set_index("site_id")["dma_id"]
    for _, s in sites.iterrows():
        for d in dates:
            lam = CONFIG["complaint_base_rate_per_site_day"]
            if (s.site_id, d) in intrusion:
                lam += 0.8
            for _ in range(rng.poisson(lam)):
                comp.append({"date": d, "dma_id": s.dma_id,
                             "category": rng.choice(["taste_odour", "discoloured", "low_pressure", "no_water", "billing"],
                                                    p=[0.3, 0.25, 0.2, 0.1, 0.15] if (s.site_id, d) in intrusion
                                                    else [0.1, 0.1, 0.3, 0.15, 0.35])})
    complaints = pd.DataFrame(comp)
    complaints.insert(0, "complaint_id", [f"C{i+1:05d}" for i in range(len(complaints))])

    # ---------- write ----------
    sites.to_csv(f"{out}/sites.csv", index=False)
    daily.to_csv(f"{out}/site_daily_conditions.csv", index=False)
    results.to_csv(f"{out}/lab_results.csv", index=False)
    qc.to_csv(f"{out}/lab_qc.csv", index=False)
    repairs.to_csv(f"{out}/repairs_cmms.csv", index=False)
    complaints.to_csv(f"{out}/complaints_1442.csv", index=False)
    pd.Series(CONFIG["limits"]).to_csv(f"{out}/limits_config.csv", header=["value"])

    print("sites", len(sites), "| daily rows", len(daily), "| results", len(results),
          "| QC rows", len(qc), "| repairs", len(repairs), "| complaints", len(complaints))
    print(results["true_cause"].value_counts().to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    main(a.out, a.seed)
