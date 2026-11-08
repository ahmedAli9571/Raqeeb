"""
Column names map. On hackathon Day 2, if Nama's files use different column names,
change ONLY the right-hand side (their name). The rest of Raqeeb stays the same.

Format:  "raqeeb_name": "name_in_your_file"
"""

FILES = {
    "sites": "sites.csv",
    "daily": "site_daily_conditions.csv",
    "results": "lab_results.csv",
    "qc": "lab_qc.csv",
    "repairs": "repairs_cmms.csv",
    "complaints": "complaints_1442.csv",
}

COLUMNS = {
    "sites": {
        "site_id": "site_id", "governorate": "governorate", "dma_id": "dma_id",
        "site_type": "site_type", "zone": "zone", "source": "source",
        "lat": "lat", "lon": "lon",
        "base_water_age_h": "base_water_age_h",   # design residence time; leave blank if unknown
    },
    "daily": {
        "date": "date", "site_id": "site_id",
        "water_temp_C": "water_temp_C", "water_age_h": "water_age_h",
    },
    "results": {
        "sample_id": "sample_id", "site_id": "site_id", "date": "date", "batch_id": "batch_id",
        "free_cl_mgL": "free_cl_mgL", "turbidity_NTU": "turbidity_NTU", "ph": "ph",
        "total_coliform_cfu100mL": "total_coliform_cfu100mL", "ecoli_cfu100mL": "ecoli_cfu100mL",
        # answer key: exists only in practice data, never in real data
        "true_cause": "true_cause",
    },
    "qc": {
        "batch_id": "batch_id", "parameter": "parameter",
        "crm_recovery_pct": "crm_recovery_pct", "days_since_calibration": "days_since_calibration",
    },
    "repairs": {"repair_id": "repair_id", "date": "date", "dma_id": "dma_id"},
    "complaints": {"complaint_id": "complaint_id", "date": "date", "dma_id": "dma_id", "category": "category"},
}
