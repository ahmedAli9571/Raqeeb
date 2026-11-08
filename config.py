"""
Raqeeb settings. This is the ONE file to edit for limits, model settings and actions.
"""

# ----------------------------------------------------------------------------
# 1. Water quality limits
#    PLACEHOLDERS. Replace with the exact edition of the Omani drinking-water
#    standard your lab uses, then re-run new_data.bat (or retrain in the app).
# ----------------------------------------------------------------------------
LIMITS = {
    "free_chlorine_min_mgL": 0.2,     # minimum free chlorine at the sampling point
    "free_chlorine_max_mgL": 5.0,     # maximum free chlorine
    "turbidity_max_NTU": 1.0,         # maximum turbidity
    "ph_min": 6.5,
    "ph_max": 8.5,
    "ecoli_max_cfu100mL": 0,          # E. coli must be absent in 100 mL
    "total_coliform_max_cfu100mL": 0,
}

# A passing result below this multiple of the chlorine minimum counts as a "near miss".
NEAR_MISS_FACTOR = 1.5

# ----------------------------------------------------------------------------
# 2. Folders
# ----------------------------------------------------------------------------
DATA_DIR = "data"
MODEL_DIR = "models"

# ----------------------------------------------------------------------------
# 3. Model settings
# ----------------------------------------------------------------------------
TRAIN_END_MONTH = 9          # train on Jan..Sep, test on Oct..Dec (time split, never random)
REPAIR_WINDOW_DAYS = 5       # repairs in the same DMA within this many days before a sample
COMPLAINT_WINDOW_DAYS = 2    # quality complaints in the same DMA within this many days
QC_TOLERANCE_PCT = 5         # CRM recovery outside 100 +/- this = QC out of control
RANDOM_SEED = 42

# ----------------------------------------------------------------------------
# 4. Recommended actions per likely cause (shown in the inbox and in the notes)
# ----------------------------------------------------------------------------
ACTIONS = {
    "real_event": {
        "en": "Treat as a real network event: notify the supervisor, resample with flushing, "
              "check recent repairs in the DMA, and follow the incident SOP.",
        "ar": "يُعامل كحدث حقيقي في الشبكة: إبلاغ المشرف، وإعادة أخذ العينة بعد التصريف، "
              "ومراجعة أعمال الإصلاح الأخيرة في المنطقة، واتباع إجراءات الحوادث المعتمدة.",
    },
    "sampling_error": {
        "en": "Likely a sampling problem: resample the same tap after flushing and flaming, "
              "with a new sterile bottle, and review the sampling technique.",
        "ar": "يُرجّح خطأ في أخذ العينة: إعادة أخذ العينة من نفس الصنبور بعد التصريف والتعقيم، "
              "بقارورة معقمة جديدة، ومراجعة طريقة أخذ العينات.",
    },
    "analytical_error": {
        "en": "Likely a lab or instrument problem: re-run the batch QC and reference material, "
              "recalibrate, and re-analyse the retained sample before sending a field team.",
        "ar": "يُرجّح خطأ تحليلي في المختبر: إعادة فحص ضبط الجودة والمادة المرجعية للدفعة، "
              "وإعادة المعايرة، وإعادة تحليل العينة المحفوظة قبل إرسال فريق ميداني.",
    },
}

# Safety rule that always wins over the model: these results are URGENT whatever the AI says.
SAFETY_OVERRIDE_ECOLI = True

CAUSE_LABELS = {
    "real_event": {"en": "Real network event", "ar": "حدث حقيقي في الشبكة"},
    "sampling_error": {"en": "Sampling error", "ar": "خطأ في أخذ العينة"},
    "analytical_error": {"en": "Analytical (lab) error", "ar": "خطأ تحليلي"},
}
