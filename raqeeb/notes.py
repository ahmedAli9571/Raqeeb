"""Bilingual investigation note built from a fixed template (works offline, no API key)."""
import config


def note(x):
    L = config.LIMITS
    c = x["pred_cause"]
    lab_en = config.CAUSE_LABELS[c]["en"]
    lab_ar = config.CAUSE_LABELS[c]["ar"]
    conf = round(100 * x["confidence"])
    date = x["date"].strftime("%Y-%m-%d")
    reasons_en = "; ".join(x["reasons"]) or "no single strong signal"

    en = (
        f"Investigation note: sample {x['sample_id']}, site {x['site_id']} ({x['governorate']}, {x['dma_id']}), {date}.\n"
        f"Result: {x['fail_reasons']}. Free chlorine {x['free_cl_mgL']:.2f} mg/L "
        f"(expected about {x['cl_expected']:.2f} for {x['water_temp_C']:.0f} °C and {x['water_age_h']:.0f} h water age; "
        f"limit {L['free_chlorine_min_mgL']}). Turbidity {x['turbidity_NTU']:.2f} NTU. "
        f"Total coliform {int(x['total_coliform'])}, E. coli {int(x['ecoli'])} per 100 mL.\n"
        f"Likely cause: {lab_en} ({conf}% confidence). Main evidence: {reasons_en}.\n"
        f"Context: {int(x['repairs_recent'])} repair(s) in the DMA in the last {config.REPAIR_WINDOW_DAYS} days; "
        f"{int(x['quality_complaints_recent'])} quality complaint(s) in the last {config.COMPLAINT_WINDOW_DAYS} days; "
        f"batch QC {'OUT of control' if x['qc_out_of_control'] else 'in control'}.\n"
        f"Recommended action: {config.ACTIONS[c]['en']}\n"
        f"This result stays open until confirmed under the lab SOP. Raqeeb orders the work; the chemist decides."
    )
    ar = (
        f"مذكرة تحقيق: العينة {x['sample_id']}، الموقع {x['site_id']} ({x['dma_id']})، بتاريخ {date}.\n"
        f"النتيجة: الكلور الحر {x['free_cl_mgL']:.2f} ملغم/لتر (المتوقع نحو {x['cl_expected']:.2f})، "
        f"العكارة {x['turbidity_NTU']:.2f} NTU، القولونيات الكلية {int(x['total_coliform'])}، "
        f"الإشريكية القولونية {int(x['ecoli'])} لكل 100 مل.\n"
        f"السبب المرجّح: {lab_ar} (بثقة {conf}%).\n"
        f"السياق: {int(x['repairs_recent'])} إصلاح في المنطقة خلال آخر {config.REPAIR_WINDOW_DAYS} أيام، "
        f"و{int(x['quality_complaints_recent'])} بلاغ عن الجودة خلال آخر {config.COMPLAINT_WINDOW_DAYS} يوم، "
        f"وضبط جودة الدفعة {'خارج الحدود' if x['qc_out_of_control'] else 'ضمن الحدود'}.\n"
        f"الإجراء الموصى به: {config.ACTIONS[c]['ar']}\n"
        f"تبقى النتيجة مفتوحة حتى تأكيدها وفق إجراءات المختبر المعتمدة؛ رقيب يرتّب العمل والقرار للكيميائي."
    )
    return en, ar
