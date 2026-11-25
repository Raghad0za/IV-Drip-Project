"""
iv_logic.py: turns the raw model outputs into what the nurse sees.

Pure logic, no models and no Streamlit, so it can be tested on its own.
The rules are the same as in the agreed mockup:

  status "done"  : no drops and the bag is under 10%            (the infusion finished)
  status "bad"   : label does not match, rate off by more than 25%, or no drops
  status "warn"  : rate off by more than 10%, bag almost empty / empty, or less than 1 hour left
  status "ok"    : everything within the prescription
"""
from __future__ import annotations

FULL_FILL = 0.72          # fill_ratio of a completely full bag (same as iv_monitor.FULL_FILL)
LEVELS = [(0.65, "full", "مليان"), (0.40, "half", "نص"), (0.10, "almost empty", "قرب يخلص"), (0.0, "empty", "فاضي")]
WARN_DEV, BAD_DEV = 10, 25  # % deviation from the prescribed rate

TONES = {
    "ok":   {"bg": "#E7F5EC", "fg": "#14653A", "border": "#BFE3CC", "bar": "#1F8A4C", "label": "طبيعي"},
    "warn": {"bg": "#FFF3D6", "fg": "#7A4F00", "border": "#F2D58A", "bar": "#C98A00", "label": "تنبيه"},
    "bad":  {"bg": "#FDEBEA", "fg": "#A11D1D", "border": "#F3B9B5", "bar": "#C0392B", "label": "حرج"},
    "done": {"bg": "#E6EEF7", "fg": "#24476B", "border": "#C3D4E6", "bar": "#5B7FA6", "label": "انتهى المحلول"},
    "idle": {"bg": "#EEF1F4", "fg": "#4A5868", "border": "#E2E7ED", "bar": "#9AA8B8", "label": "بانتظار التحليل"},
}


def level_of(fill):
    """fill_ratio -> (english, arabic) level name."""
    if fill is None:
        return "no bag found", "ما انلقى الكيس"
    for threshold, en, ar in LEVELS:
        if fill >= threshold:
            return en, ar
    return "empty", "فاضي"


def remaining_ml(fill, volume_ml):
    if fill is None or not volume_ml:
        return None
    return round(min(fill / FULL_FILL, 1.0) * volume_ml)


def format_time_left(hours):
    if hours is None:
        return "—"
    if hours < 1:
        return f"~{round(hours * 60)} د"
    return f"~{hours:.1f} س"


def evaluate(rate, bag, label, rx_mlh, drug):
    """
    rate  : measure_rate output  {"dpm", "ml_h", "drops", "status", ...}
    bag   : bag_level output     {"level", "level_ar", "fill_ratio"}
    label : check_label output   {"text", "match", "score", "matched_name", "volume_ml"}
    rx_mlh: prescribed rate in mL/h
    drug  : prescribed drug name
    Returns one flat dict with everything the dashboard shows.
    """
    dpm = float(rate.get("dpm") or 0)
    mlh = float(rate.get("ml_h") or 0)
    fill = bag.get("fill_ratio")
    level_en, level_ar = level_of(fill)
    label_ok = bool(label.get("match"))
    volume = label.get("volume_ml")
    ml_left = remaining_ml(fill, volume)
    share = None if fill is None else min(fill / FULL_FILL, 1.0)
    hours = ml_left / mlh if (ml_left is not None and mlh > 0) else None

    out = {
        "drug": drug, "rx_mlh": rx_mlh, "dpm": round(dpm, 1), "ml_h": round(mlh), "drops": rate.get("drops", 0),
        "rate_method": rate.get("method"),
        "label_ok": label_ok, "label_score": label.get("score", 0), "label_text": label.get("text", ""),
        "label_read": label.get("matched_name") if label_ok else _short(label.get("text", "")),
        "volume_ml": volume, "fill": fill, "level_en": level_en, "level_ar": level_ar,
        "pct": None if share is None else round(share * 100),
        "ml_left": ml_left, "hours_left": hours, "time_left": format_time_left(hours),
    }

    # finished infusion
    if dpm == 0 and fill is not None and fill < 0.10:
        out.update(status="done", reason="المحلول خلص", dev_pct=None, dev_text="—", dev_status="idle",
                   bag_status="done", time_left="خلص")
        return out

    dev = (mlh - rx_mlh) / rx_mlh * 100 if rx_mlh else 0.0
    ad = abs(dev)
    low = level_en in ("almost empty", "empty") or (hours is not None and hours < 1)
    no_bag = fill is None

    if not label_ok or ad > BAD_DEV or dpm == 0:
        status = "bad"
    elif ad > WARN_DEV or low or no_bag:
        status = "warn"
    else:
        status = "ok"

    if not label_ok:
        reason = "الملصق لا يطابق الوصفة"
    elif dpm == 0:
        reason = "ما فيه قطرات: المحلول متوقف"
    elif ad > BAD_DEV:
        reason = "السرعة منحرفة عن الوصفة"
    elif ad > WARN_DEV:
        reason = "السرعة تحتاج ضبط"
    elif no_bag:
        reason = "ما انلقى الكيس في الصورة"
    elif low:
        reason = "الكيس قرّب يخلص"
    else:
        reason = "ضمن الوصفة"

    out.update(
        status=status, reason=reason, dev_pct=round(dev),
        dev_text=("+" if dev >= 0 else "−") + f"{abs(dev):.0f}%",
        dev_status="bad" if ad > BAD_DEV else "warn" if ad > WARN_DEV else "ok",
        bag_status="warn" if (low or no_bag) else "ok",
    )
    return out


def _short(text, n=40):
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n] + "…"


# --------------------------------------------------------------- example beds (same as the mockup)
# rate / bag / label are written in the same shape the real functions return.
def _ex(drug, read, dpm, rx, vol, fill, match, factor=20):
    return {
        "inputs": {"drug": drug, "rx_mlh": rx, "drop_factor": factor},
        "rate": {"dpm": dpm, "ml_h": round(dpm * 60 / factor, 1), "drops": 0 if dpm == 0 else 15,
                 "status": "ok" if dpm else "no_drops", "method": "example"},
        "bag": {"fill_ratio": fill},
        "label": {"text": read, "match": match, "score": 100 if match else 42,
                  "matched_name": read if match else None, "volume_ml": vol},
    }


EXAMPLE_BEDS = {
    1: _ex("Normal Saline 0.9%", "Sodium Chloride 0.9%", 32, 100, 1000, 0.45, True),
    2: _ex("Dextrose 5%", "Dextrose 5%", 27, 80, 500, 0.12, True),
    3: _ex("Ringer Lactate", "Dextrose 5%", 25, 75, 500, 0.60, False),
    4: _ex("Normal Saline 0.9%", "Normal Saline 0.9%", 35, 75, None, 0.70, True),
    6: _ex("Normal Saline 0.9%", "NaCl 0.9%", 0, 100, 500, 0.03, True),
}


def evaluate_saved(bed):
    """Evaluate a stored bed {"inputs", "rate", "bag", "label"}."""
    i = bed["inputs"]
    return evaluate(bed["rate"], bed["bag"], bed["label"], i["rx_mlh"], i["drug"])
