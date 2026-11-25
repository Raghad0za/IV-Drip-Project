"""
backends.py: loads Jana's and Hasan's code and runs it on the uploaded files.

  drop_detection/rate.py      -> measure_rate(video, drop_factor)          (Jana)
  bag_label/iv_monitor.py     -> bag_level(image), check_label(image, drug) (Hasan)

If a model file or a library is missing, that part falls back so the demo still runs:
  rate : drops.pt missing      -> rate.py's own motion method (no model needed)
         rate.py can't import  -> DEMO values
  bag/label: best.pt / ultralytics / easyocr missing -> DEMO values
Every result says which mode produced it, and the app shows it.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DROP_DIR = ROOT / "drop_detection"
BAG_DIR = ROOT / "bag_label"
for p in (DROP_DIR, BAG_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _try_import(name):
    try:
        return __import__(name), None
    except Exception as e:                      # missing library, syntax error, ...
        return None, f"{type(e).__name__}: {e}"


rate_mod, RATE_ERR = _try_import("rate")
bag_mod, BAG_ERR = _try_import("iv_monitor")


def status():
    """What is available, for the sidebar."""
    drops_pt = DROP_DIR / "drops.pt"
    best_pt = BAG_DIR / "best.pt"
    libs_bag = all(_has(m) for m in ("ultralytics", "easyocr", "rapidfuzz"))
    if rate_mod is None:
        rate_mode = "demo"
    elif drops_pt.exists() and _has("ultralytics"):
        rate_mode = "yolo"
    else:
        rate_mode = "motion"
    bag_mode = "real" if (bag_mod is not None and best_pt.exists() and libs_bag) else "demo"
    return {"rate": rate_mode, "bag": bag_mode, "drops_pt": drops_pt.exists(), "best_pt": best_pt.exists(),
            "rate_error": RATE_ERR, "bag_error": BAG_ERR}


def _has(module):
    try:
        __import__(module)
        return True
    except Exception:
        return False


# --------------------------------------------------------------- drip rate (Jana)
def run_rate(video_path, drop_factor):
    mode = status()["rate"]
    if mode == "demo":
        return {"dpm": 33.0, "ml_h": round(33 * 60 / drop_factor, 1), "drops": 17, "status": "ok",
                "method": "demo", "events": []}
    method = "auto" if mode == "yolo" else "motion"
    out = rate_mod.measure_rate(str(video_path), drop_factor=drop_factor, method=method, debug=True)
    out.setdefault("method", method)
    return out


# --------------------------------------------------------------- bag level + label (Hasan)
def run_bag(image):
    if status()["bag"] == "demo":
        return {"level": "half", "level_ar": "نص", "fill_ratio": 0.52, "mode": "demo"}
    out = bag_mod.bag_level(image)
    out["mode"] = "real"
    return out


def run_label(image, drug):
    if status()["bag"] == "demo":
        return {"text": "Sodium Chloride 0.9% 500 mL IV Infusion", "match": _demo_match(drug), "score": 100,
                "matched_name": "Sodium Chloride 0.9%", "volume_ml": 500, "mode": "demo"}
    out = bag_mod.check_label(image, drug)
    out["mode"] = "real"
    return out


def _demo_match(drug):
    """Demo label reads Normal Saline: match only if that is what was prescribed."""
    names = ["normal saline 0.9%", "sodium chloride 0.9%", "nacl 0.9%", "محلول ملحي 0.9%", "كلوريد الصوديوم 0.9%"]
    return drug.strip().lower() in names


def analyze_bed(video_path, bag_image, label_image, drug, rx_mlh, drop_factor):
    """Run the three checks for one bed. Images are OpenCV (BGR) arrays or paths."""
    rate = run_rate(video_path, drop_factor)
    bag = run_bag(bag_image)
    label = run_label(label_image, drug)
    return {"inputs": {"drug": drug, "rx_mlh": rx_mlh, "drop_factor": drop_factor},
            "rate": rate, "bag": bag, "label": label}
