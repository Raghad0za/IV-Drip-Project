"""
backends.py: loads Jana's and Hasan's code and runs it on the uploaded files.

  drop_detection/rate.py      -> measure_rate(video, drop_factor)          (Jana)
  bag_label/iv_monitor.py     -> bag_level(image), check_label(image, drug) (Hasan)

If a model file or a library is missing, that part falls back so the demo still runs:
  rate : drops.pt missing or fails to load -> rate.py's own motion method (no model needed)
         rate.py can't import              -> DEMO values
  bag/label: best.pt / ultralytics / easyocr missing or fail to load -> DEMO values
Every result says which mode produced it, and the app marks demo values on the card.

Settings (environment variables):
  IV_MAX_SECONDS   seconds of video analysed (default 30, as in rate.py). Use 15 on a slow laptop.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DROP_DIR = ROOT / "drop_detection"
BAG_DIR = ROOT / "bag_label"
# Added at the END of the search path: bag_label/ has its own copies of app.py, backends.py and
# iv_logic.py, and those must not replace the app's files (Python takes the first match it finds).
for p in (DROP_DIR, BAG_DIR):
    if str(p) not in sys.path:
        sys.path.append(str(p))

# EasyOCR downloads its models the first time it runs. Python from python.org on a Mac has no
# certificate list, so that download fails with CERTIFICATE_VERIFY_FAILED: use certifi's list instead
# (certifi is installed with ultralytics / requests).
if sys.platform == "darwin":
    try:
        import certifi
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
        os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
    except ImportError:
        pass

MAX_SECONDS = float(os.environ.get("IV_MAX_SECONDS", 30))
MAX_IMAGE_SIDE = 1280   # phone photos (4000 px) are shrunk first: OCR is ~3x faster and reads labels just as well


def _try_import(name):
    try:
        return __import__(name), None
    except Exception as e:                      # missing library, syntax error, ...
        return None, f"{type(e).__name__}: {e}"


rate_mod, RATE_ERR = _try_import("rate")
bag_mod, BAG_ERR = _try_import("iv_monitor")
drugs_mod, _ = _try_import("drugs")             # no dependencies: used for label matching in any mode

_DISABLED: dict[str, str] = {}                  # parts whose model failed to load in warmup() -> error


def _has(module):
    try:
        __import__(module)
        return True
    except Exception:
        return False


def status():
    """What is available, for the sidebar."""
    drops_pt = DROP_DIR / "drops.pt"
    best_pt = BAG_DIR / "best.pt"
    libs_bag = all(_has(m) for m in ("ultralytics", "easyocr", "rapidfuzz"))
    if rate_mod is None:
        rate_mode = "demo"
    elif drops_pt.exists() and _has("ultralytics") and "yolo" not in _DISABLED:
        rate_mode = "yolo"
    else:
        rate_mode = "motion"
    bag_mode = "real" if (bag_mod is not None and best_pt.exists() and libs_bag and "bag" not in _DISABLED) else "demo"
    return {"rate": rate_mode, "bag": bag_mode, "drops_pt": drops_pt.exists(), "best_pt": best_pt.exists(),
            "rate_error": RATE_ERR, "bag_error": BAG_ERR, "disabled": dict(_DISABLED)}


def warmup():
    """Load every model once at start-up, so the first analysis is not slow and a model that cannot
    load (corrupt file, EasyOCR download blocked) is found before the demo, not during it."""
    s = status()
    if s["rate"] == "yolo":
        try:
            rate_mod._model()
        except Exception as e:
            _DISABLED["yolo"] = f"{type(e).__name__}: {e}"
    if s["bag"] == "real":
        try:
            bag_mod._get_model()
            bag_mod._get_readers()
        except Exception as e:
            _DISABLED["bag"] = f"{type(e).__name__}: {e}"
    return status()


def missing():
    """What is missing, in plain words: {"rate": [...], "bag": [...]}. Empty lists = everything is real."""
    out = {"rate": [], "bag": []}
    if rate_mod is None:
        out["rate"].append(f"rate.py ما اشتغل ({RATE_ERR})")
    else:
        if not (DROP_DIR / "drops.pt").exists():
            out["rate"].append("ملف drops.pt مو موجود في مجلد drop_detection/")
        if not _has("ultralytics"):
            out["rate"].append("مكتبة ultralytics مو منزّلة")
        if "yolo" in _DISABLED:
            out["rate"].append(f"drops.pt ما انحمّل ({_DISABLED['yolo']})")
    if bag_mod is None:
        out["bag"].append(f"iv_monitor.py ما اشتغل ({BAG_ERR})")
    if not (BAG_DIR / "best.pt").exists():
        out["bag"].append("ملف best.pt مو موجود في مجلد bag_label/")
    for lib in ("ultralytics", "easyocr", "rapidfuzz"):
        if not _has(lib):
            out["bag"].append(f"مكتبة {lib} مو منزّلة")
    if "bag" in _DISABLED:
        err = _DISABLED["bag"]
        if "CERTIFICATE_VERIFY_FAILED" in err:
            out["bag"].append("EasyOCR ما قدر ينزّل موديلاته (شهادات SSL في الماك): شغّلي python3 -m pip install --upgrade certifi ثم أعيدي تشغيل التطبيق")
        else:
            out["bag"].append(f"موديل الكيس أو EasyOCR ما انحمّل ({err})")
    return out


# --------------------------------------------------------------- drip rate (Jana)
def run_rate(video_path, drop_factor):
    mode = status()["rate"]
    if mode == "demo":
        return {"dpm": 33.0, "ml_h": round(33 * 60 / drop_factor, 1), "drops": 17, "status": "ok",
                "method": "demo", "events": []}
    kw = dict(drop_factor=drop_factor, debug=True, max_seconds=MAX_SECONDS)
    if mode == "yolo":
        try:
            return rate_mod.measure_rate(str(video_path), method="auto", **kw)
        except FileNotFoundError:
            raise                                   # unreadable video: the motion method would fail too
        except Exception:
            pass                                    # model error on this video: count with motion instead
    return rate_mod.measure_rate(str(video_path), method="motion", **kw)


# --------------------------------------------------------------- bag level + label (Hasan)
DEMO_LABEL_TEXT = "Sodium Chloride 0.9% 500 mL IV Infusion"


def _shrink(image, side=MAX_IMAGE_SIDE):
    """Large photo (OpenCV array) -> longest side `side` px. Paths and small images pass through."""
    if not hasattr(image, "shape") or max(image.shape[:2]) <= side:
        return image
    import cv2
    s = side / max(image.shape[:2])
    return cv2.resize(image, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


def run_bag(image):
    image = _shrink(image)
    if status()["bag"] == "demo":
        return {"level": "half", "level_ar": "نص", "fill_ratio": 0.52, "mode": "demo"}
    out = bag_mod.bag_level(image)
    out["mode"] = "real"
    return out


def run_label(image, drug):
    image = _shrink(image)
    if status()["bag"] == "demo":
        out = _simple_match(DEMO_LABEL_TEXT, drug)
        out.update(text=DEMO_LABEL_TEXT, volume_ml=500, mode="demo")
        return out
    out = bag_mod.check_label(image, drug)
    out["mode"] = "real"
    return out


def rematch_label(label, drug):
    """The prescribed drug changed after analysis: match the label text already read against the new
    drug. No OCR is run again, so this is instant. The bag size read from the label is kept."""
    text = label.get("text", "")
    out = None
    if bag_mod is not None:
        try:
            out = bag_mod.match_label(text, drug)
        except Exception:
            out = None
    if out is None:
        out = _simple_match(text, drug)
        out["text"] = text
    out["volume_ml"] = label.get("volume_ml")
    out["mode"] = label.get("mode")
    return out


def _simple_match(text, drug):
    """Match without rapidfuzz: any name of the drug (drugs.py) written in the text."""
    names = drugs_mod.other_names(drug) if drugs_mod is not None else [drug]
    t = " ".join(str(text).lower().split())
    hit = next((n for n in names if " ".join(n.lower().split()) in t), None)
    return {"match": hit is not None, "score": 100 if hit else 0, "matched_name": hit}


def analyze_bed(video_path, bag_image, label_image, drug, rx_mlh, drop_factor):
    """Run the three checks for one bed. Images are OpenCV (BGR) arrays or paths."""
    rate = run_rate(video_path, drop_factor)
    bag = run_bag(bag_image)
    label = run_label(label_image, drug)
    return {"inputs": {"drug": drug, "rx_mlh": rx_mlh, "drop_factor": drop_factor},
            "rate": rate, "bag": bag, "label": label}


if __name__ == "__main__":      # python app/backends.py  -> what works and what is missing
    s = warmup()
    print("القطرات:", {"yolo": "YOLO (حقيقي)", "motion": "طريقة الحركة (حقيقي، بدون YOLO)", "demo": "قيم تجريبية"}[s["rate"]])
    print("الكيس والملصق:", "حقيقي" if s["bag"] == "real" else "قيم تجريبية")
    for part, items in missing().items():
        for m in items:
            print(f"  ناقص ({part}): {m}")
