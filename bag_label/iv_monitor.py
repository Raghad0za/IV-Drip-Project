"""
IV / Saline bag monitor.

    from iv_monitor import analyze, bag_level, check_label

    bag_level("bag.jpg")                          # how full the bag is
    check_label("bag.jpg", "Normal Saline 0.9%")  # is it the prescribed drug?
    analyze("bag.jpg", "Normal Saline 0.9%")      # both at once

Command line:
    python iv_monitor.py bag.jpg ["Normal Saline 0.9%"]
"""
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from rapidfuzz import fuzz

from drugs import other_names

# ------------------------------------------------------------------ settings
MODEL_PATH = Path(__file__).with_name("best.pt")   # YOLOv8m-seg
EMPTY, WATER = 0, 1                                 # model classes: 0 = bottle_seg (the empty/air part), 1 = water_seg

# fill ratio -> level. A full bag still has air in its neck, so it reads ~0.72, not 1.0.
LEVELS = [
    (0.65, "full",         "مليان"),
    (0.40, "half",         "نص"),
    (0.10, "almost empty", "قرب يخلص"),
    (0.00, "empty",        "فاضي"),
]

MATCH_THRESHOLD = 80   # fuzzy score (0-100) needed for a label match

FULL_FILL = 0.72       # fill_ratio of a completely full bag (measured on the test photos)
BAG_SIZES = [50, 100, 250, 500, 1000, 2000]   # standard IV bag sizes in mL

ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

_model = None
_readers = None


# ------------------------------------------------------------------ public API
def bag_level(image):
    """How full the bag is.

    image: file path or OpenCV (BGR numpy) image.
    Returns {"level": "half", "level_ar": "نص", "fill_ratio": 0.52}
    or      {"level": "no bag found", "level_ar": "ما فيه كيس", "fill_ratio": None}
    """
    result, water_i, empty_i = _detect(image)
    if result is None:
        return {"level": "no bag found", "level_ar": "ما فيه كيس", "fill_ratio": None}

    if water_i is None:
        ratio = 0.0                                          # bag found, no fluid
    else:
        masks = result.masks.data.cpu().numpy() > 0.5
        water = masks[water_i]
        empty = masks[empty_i] & ~water if empty_i is not None else np.zeros_like(water)
        ratio = water.sum() / (water.sum() + empty.sum())    # fill = water / (water + air)

    for threshold, name_en, name_ar in LEVELS:
        if ratio >= threshold:
            return {"level": name_en, "level_ar": name_ar, "fill_ratio": round(float(ratio), 2)}


def check_label(image, drug_prescribed, threshold=MATCH_THRESHOLD):
    """Read the bag label (Arabic + English) and compare it with the prescribed drug.

    drug_prescribed: English or Arabic name, or a synonym (drugs.py) - all names of the drug are tried.
    Returns {"text": "<all text read>", "match": True, "score": 100,
             "matched_name": "Normal Saline 0.9%", "volume_ml": 500}
    """
    return match_label(read_label(image), drug_prescribed, threshold)


def match_label(text, drug_prescribed, threshold=MATCH_THRESHOLD):
    """Same as check_label, but on text that was already read (OCR is the slow part,
    so read once with read_label and match as many drugs as needed)."""
    clean_text = _normalize(text)

    best_score, best_name = 0, None
    for name in other_names(drug_prescribed):
        clean_name = _normalize(name)
        score = max(
            fuzz.partial_ratio(clean_name, clean_text),   # name appears as written
            _all_words_score(clean_name, clean_text),     # every word appears, any order
        )
        if score > best_score:
            best_score, best_name = score, name

    # Numbers in the prescription ("5%", "0.9%") must be on the label: Dextrose 5% must not match Dextrose 10%.
    numbers = re.findall(r"\d+(?:\.\d+)?", _normalize(drug_prescribed))
    numbers_ok = all(n in re.findall(r"\d+(?:\.\d+)?", clean_text) for n in numbers)

    return {
        "text": text,
        "match": best_score >= threshold and numbers_ok,
        "score": round(best_score),
        "matched_name": best_name,
        "volume_ml": bag_volume(text),
    }


def analyze(image, drug_prescribed=None):
    """Both checks in one call, plus an estimate of the mL left in the bag.
    The label check (and so the mL estimate) is skipped when no drug is given.

    Returns {"level": {...bag_level...},
             "label": {...check_label...} or None,
             "remaining_ml": 250 or None}
    """
    img = _load_image(image)                                 # read the file once
    level = bag_level(img)
    label = check_label(img, drug_prescribed) if drug_prescribed else None
    return {
        "level": level,
        "label": label,
        "remaining_ml": remaining_ml(level["fill_ratio"], label["volume_ml"] if label else None),
    }


def bag_volume(text):
    """Bag size in mL read from the label text ("500 mL" -> 500), or None.
    Only standard bag sizes count, and the largest one wins, because labels also say
    things like "each 100 mL contains..." on a 1000 mL bag."""
    # unit is often misread ("mt", "m1t", "TL"), so accept any short word starting with m/t/l
    found = [int(n) for n in re.findall(r"\b(\d{2,4})\s*[mtl]\w{0,3}\b", _normalize(text))]
    sizes = [n for n in found if n in BAG_SIZES]
    return max(sizes) if sizes else None


def remaining_ml(fill_ratio, volume_ml):
    """Estimated mL left: fill ratio scaled so a full bag (FULL_FILL) = 100%, times the bag size.
    An estimate from the image area, not a measurement. None if either input is missing."""
    if fill_ratio is None or not volume_ml:
        return None
    return round(min(fill_ratio / FULL_FILL, 1.0) * volume_ml)


def read_label(image):
    """All text OCR finds on the image, as one string.
    Two passes: English-only (reads Latin text cleanly) + Arabic (reads Arabic text).
    A single mixed reader corrupts English words with Arabic letters ("Dextrose" -> "D٥»I0$0")."""
    img = _load_image(image)
    english, arabic = _get_readers()
    return " ".join(english.readtext(img, detail=0) + arabic.readtext(img, detail=0))


def bag_box(image):
    """Box (x1, y1, x2, y2) around the whole bag, or None. Used to crop to the bag."""
    result, water_i, empty_i = _detect(image)
    if result is None:
        return None
    boxes = result.boxes.xyxy.cpu().numpy()
    parts = [boxes[i] for i in (water_i, empty_i) if i is not None]
    return (int(min(p[0] for p in parts)), int(min(p[1] for p in parts)),
            int(max(p[2] for p in parts)), int(max(p[3] for p in parts)))


# ------------------------------------------------------------------ internals
def _detect(image):
    """Run YOLO and pick the bag's two parts: (result, water_index, empty_index).
    - fluid = most confident water_seg
    - air   = most confident bottle_seg above/below the fluid (same column),
              which ignores false detections on walls, hair, etc.
    result is None when no bag is found; an index is None when that part is missing."""
    result = _get_model()(_load_image(image), verbose=False)[0]
    if result.masks is None:
        return None, None, None

    classes = result.boxes.cls.cpu().numpy().astype(int)
    conf = result.boxes.conf.cpu().numpy()
    boxes = result.boxes.xyxy.cpu().numpy()
    water_ids = np.where(classes == WATER)[0]
    empty_ids = np.where(classes == EMPTY)[0]

    if len(water_ids) == 0:
        if len(empty_ids) == 0:
            return None, None, None
        return result, None, empty_ids[np.argmax(conf[empty_ids])]

    w = water_ids[np.argmax(conf[water_ids])]
    same_column = [i for i in empty_ids if boxes[i][0] < boxes[w][2] and boxes[i][2] > boxes[w][0]]
    e = max(same_column, key=lambda i: conf[i]) if same_column else None
    return result, w, e


def _all_words_score(name, text):
    """Lowest score among the drug-name words, so every word must be found (any order).
    Numbers are skipped; check_label checks them separately."""
    words = [w for w in name.split() if not re.search(r"\d", w)]
    text_words = text.split()
    if not words or not text_words:
        return 0
    return min(max(fuzz.ratio(w, t) for t in text_words) for w in words)


def _normalize(text):
    """Lowercase, Arabic digits -> 0-9, decimal comma -> dot, remove Arabic diacritics,
    unify أ/إ/آ->ا, ى->ي, ة->ه, drop punctuation."""
    text = text.lower().translate(ARABIC_DIGITS)
    text = re.sub(r"(\d)[,٫](\d)", r"\1.\2", text)                 # 0,9 -> 0.9
    text = re.sub(r"[ً-ْـ]", "", text)
    text = re.sub("[أإآ]", "ا", text)
    text = text.replace("ى", "ي").replace("ة", "ه")
    text = re.sub(r"[^\w\s.%]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _load_image(image):
    """File path -> OpenCV image; an image that is already loaded passes through."""
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image))
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image}")
        return img
    return image


def _get_model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO(str(MODEL_PATH))
    return _model


def _get_readers():
    global _readers
    if _readers is None:
        import easyocr
        import torch
        gpu = torch.cuda.is_available()
        _readers = (easyocr.Reader(["en"], gpu=gpu), easyocr.Reader(["ar", "en"], gpu=gpu))
    return _readers


# ------------------------------------------------------------------ command line
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    result = analyze(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    print("Level:", result["level"])
    if result["label"]:
        print("Label:", result["label"])
        print("Remaining (estimate):", result["remaining_ml"], "mL")
