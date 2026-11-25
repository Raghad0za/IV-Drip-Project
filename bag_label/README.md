# Bag Level & Label Check

Part of **IV-Drip-Project**: this folder (`bag_label/`) handles the IV bag itself.

Checks a photo of an IV / saline bag for:

1. **Fluid level:** full · half · almost empty · empty (segmentation model)
2. **Drug label:** reads the Arabic/English label and checks it against the prescribed drug (OCR)
3. **Remaining volume (estimate):** fill level × bag size read from the label, e.g. half of 500 mL ≈ 250 mL

```python
from iv_monitor import analyze

analyze("bag.jpg", "Normal Saline 0.9%")
# {'level': {'level': 'full', 'level_ar': 'مليان', 'fill_ratio': 0.72},
#  'label': {'text': 'Normal Saline 0.9% محلول ملحي 500 mL ...', 'match': True, 'score': 100,
#            'matched_name': 'Normal Saline 0.9%', 'volume_ml': 500},
#  'remaining_ml': 500}
```

---

## 1. Setup

```bash
cd bag_label
pip install -r requirements.txt
```

- Python 3.9+. Run commands from inside `bag_label/`. To use it from another folder (e.g. `app/`), add it to the path:
  ```python
  import sys; sys.path.append("../bag_label")
  from iv_monitor import analyze
  ```
- `best.pt`, `iv_monitor.py` and `drugs.py` must stay together.
- **First run:** EasyOCR downloads its Arabic + English models (~100 MB) once, so it needs internet.
- **GPU:** used automatically when available (CUDA PyTorch). On CPU, OCR takes a few seconds per image.

## 2. API

Every function takes `image` as **a file path** or **an OpenCV image** (BGR numpy array, e.g. a webcam frame).

### `analyze(image, drug_prescribed=None)` → dict
Runs both checks and estimates the mL left. If `drug_prescribed` is not given, the label check is skipped (`"label": None`), and so is the mL estimate (`"remaining_ml": None`), because the bag size comes from the label.

```python
{"level": <bag_level result>, "label": <check_label result or None>, "remaining_ml": 250 or None}
```

### `bag_level(image)` → dict

| Key | Type | Example |
|---|---|---|
| `level` | str | `"full"`, `"half"`, `"almost empty"`, `"empty"`, `"no bag found"` |
| `level_ar` | str | `"مليان"`, `"نص"`, `"قرب يخلص"`, `"فاضي"`, `"ما فيه كيس"` |
| `fill_ratio` | float or None | `0.52` (0 = empty … ~0.72 = full). `None` if no bag found |

### `check_label(image, drug_prescribed, threshold=80)` → dict

| Key | Type | Meaning |
|---|---|---|
| `match` | bool | **The answer.** `True` = the label is the prescribed drug |
| `score` | int 0–100 | How closely the name matched (≥ `threshold` is required) |
| `matched_name` | str or None | The spelling that matched (English or Arabic) |
| `volume_ml` | int or None | Bag size read from the label (`500 mL` → 500). Only standard sizes count; `None` if not readable |
| `text` | str | All the text OCR read, useful for logs and debugging |

`drug_prescribed` can be English or Arabic (`"Paracetamol"` or `"باراسيتامول"`).

### Smaller helpers
- `read_label(image)` → str: OCR only, no matching.
- `bag_box(image)` → `(x1, y1, x2, y2)` or `None`: box around the bag, e.g. for cropping or drawing.

## 3. Integration examples

**Webcam / video frame**
```python
import cv2
from iv_monitor import bag_level

cap = cv2.VideoCapture(0)
ok, frame = cap.read()
if ok:
    print(bag_level(frame))
```

**Alert when a bag is running low**
```python
result = bag_level(frame)
if result["level"] in ("almost empty", "empty"):
    send_alert(f"IV bag {result['level']} ({result['fill_ratio']:.0%})")   # your own function
```

**Web API (FastAPI)**
```python
# pip install fastapi uvicorn python-multipart      run: uvicorn server:app
import cv2, numpy as np
from fastapi import FastAPI, UploadFile, Form
from iv_monitor import analyze

app = FastAPI()

@app.post("/check")
async def check(image: UploadFile, drug: str = Form(None)):
    img = cv2.imdecode(np.frombuffer(await image.read(), np.uint8), cv2.IMREAD_COLOR)
    return analyze(img, drug)
```

**Command line**
```bash
python iv_monitor.py bag.jpg "Normal Saline 0.9%"
```

### Good to know
- **Loading:** the YOLO and OCR models load on the first call and are reused after that. The first call is slow; later calls are fast. In a server, import once at startup.
- **Errors:** a path that can't be read raises `FileNotFoundError`. No bag in the image is **not** an error: you get `level = "no bag found"`.
- **Thread safety:** the models are shared. In a multi-threaded server, run one worker per process or put a lock around calls.

## 4. Settings (top of `iv_monitor.py`)

| Setting | Default | What it does |
|---|---|---|
| `LEVELS` | 0.65 / 0.40 / 0.10 | Fill-ratio limits for full / half / almost empty |
| `MATCH_THRESHOLD` | 80 | Minimum name score for a label match |
| `MODEL_PATH` | `best.pt` next to the script | The YOLO model file |
| `FULL_FILL` | 0.72 | fill_ratio of a completely full bag, used for the mL estimate |
| `BAG_SIZES` | 50 … 2000 | Standard bag sizes in mL that the label reader accepts |

The drug list (English ↔ Arabic) is in `drugs.py`, together with `SYNONYMS`: other names for the same drug as printed on real labels (Normal Saline = Sodium Chloride, Dextrose = Glucose). A prescription in any of these names matches a label showing any other.

## 5. How it works

**Level.** `best.pt` is a YOLOv8m segmentation model with two classes:
`water_seg` = the fluid, and `bottle_seg` = the **empty (air) part** of the bag.

`fill_ratio = water pixels ÷ (water + air pixels)`

Only the strongest fluid detection is used, plus the strongest air detection in the same column, so stray detections on walls or hair are ignored. A full bag still has air in its neck, so it reads about 0.72, not 1.0.

**Label.**
1. EasyOCR reads the image twice: an English-only pass and an Arabic pass. A single mixed reader corrupts English words with Arabic letters (`Dextrose` → `D٥»I0$0`).
2. Both texts are normalized: lowercase, Arabic digits → 0–9 (`٥` → `5`), decimal comma → dot (`0,9` → `0.9`), no Arabic diacritics, أ/إ/آ → ا, ى → ي, ة → ه.
3. Every name of the prescribed drug is tried: English, Arabic and synonyms.
4. The name scores a match if it appears as written (`partial_ratio`) **or** if every word appears in any order. Both tolerate OCR typos.
5. **Safety rule:** numbers in the prescription (`5%`, `0.9%`) must appear on the label, so `Dextrose 5%` never matches a `Dextrose 10%` bag.

## 6. Testing

```bash
python make_synthetic_labels.py    # prints drug labels on the photos in test_bags/ -> synthetic_labels/
python evaluate_ocr.py             # each image: right drug, wrong drug, and bag size (mL)
python show_ocr.py                 # saves real_labels/ with OCR boxes + result drawn -> ocr_results/
```

Results:

| Test | Result |
|---|---|
| Level, 10 test/val photos | 3 full, 5 half, 2 almost empty; all match what the photos show |
| Label, synthetic (25 images): correct drug matched | 24 / 25 |
| Label, synthetic: wrong drug rejected | **25 / 25** |
| Label, **real** bag labels (5 images): correct drug matched | 4 / 5 |
| Label, real: wrong drug rejected | **5 / 5** |

Run the real-label test with `python evaluate_ocr.py --folder real_labels`.

Misses:
- Synthetic: a very blurry label where OCR read `0.9%` as `0 9%`, so the safety rule refused to confirm it.
- Real: an Egyptian Glucose 5% bottle with red stylized print, where OCR read `Glucose` as `Glucozo` (score 71, threshold 80).

A wrong drug was never accepted.

**Remaining volume.** `remaining_ml = min(fill_ratio ÷ 0.72, 1) × bag size`. A full bag reads ~0.72 because the neck holds air.
It is an estimate from the image area, not a measurement. The bag size was read correctly on 23/25 synthetic and 2/5 real labels
(on the other 3 the "mL" print is tiny or vertical), and never read as a wrong size: when unsure it returns `None`.

## 7. Files

| File | Purpose |
|---|---|
| `iv_monitor.py` | **The module to import:** `analyze`, `bag_level`, `check_label` |
| `drugs.py` | Drug names, English ↔ Arabic, plus synonyms |
| `best.pt` | YOLOv8m-seg model |
| `requirements.txt` | Dependencies |
| `make_synthetic_labels.py`, `evaluate_ocr.py`, `show_ocr.py` | Testing only, not needed in production. `show_ocr.py` saves each test image with the OCR boxes and result drawn on it, to `ocr_results/` |
| `test_bags/`, `synthetic_labels/`, `real_labels/` | Test images (each label folder has a `labels.csv` with the correct drug) |

## 8. Limitations
- Tested on 10 photos of one bottle type; tune `LEVELS` for other bags or camera angles.
- No empty bag was in the test photos, so the "empty" level is untested.
- Only 5 real labels tested so far. Stylized print on a curved bottle is the weak spot: `Glucose` read as `Glucozo` (score 71) and the Arabic text on that label wasn't readable.
- OCR needs the label in focus and reasonably large in the image. A wide room shot won't be readable.
