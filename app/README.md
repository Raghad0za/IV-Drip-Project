# App: nurse dashboard demo

Streamlit app that follows the agreed iPad mockup (Arabic, RTL):

1. **Bed board**: 6 beds. Beds 1, 2, 3, 4 and 6 are examples that show every status; bed 5 is free for the live demo.
2. **Bed page**: upload the drip video, a bag photo and a label photo, enter the prescription (drug, mL/h, drop factor), press **حلّل السرير**.
3. **Result**: rate vs prescription, mL left and time to empty, label match. The bed then appears on the board.
4. **Discharge** with a confirm dialog and an undo.

| File | What it does |
| --- | --- |
| `app.py` | The screens |
| `backends.py` | Calls Jana's `measure_rate` (drop_detection/rate.py) and Hasan's `bag_level` / `check_label` (bag_label/iv_monitor.py) |
| `iv_logic.py` | The status rules (normal / warning / critical / finished) and the example beds |

## Status rules

| Status | When |
| --- | --- |
| Finished (انتهى المحلول) | no drops and bag under 10% |
| Critical (حرج) | label does not match, rate off by more than 25%, or no drops |
| Warning (تنبيه) | rate off by more than 10%, bag almost empty, or under 1 hour left |
| Normal (طبيعي) | everything within the prescription |

## Run it locally (recommended for the demo)

From the repo root:

```bash
pip install -r app/requirements.txt
streamlit run app/app.py
```

The model files must be in place:

- `drop_detection/drops.pt` (Jana): **not in the repo yet**. Without it the app uses rate.py's motion method.
- `bag_label/best.pt` (Hasan): already there.

The sidebar shows which models are running. Anything missing falls back to demo values, so the app never crashes during the presentation.

## Put it online (Hugging Face Spaces, free)

1. Create a new Space on huggingface.co: SDK **Streamlit**, hardware **CPU basic** (free, 16 GB RAM, enough for YOLO + EasyOCR).
2. Push the repo to the Space (`drop_detection/`, `bag_label/`, `app/`).
3. At the root of the Space put a `README.md` that starts with:
   ```yaml
   ---
   title: IV Monitor
   sdk: streamlit
   app_file: app/app.py
   ---
   ```
4. Copy `app/requirements.txt` to the root as `requirements.txt`.
5. `.pt` files are large: track them with Git LFS (`git lfs track "*.pt"`), or upload them through the Space's web page.

The first analysis after the Space wakes up is slow (EasyOCR downloads its models). Run one analysis before the presentation starts.
