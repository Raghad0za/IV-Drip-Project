# Drop Detection & Drip Rate

**Owner:** Jana  
**Part of:** IV Drip Project (drip rate, bag level, label check, nursing dashboard)

This module counts the drops falling in the drip chamber from a video and returns the drip rate. It uses a YOLOv8n model trained to detect the drop at the drip tip, and counting logic on top of the detections.

## Status

| Step | Status |
|---|---|
| EDA and data preparation | ✅ Done |
| Model training (YOLOv8n) | ⏳ Next |
| Drop counting and `measure_rate` | ⏳ After training |
| Evaluation on external videos | ⏳ With the evaluation team |

## Output for integration

```python
from rate import measure_rate

measure_rate(video_path, drop_factor=20)
# → {"dpm": 32.4, "ml_h": 97.2, "drops": 15, "status": "ok"}
```

| Key | Meaning |
|---|---|
| `dpm` | drops per minute |
| `ml_h` | millilitres per hour = dpm × 60 / drop_factor |
| `drops` | number of drops counted |
| `status` | `"ok"`, or `"no_drops"` if fewer than 3 drops were found |

Only the first 20 to 30 seconds of the video are analysed.

## Data

**Source:** my Roboflow fork of [Drip Infusion V2](https://universe.roboflow.com/andika-rizaldy/drip-infusion-v2) by Andika Rizaldy, licensed CC BY 4.0. The uploader also coauthored *Drip Infusion Monitoring and Data Logging System Based on YOLOv5* (JITK, 2025).

**Changes in my fork:** 7 frames that showed a clear drop without a box were annotated, and 3 frames with drops too small to box reliably were removed.

**Final dataset**

| | Value |
|---|---|
| Images | 9,468 |
| Boxes | 9,401 |
| Empty frames (confirmed negatives) | 102 |
| Videos | 16 (8 black background, 8 white background) |
| Class | `drop` (originally `tetesan`) |
| Median drop size at imgsz 640 | about 13 × 16 px |

**Split by video** (frames from one video never appear in two splits)

| Split | Videos | Images |
|---|---|---|
| Train | 11 | 6,442 |
| Val | 3 (60H-3, 60P-2, 80H-1) | 1,681 |
| Test | 2 (60P-3, 70H-3) | 1,345 |

Both backgrounds appear in every split. The test videos are not used for any training decision.

## Cleaning

| Problem | Count | Fix |
|---|---|---|
| Visible drop but no box | 11 | 7 annotated in the fork, 4 removed |
| Duplicate boxes on one drop | 3 | Kept the larger box |
| Polygon labels | 7 | Converted to bounding boxes |
| Broken boxes (zero width or height) | 7 | Image removed |
| Box far from the chamber | 1 | Image removed |
| Two boxes: drop at the tip + previous drop falling | 35 | Valid, kept |

The original download is never modified. All fixes are applied to a copy, and every fix is previewed before it is applied.

## Key findings

1. Roboflow's original split put frames of 15 out of 16 videos in both train and valid, so I split by video instead.
2. Only 18 to 44% of the frame indices are present in each video, so the videos cannot be rebuilt from the images. Drip rate must be tested on complete external videos.
3. The drop is small and sits at almost the same position in every video. Models will be compared by recall, and stronger translate and scale augmentation will be tested.
4. In labeled sequences the drop forms at the tip, hangs, stretches, and is replaced by a new small drop. This is the starting hypothesis for the counting logic and will be confirmed on model predictions.

Full analysis: [`EDA_drop_detection.ipynb`](EDA_drop_detection.ipynb)

## Rebuild the dataset

1. Open `EDA_drop_detection.ipynb` in Google Colab.
2. Upload `prepare_data.py` to `/content`.
3. In Section 1, set `WORKSPACE`, `PROJECT` and `VERSION` for the Roboflow fork.
4. Run all cells. The notebook asks for a Roboflow API key (it is never saved).
5. The finished dataset is written to `/content/dataset` and saved to Google Drive as `iv_drip/dataset.zip`.

```bash
python prepare_data.py --src /content/clean --out /content/dataset
```

## Files

| File | Description |
|---|---|
| `EDA_drop_detection.ipynb` | Data analysis, label review and cleaning |
| `prepare_data.py` | Converts polygons, removes faulty boxes, splits by video, writes `data.yaml` |
| `train_drops.ipynb` | Model training *(coming next)* |
| `rate.py` | `measure_rate` function *(coming next)* |
| `drops.pt` | Trained model *(coming next)* |

## Test video requirements

* No slow motion, time lapse or cuts
* Fixed camera with the whole drip chamber and the drip tip clearly visible
* At least 30 seconds and 24 fps
* Below 40 drops per minute, ideally covering 30 to 40
* A dark background is preferred
* Not taken from the training videos

## References

1. Hegde et al., *An Integrated Dual-Module Computer Vision System for IV Drip Rate and Fluid Level Monitoring*, IEEE Access, vol. 14, 2026.
2. Giaquinto et al., *Deep Learning-Based Computer Vision for Real-Time IV Drip Infusion Monitoring*, IEEE Sensors Journal, 2020.
3. Rizaldy et al., *Drip Infusion Monitoring and Data Logging System Based on YOLOv5*, JITK, 2025.
4. Dataset: [Drip Infusion V2](https://universe.roboflow.com/andika-rizaldy/drip-infusion-v2), Andika Rizaldy, Roboflow Universe, CC BY 4.0.
