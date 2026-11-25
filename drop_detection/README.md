# Drop Detection & Drip Rate

**Owner:** Jana  
**Part of:** IV Drip Project (drip rate, bag level, label check, nursing dashboard)

This module counts the drops falling in the drip chamber from a video and returns the drip rate. A YOLOv8n model detects the drop at the drip tip, and counting logic on top of the detections turns them into drops per minute. When the model cannot see the drop (a very different background or lighting), a motion based method takes over automatically.

## Status

| Step | Status |
|---|---|
| EDA and data preparation | ✅ Done |
| Model training (YOLOv8n, experiments A and B) | ✅ Done, model A selected |
| Drop counting and `measure_rate` | ✅ Done, YOLO counting + motion fallback |
| Evaluation on external videos | ⏳ With the evaluation team |

## Usage

Put `rate.py` and `drops.pt` in the same folder.

```python
from rate import measure_rate

measure_rate("video.mp4", drop_factor=20)
# → {"dpm": 49.2, "ml_h": 147.5, "drops": 8, "status": "ok"}
```

| Key | Meaning |
|---|---|
| `dpm` | drops per minute |
| `ml_h` | millilitres per hour = dpm × 60 / drop_factor |
| `drops` | number of drops counted |
| `status` | `"ok"`, or `"no_drops"` if fewer than 3 drops were found |

Only the first 30 seconds of the video are analysed. With `debug=True` the result also includes the method used, the drop times and the signal, and `plot_debug(result)` draws it.

From the command line: `python rate.py video.mp4 --drop-factor 20`

Install: `pip install -r requirements.txt` (Ultralytics is pinned to 8.4.218, the version the model was trained with).

## How a drop is counted

A drop is counted at the moment it **detaches from the drip tip**. Drops still forming at the end of the clip, and drops already falling at its start, are not counted.

**1. YOLO counting (default).** The model finds the drop at the tip in every frame (conf 0.4). The drop grows, so the bottom edge of its box moves down, then it falls and a new small drop replaces it, so the bottom edge jumps back up. One confirmed jump is one drop. Guards: the old drop must be seen for 3 frames, the jump must hold for 2 frames, short detection dropouts are ignored, and drops must be at least 0.4 s apart.

**2. Motion fallback.** If YOLO sees the drop in less than half of the frames sampled from the first 2 seconds, the code switches to a method that does not depend on how the drop looks. It finds the drip tip automatically (the spot whose frame to frame change is the most spiky), follows it through camera movement, and counts every sharp spike of change there as one drop.

**Rate:** dpm = 60 / median of the last 8 intervals between drops.

## Data

**Source:** my Roboflow fork of [Drip Infusion V2](https://universe.roboflow.com/andika-rizaldy/drip-infusion-v2) by Andika Rizaldy, licensed CC BY 4.0. In the fork, 7 frames that showed a clear drop without a box were annotated, and 3 frames with drops too small to box reliably were removed.

| | Value |
|---|---|
| Images | 9,468 from 16 videos (8 black background, 8 white) |
| Boxes | 9,401, class `drop` |
| Empty frames (confirmed negatives) | 102 |
| Median drop size at imgsz 640 | about 13 × 16 px |
| Split by video | train 11 videos (6,442 images), val 3 (1,681), test 2 (1,345) |

Full analysis and cleaning: [`EDA_drop_detection.ipynb`](EDA_drop_detection.ipynb)

## Training

YOLOv8n pretrained, imgsz 640, up to 80 epochs, patience 15, seed 42, on a T4 GPU.

| | A (default augmentation) | B (translate 0.3, scale 0.3, degrees 5, close_mosaic 15) |
|---|---|---|
| Val recall | 0.987 | 0.986 |
| Val mAP50 | 0.994 | 0.994 |
| Empty val frames with a false detection at conf 0.25 | 8 of 35 | 35 of 35 |
| Conf needed for (almost) no false detections | 0.4, recall 0.985 | 0.6, recall 0.972 |

**Selected: model A at conf 0.4.** Both find the drops equally well, but B also fires on empty drip tips, and a false box on an empty frame hides a drop from the counter.

Full training and comparison: [`train_drops.ipynb`](train_drops.ipynb)

## Results

**Test set** (2 unseen videos, model A):

| mAP50 | mAP50-95 | Precision | Recall | Recall at conf 0.4 |
|---|---|---|---|---|
| 0.995 | 0.587 | 0.990 | 0.989 | 0.987 |

Per video at conf 0.4: 70H-3 (black background) recall 0.995, 60P-3 (white background) recall 0.973. False detections on 5 of 16 empty test frames.

**Speed:** about 120 ms per frame on a Colab CPU with YOLO. The motion method processes a 10 s video in about 2.5 s on CPU.

**Counting on simulated drop sequences** (with missed detections, jitter and false boxes): 99 to 100% of drops counted between 15 and 40 dpm, rate error at most 2.7%.

**External videos** (stock footage, setups different from the training data):

| Video | Setup | Manual count | Counted | dpm | Method |
|---|---|---|---|---|---|
| 1 | Backlit window, different drip set, 50 fps | 8 | 8 | 49.2 | motion |
| 2 | Green background, 24 fps | 3 | 3 | 24.2 | motion |

On both videos YOLO did not see the drop at all (confidence 0, even after cropping to the chamber), so the motion fallback was used. Charts: [`results/`](results/)

## Limitations

1. All training videos come from one setup (same position, same pole, plain dark or grey wall). The model does not recognise drops on very different backgrounds or lighting, which is why the motion fallback exists.
2. Only about 1% of the training frames are empty, so there are few examples of a drip tip with no drop.
3. The rates in the training filenames (20, 60, 70, 80 dpm) cannot be verified, and our target range (below 40 dpm) is covered only by the 20 dpm videos.
4. The motion method's thresholds were tuned on a small number of videos. Strong camera shake or something moving in front of the chamber can disturb it.
5. Above about 60 dpm, counting accuracy drops (the splash ripple can merge with the next drop).

## Future work

**Two stage detection:** train a separate model to detect the drip chamber, crop it, and detect the drop inside the crop. This makes the small drop larger for the model, removes false detections outside the chamber and makes the system work with any camera framing. About 100 to 150 labeled chamber images from different setups would be enough, since the chamber is a large, easy object.

**More varied training data:** drops on bright, backlit and coloured backgrounds, and more empty frames.

## Files

| File | Description |
|---|---|
| `rate.py` | `measure_rate`: YOLO counting with motion fallback |
| `drops.pt` | Trained model (experiment A) |
| `EDA_drop_detection.ipynb` | Data analysis, label review and cleaning |
| `prepare_data.py` | Converts polygons, removes faulty boxes, splits by video, writes `data.yaml` |
| `train_drops.ipynb` | Training of experiments A and B, comparison and model selection |
| `requirements.txt` | Python packages |
| `results/` | Counting charts on the external videos |

## Test video requirements

* No slow motion, time lapse or cuts
* Fixed camera with the drip tip clearly visible
* At least 24 fps, and long enough for 5 or more drops (about 15 s at 20 dpm, 8 s at 40 dpm)
* Not taken from the training videos

## References

1. Hegde et al., *An Integrated Dual-Module Computer Vision System for IV Drip Rate and Fluid Level Monitoring*, IEEE Access, vol. 14, 2026.
2. Giaquinto et al., *Deep Learning-Based Computer Vision for Real-Time IV Drip Infusion Monitoring*, IEEE Sensors Journal, 2020.
3. Rizaldy et al., *Drip Infusion Monitoring and Data Logging System Based on YOLOv5*, JITK, 2025.
4. Dataset: [Drip Infusion V2](https://universe.roboflow.com/andika-rizaldy/drip-infusion-v2), Andika Rizaldy, Roboflow Universe, CC BY 4.0.
