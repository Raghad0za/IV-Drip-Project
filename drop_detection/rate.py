"""
rate.py: drip rate from a video of an IV drip chamber.

    from rate import measure_rate
    measure_rate("video.mp4", drop_factor=20)
    # {"dpm": 32.4, "ml_h": 97.2, "drops": 15, "status": "ok"}

How it works
1. Detection: YOLOv8n (drops.pt) finds the drop at the drip tip in every frame (conf 0.4).
   If a frame has several boxes, the top-most one (smallest top edge) is the drop at the tip.
2. Counting: the drop at the tip grows (its bottom edge moves down), then falls and is replaced
   by a new small drop (the bottom edge jumps back up). One confirmed jump = one drop.
   Guards against miscounts:
     * the old drop must have been seen for MIN_DROP_FRAMES frames,
     * the jump must hold for CONFIRM_FRAMES frames (a one-frame jitter is ignored),
       staying at least CONFIRM_FRAC of the old drop's height above its old bottom,
     * short detection drop-outs are ignored (up to MAX_MISSING_S),
     * at least MIN_GAP_S seconds between two counted drops (max 150 dpm).
3. Rate: dpm = 60 / median of the last 8 intervals between drops.
   ml/h = dpm * 60 / drop_factor. Fewer than 3 drops → status "no_drops".

Fallback: if YOLO does not see the drop in the first 2 s (a very different background, lighting
or drip set), a motion method counts sharp changes in a window at the drip tip instead.

Detection (needs the model) and counting (pure logic, `count_drops`) are separate,
so the counting can be tested without a model or a video.
"""
from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass, field
from pathlib import Path

# ── settings ──────────────────────────────────────────────────────────────
WEIGHTS = Path(__file__).with_name("drops.pt")
CONF = 0.4              # chosen on val: recall 0.985 with 1 of 35 empty frames falsely detected
IMGSZ = 640
MAX_SECONDS = 30        # only the first 30 s are analysed
MIN_GAP_S = 0.4         # design parameter: at most 150 dpm (> 200 dpm is free flow and shows no separate drops)
RESET_FRAC = 0.5        # the bottom edge must jump up by more than half the old drop's height
CONFIRM_FRAC = 0.25     # ...and stay at least a quarter of that height above the old bottom (the new drop grows fast)
MIN_DROP_FRAMES = 3     # the old drop must have been seen this many frames
CONFIRM_FRAMES = 2      # the jump must hold this many frames
MAX_MISSING_S = 0.5     # longer gaps without a detection reset the tracker (no count)
COLUMN_WIDTHS = 4.0     # ignore boxes further than this many box widths from the drip column
LAST_INTERVALS = 8

_MODEL = None


def _model(weights=WEIGHTS):
    global _MODEL
    if _MODEL is None:
        from ultralytics import YOLO
        _MODEL = YOLO(str(weights))
    return _MODEL


# ── counting (pure logic) ─────────────────────────────────────────────────
@dataclass
class Obs:
    t: float            # seconds
    top: float          # pixels
    bottom: float
    height: float
    x: float            # centre x


@dataclass
class CountResult:
    events: list = field(default_factory=list)     # times (s) of counted drops
    resets_ignored: int = 0                        # confirmed resets closer than MIN_GAP_S
    jitters_ignored: int = 0                       # jumps that did not hold


def count_drops(observations, min_gap_s=MIN_GAP_S, reset_frac=RESET_FRAC, confirm_frac=CONFIRM_FRAC,
                min_drop_frames=MIN_DROP_FRAMES, confirm_frames=CONFIRM_FRAMES,
                max_missing_s=MAX_MISSING_S) -> CountResult:
    """observations: list of Obs (one per processed frame) or None where nothing was detected."""
    res = CountResult()
    last, age, pending, last_event = None, 0, None, -1e9
    for o in observations:
        if o is None:
            continue
        if last is not None and o.t - last.t > max_missing_s:     # long drop-out: start fresh
            last, age, pending = None, 0, None
        if last is None:
            last, age = o, 1
            continue

        if pending is not None:
            still_up = o.bottom < pending["old_bottom"] - confirm_frac * pending["old_height"]
            if still_up:
                pending["frames"] += 1
                if pending["frames"] >= confirm_frames:
                    if pending["t"] - last_event >= min_gap_s:
                        res.events.append(pending["t"]); last_event = pending["t"]
                    else:
                        res.resets_ignored += 1
                    age, pending = pending["frames"], None
            else:                                                # went back down: it was jitter
                res.jitters_ignored += 1
                pending = None
                age += 1
            last = o
            continue

        jump_up = last.bottom - o.bottom
        if jump_up > reset_frac * last.height and age >= min_drop_frames:
            pending = {"t": o.t, "old_bottom": last.bottom, "old_height": last.height, "frames": 1}
            if confirm_frames <= 1:
                if o.t - last_event >= min_gap_s:
                    res.events.append(o.t); last_event = o.t
                pending, age = None, 1
        else:
            age += 1
        last = o
    return res


def rate_from_events(events, drop_factor=20, last_intervals=LAST_INTERVALS):
    if len(events) < 3:
        return {"dpm": 0, "ml_h": 0, "drops": len(events), "status": "no_drops"}
    gaps = [b - a for a, b in zip(events, events[1:])][-last_intervals:]
    dpm = 60 / statistics.median(gaps)
    return {"dpm": round(dpm, 1), "ml_h": round(dpm * 60 / drop_factor, 1), "drops": len(events), "status": "ok"}


# ── detection ─────────────────────────────────────────────────────────────
def detect(video_path, conf=CONF, imgsz=IMGSZ, max_seconds=MAX_SECONDS, frame_step=1, weights=WEIGHTS, device=None):
    """Run the model on the first max_seconds of the video. Returns (observations, fps, n_frames)."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    model = _model(weights)
    raw, f = [], 0
    while f < max_seconds * fps:
        ok, frame = cap.read()
        if not ok:
            break
        if f % frame_step == 0:
            r = model.predict(frame, conf=conf, imgsz=imgsz, verbose=False, device=device)[0]
            boxes = r.boxes.xyxy.cpu().numpy().tolist() if len(r.boxes) else []
            raw.append((f / fps, boxes))
        f += 1
    cap.release()

    # lock onto the drip column: drops fall straight down, so boxes far to the side are not drops
    xs = sorted((b[0] + b[2]) / 2 for _, bs in raw for b in bs)
    ws = sorted(b[2] - b[0] for _, bs in raw for b in bs)
    x_med = xs[len(xs) // 2] if xs else None
    w_med = ws[len(ws) // 2] if ws else None
    obs = []
    for t, boxes in raw:
        if x_med is not None:
            boxes = [b for b in boxes if abs((b[0] + b[2]) / 2 - x_med) <= COLUMN_WIDTHS * max(w_med, 1)]
        if not boxes:
            obs.append(None); continue
        b = min(boxes, key=lambda b: b[1])            # top-most box = drop at the tip
        obs.append(Obs(t=t, top=b[1], bottom=b[3], height=b[3] - b[1], x=(b[0] + b[2]) / 2))
    return obs, fps, f


# ── fallback: motion method (no learning) ─────────────────────────────────
# When the model does not recognise the drop (a very different background, lighting or drip set),
# a falling drop still changes the pixels below the drip tip for 1 to 2 frames. The tip is found
# automatically (the spot whose frame-to-frame change is the most "spiky"), followed through camera
# drift by template matching, and every sharp spike of change there is one drop.
MOTION_WIDTH = 384          # frames are downscaled to this width (speed, and less compression noise)
LOCATE_SECONDS = 10         # seconds used to find the drip tip
ROI_W, ROI_H = 0.040, 0.10  # window around the tip, as a fraction of frame width / height
PEAK_K = 8.0                # a spike must exceed median + PEAK_K * MAD of the signal...
PEAK_REL = 0.35             # ...and PEAK_REL of the strongest spikes...
PEAK_MIN_RATIO = 6.0        # ...and PEAK_MIN_RATIO times the median (real drops ~12x, noise ~5x on test videos)...
PEAK_MIN_ABS = 4.0          # ...and at least 4 grey levels of mean change (noise peaked at ~2)
PROBE_FRAMES = 20           # frames sampled from the first 2 s to decide between YOLO and motion
PROBE_MIN_RATE = 0.5        # YOLO is used if it sees the drop in at least half of the probe frames


def _read_gray(video_path, max_seconds, width=MOTION_WIDTH):
    import cv2, numpy as np
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames, f = [], 0
    while f < max_seconds * fps:
        ok, fr = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        s = width / g.shape[1]
        frames.append(cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA).astype(np.float32))
        f += 1
    cap.release()
    return frames, fps


def _locate_tip(frames, fps):
    """Spot whose frame-to-frame change is the most spiky: p99 / median of the change, smoothed."""
    import cv2, numpy as np
    n = min(len(frames), int(LOCATE_SECONDS * fps))
    D = np.stack([cv2.blur(np.abs(frames[i] - frames[i - 1]), (5, 5)) for i in range(1, n)])
    score = np.percentile(D, 99, axis=0) / (np.median(D, axis=0) + 1.0)
    score = cv2.GaussianBlur(score, (0, 0), 3)
    y, x = np.unravel_index(np.argmax(score), score.shape)
    return int(x), int(y), float(score.max())


def _track(frames, x, y):
    """Sub-pixel offset of the chamber around (x, y) in every frame, relative to the middle frame."""
    import cv2, numpy as np
    H, W = frames[0].shape
    ref = frames[len(frames) // 2]
    tw, th = int(0.16 * W), int(0.30 * H)                    # template: chamber around and above the tip
    tx0, ty0 = max(0, x - tw // 2), max(0, y - int(0.75 * th))
    tx1, ty1 = min(W, tx0 + tw), min(H, ty0 + th)
    T = ref[ty0:ty1, tx0:tx1]
    win = cv2.createHanningWindow((T.shape[1], T.shape[0]), cv2.CV_32F)
    m = int(0.08 * W)                                         # search margin
    sx0, sy0 = max(0, tx0 - m), max(0, ty0 - m)
    sx1, sy1 = min(W, tx1 + m), min(H, ty1 + m)
    offs = []
    for g in frames:
        r = cv2.matchTemplate(g[sy0:sy1, sx0:sx1], T, cv2.TM_CCOEFF_NORMED)
        _, _, _, (mx, my) = cv2.minMaxLoc(r)
        dx, dy = mx + sx0 - tx0, my + sy0 - ty0
        P = g[ty0 + dy:ty0 + dy + T.shape[0], tx0 + dx:tx0 + dx + T.shape[1]]
        if P.shape == T.shape:
            (rx, ry), _ = cv2.phaseCorrelate(T, P, win)
            if abs(rx) < 2 and abs(ry) < 2:
                dx, dy = dx + rx, dy + ry
        offs.append((dx, dy))
    return _smooth_path(offs)


def _smooth_path(offs, med_win=15, mean_win=9):
    """The camera moves slowly, so frame-to-frame jumps in the tracked offset are tracking noise.
    A running median then a running mean removes them (otherwise each jump looks like a drop)."""
    import numpy as np
    o = np.asarray(offs, float)
    out = np.empty_like(o)
    for k in range(o.shape[1]):
        v = o[:, k]
        pad = np.pad(v, med_win // 2, mode="edge")
        med = np.array([np.median(pad[i:i + med_win]) for i in range(len(v))])
        pad = np.pad(med, mean_win // 2, mode="edge")
        out[:, k] = np.convolve(pad, np.ones(mean_win) / mean_win, mode="valid")
    return [tuple(r) for r in out]


def motion_events(video_path, max_seconds=MAX_SECONDS, min_gap_s=MIN_GAP_S):
    """→ (events, info). Count drops from sharp changes in a window at the drip tip."""
    import cv2, numpy as np
    frames, fps = _read_gray(video_path, max_seconds)
    if len(frames) < 3:
        return [], {"fps": fps, "frames": len(frames)}
    H, W = frames[0].shape
    x, y, peakiness = _locate_tip(frames, fps)
    offs = _track(frames, x, y)
    w, h = max(8, int(ROI_W * W)), max(12, int(ROI_H * H))
    roi = lambda i: cv2.getRectSubPix(frames[i], (w, h), (x + offs[i][0], y + offs[i][1]))
    sig = np.array([0.0] + [np.abs(roi(i) - roi(i - 1)).mean() for i in range(1, len(frames))])
    med = np.median(sig); mad = np.median(np.abs(sig - med)) + 1e-6
    top = np.sort(sig)[-max(3, len(sig) // 100):].mean()
    thr = max(med + PEAK_K * mad, PEAK_REL * top, PEAK_MIN_RATIO * med, PEAK_MIN_ABS)
    events, last = [], -1e9
    for i in range(1, len(sig) - 1):
        if sig[i] >= thr and sig[i] >= sig[i - 1] and sig[i] >= sig[i + 1]:
            t = i / fps
            if t - last >= min_gap_s:
                events.append(t); last = t
            elif events and sig[i] > sig[int(round(last * fps))]:   # keep the stronger spike of a close pair
                events[-1] = t; last = t
    info = {"fps": fps, "frames": len(frames), "tip_xy": (round(x / W, 3), round(y / H, 3)),
            "peakiness": round(peakiness, 1), "threshold": round(float(thr), 2),
            "signal": [(i / fps, float(v)) for i, v in enumerate(sig)]}
    return events, info


def _yolo_probe(video_path, conf=CONF, imgsz=IMGSZ, weights=WEIGHTS, device=None):
    """Share of frames (sampled from the first 2 s) where YOLO sees a drop."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(min(2 * fps, cap.get(cv2.CAP_PROP_FRAME_COUNT) or 2 * fps))
    idx = sorted(set(int(i * n / PROBE_FRAMES) for i in range(PROBE_FRAMES)))
    model, seen, tried = _model(weights), 0, 0
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i); ok, fr = cap.read()
        if not ok:
            continue
        tried += 1
        seen += len(model.predict(fr, conf=conf, imgsz=imgsz, verbose=False, device=device)[0].boxes) > 0
    cap.release()
    return seen / max(tried, 1)


# ── main function ─────────────────────────────────────────────────────────
def measure_rate(video_path, drop_factor=20, debug=False, method="auto", **kw):
    """→ {"dpm", "ml_h", "drops", "status"}; with debug=True also events, signal and counters.

    method: "auto" (YOLO if it sees the drop in the first 2 s, otherwise motion), "yolo" or "motion".
    """
    det_kw = {k: kw[k] for k in ("conf", "imgsz", "max_seconds", "frame_step", "weights", "device") if k in kw}
    cnt_kw = {k: kw[k] for k in ("min_gap_s", "reset_frac", "confirm_frac", "min_drop_frames", "confirm_frames", "max_missing_s") if k in kw}
    probe = None
    if method == "auto":
        probe = _yolo_probe(video_path, **{k: v for k, v in det_kw.items() if k in ("conf", "imgsz", "weights", "device")})
        method = "yolo" if probe >= PROBE_MIN_RATE else "motion"

    if method == "motion":
        events, info = motion_events(video_path, max_seconds=kw.get("max_seconds", MAX_SECONDS),
                                     min_gap_s=kw.get("min_gap_s", MIN_GAP_S))
        out = rate_from_events(events, drop_factor)
        if debug:
            out.update(method="motion", yolo_probe=probe, events=[round(t, 3) for t in events],
                       fps=info["fps"], frames=info["frames"], detection_rate=None,
                       tip_xy=info.get("tip_xy"), threshold=info.get("threshold"),
                       motion_signal=info.get("signal", []))
        return out

    obs, fps, n_frames = detect(video_path, **det_kw)
    cr = count_drops(obs, **cnt_kw)
    out = rate_from_events(cr.events, drop_factor)
    if debug:
        seen = sum(o is not None for o in obs)
        out.update(method="yolo", yolo_probe=probe, events=[round(t, 3) for t in cr.events], fps=fps, frames=n_frames,
                   detection_rate=round(seen / max(len(obs), 1), 3),
                   resets_ignored=cr.resets_ignored, jitters_ignored=cr.jitters_ignored,
                   signal=[(o.t, o.bottom, o.height) if o else None for o in obs])
    return out


def plot_debug(result, title="", path=None):
    """Signal over time with counted drops as red lines (needs debug=True).
    YOLO: bottom edge of the tip drop. Motion: change in the window at the tip."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 3))
    if result.get("method") == "motion":
        sig = result["motion_signal"]
        ax.plot([p[0] for p in sig], [p[1] for p in sig], lw=1, color="#2a78d6")
        if result.get("threshold") is not None:
            ax.axhline(result["threshold"], color="#8a8a85", lw=1, ls="--")
        ax.set_ylabel("change at the tip")
        extra = f"motion method · tip at {result.get('tip_xy')}"
    else:
        pts = [s for s in result["signal"] if s]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], ".", ms=3, color="#2a78d6")
        ax.invert_yaxis(); ax.set_ylabel("drop bottom (px)")
        extra = f"YOLO · detection {result['detection_rate']:.0%}"
    for t in result["events"]:
        ax.axvline(t, color="#e5484d", lw=0.8)
    ax.set_xlabel("time (s)")
    ax.set_title(f"{title}  {result['drops']} drops · {result['dpm']} dpm · {extra}")
    plt.tight_layout()
    if path: fig.savefig(path, dpi=130)
    plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Drip rate from a video")
    ap.add_argument("video"); ap.add_argument("--drop-factor", type=float, default=20)
    ap.add_argument("--conf", type=float, default=CONF); ap.add_argument("--weights", default=str(WEIGHTS))
    ap.add_argument("--method", default="auto", choices=["auto", "yolo", "motion"])
    a = ap.parse_args()
    print(measure_rate(a.video, drop_factor=a.drop_factor, conf=a.conf, weights=a.weights, method=a.method))
