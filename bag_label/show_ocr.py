"""
Draw the OCR result on each test image, so you can SEE what was read.

    python show_ocr.py                        # real_labels/  ->  ocr_results/
    python show_ocr.py --folder synthetic_labels

For every image in the folder's labels.csv it saves ocr_results/<name>.jpg with:
  - a green box around every piece of text the English pass found
  - an orange box around every piece of text the Arabic pass found
  - a banner on top: correct drug, MATCH / NO MATCH, score, and the bag level
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from iv_monitor import _get_readers, bag_level, match_label

GREEN, ORANGE = (40, 170, 40), (0, 140, 255)


def draw(image_path, drug):
    img = cv2.imread(str(image_path))
    level = bag_level(img)

    # OCR once: draw the boxes and keep the text for matching
    texts = []
    english, arabic = _get_readers()
    for reader, color in ((english, GREEN), (arabic, ORANGE)):
        for box, text, _conf in reader.readtext(img):
            texts.append(text)
            cv2.polylines(img, [np.array(box, dtype=np.int32)], True, color, 2)

    label = match_label(" ".join(texts), drug)
    ok = label["match"]

    # banner above the image
    w = img.shape[1]
    scale = max(w / 900, 0.5)
    banner = np.full((int(110 * scale), w, 3), (40, 120, 40) if ok else (40, 40, 190), np.uint8)
    lines = [
        f"{'MATCH' if ok else 'NO MATCH'}  score {label['score']}  ({drug})",
        f"level: {level['level']}  fill: {level['fill_ratio']}   green=EN  orange=AR",
    ]
    for i, text in enumerate(lines):
        cv2.putText(banner, text, (10, int((40 + i * 50) * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9 * scale, (255, 255, 255), max(1, int(2 * scale)), cv2.LINE_AA)
    return np.vstack([banner, img])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--folder", default="real_labels")
    p.add_argument("--out", default="ocr_results")
    args = p.parse_args()
    folder, out = Path(args.folder), Path(args.out)
    out.mkdir(exist_ok=True)

    with open(folder / "labels.csv", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            result = draw(folder / row["file"], row["drug_en"])
            name = Path(row["file"]).stem + ".jpg"
            cv2.imwrite(str(out / name), result)
            print("saved", out / name)


if __name__ == "__main__":
    main()
