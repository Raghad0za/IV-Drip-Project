"""
Test the label check on a folder of labelled images (default: synthetic_labels/).

For every image in the folder's labels.csv:
  1. the CORRECT drug  -> should match       (catches OCR misses)
  2. a WRONG drug      -> should NOT match   (catches false alarms, the dangerous case)
  3. the bag size (mL) read from the label   -> should equal the "volume" column

    python evaluate_ocr.py                        # synthetic_labels/
    python evaluate_ocr.py --folder real_labels
"""
import argparse
import csv
import random
from pathlib import Path

from drugs import DRUGS
from iv_monitor import bag_volume, match_label, read_label


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--folder", default="synthetic_labels")
    args = p.parse_args()
    folder = Path(args.folder)
    random.seed(0)

    with open(folder / "labels.csv", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    correct_hits = wrong_rejected = volume_ok = 0
    for i, row in enumerate(rows):
        text = read_label(folder / row["file"])            # OCR once, then match both drugs
        prescribed = row["drug_en"] if i % 2 == 0 else row["drug_ar"]   # alternate English / Arabic
        wrong = random.choice([en for en, _ in DRUGS if en != row["drug_en"]])

        right = match_label(text, prescribed)
        bad = match_label(text, wrong)
        volume = bag_volume(text)
        correct_hits += right["match"]
        wrong_rejected += not bad["match"]
        volume_ok += volume == int(row["volume"].split()[0])

        status = "OK " if right["match"] and not bad["match"] else "ERR"
        print(f"{status} {row['file']}  prescribed={prescribed!r:24} score={right['score']:3}  "
              f"wrong={wrong!r:22} score={bad['score']:3}  volume={volume}  | read: {text}")

    n = len(rows)
    print(f"\nCorrect drug matched : {correct_hits}/{n}")
    print(f"Wrong drug rejected  : {wrong_rejected}/{n}")
    print(f"Bag size read right  : {volume_ok}/{n}")


if __name__ == "__main__":
    main()
