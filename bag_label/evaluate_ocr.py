"""
Test check_label() on the synthetic images.

For every image we check two things:
  1. the CORRECT drug  -> should match       (catches OCR misses)
  2. a WRONG drug      -> should NOT match   (catches false alarms, the dangerous case)

    python evaluate_ocr.py                       # uses synthetic_labels/
    python evaluate_ocr.py --folder other_folder
"""
import argparse
import csv
import random
from pathlib import Path

from drugs import DRUGS
from iv_monitor import check_label


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--folder", default="synthetic_labels")
    args = p.parse_args()
    folder = Path(args.folder)
    random.seed(0)

    with open(folder / "labels.csv", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    correct_hits, wrong_rejected = 0, 0
    for i, row in enumerate(rows):
        image = folder / row["file"]
        # Alternate between asking in English and in Arabic
        prescribed = row["drug_en"] if i % 2 == 0 else row["drug_ar"]
        wrong = random.choice([en for en, _ in DRUGS if en != row["drug_en"]])

        right = check_label(image, prescribed)
        bad = check_label(image, wrong)
        correct_hits += right["match"]
        wrong_rejected += not bad["match"]

        status = "OK " if right["match"] and not bad["match"] else "ERR"
        print(f"{status} {row['file']}  prescribed={prescribed!r:24} score={right['score']:3}  "
              f"wrong={wrong!r:22} score={bad['score']:3}  | read: {right['text']}")

    n = len(rows)
    print(f"\nCorrect drug matched : {correct_hits}/{n}")
    print(f"Wrong drug rejected  : {wrong_rejected}/{n}")


if __name__ == "__main__":
    main()
