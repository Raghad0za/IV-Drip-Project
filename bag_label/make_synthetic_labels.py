"""
Make synthetic label images to test the OCR: prints an Arabic + English drug label
onto each real bag photo (the model finds the bag, crops around it, places the label on it).

    python make_synthetic_labels.py                         # 25 images from test_bags/
    python make_synthetic_labels.py --bags my_photos --n 30

Output: synthetic_labels/img_XX.jpg + synthetic_labels/labels.csv (the correct drug per image).
"""
import argparse
import csv
import random
from pathlib import Path

import arabic_reshaper
import numpy as np
from bidi.algorithm import get_display
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from drugs import DRUGS, VOLUMES
from iv_monitor import bag_box

# Fonts that contain Arabic letters (first one found is used)
FONT_CANDIDATES = [
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/tahoma.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def find_font(user_font):
    for path in ([user_font] if user_font else []) + FONT_CANDIDATES:
        if path and Path(path).exists():
            return path
    raise FileNotFoundError("No Arabic-capable font found. Pass one with --font")


def arabic(text):
    """PIL draws letters left-to-right and unjoined; this fixes Arabic shaping + direction."""
    return get_display(arabic_reshaper.reshape(text))


def font(path, size):
    # BASIC layout = we handle Arabic ourselves with arabic() above
    return ImageFont.truetype(path, size, layout_engine=ImageFont.Layout.BASIC)


def make_label(font_path, drug_en, drug_ar, volume, width):
    """White printed label with English name, Arabic name, volume and a lot number."""
    height = int(width * 0.55)
    label = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(label)
    big, small = font(font_path, width // 11), font(font_path, width // 16)

    lines = [
        (drug_en, big),
        (arabic(drug_ar), big),
        (f"{volume}  -  IV Infusion", small),
        (f"LOT {random.randint(10000, 99999)}   EXP 12/2027", small),
    ]
    y = height * 0.08
    for text, f in lines:
        text_w = d.textlength(text, font=f)
        d.text(((width - text_w) / 2, y), text, font=f, fill=(20, 20, 20))
        y += f.size * 1.35
    d.rectangle([0, 0, width - 1, height - 1], outline=(0, 90, 160), width=4)
    return label


def crop_to_bag(photo, out_width=480):
    """Use the YOLO model to find the bag, crop around it (like a close-up camera)
    and resize. Returns (cropped image, bag box in the cropped image) or (photo, None)."""
    box = bag_box(np.asarray(photo)[:, :, ::-1])         # PIL RGB -> OpenCV BGR
    if box is None:
        return photo, None

    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    pad_x, pad_y = int(bw * 0.6), int(bh * 0.15)
    cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    cx2, cy2 = min(photo.width, x2 + pad_x), min(photo.height, y2 + pad_y)
    crop = photo.crop((cx1, cy1, cx2, cy2))

    scale = out_width / crop.width
    crop = crop.resize((out_width, int(crop.height * scale)))
    new_box = [int((x1 - cx1) * scale), int((y1 - cy1) * scale), int((x2 - cx1) * scale), int((y2 - cy1) * scale)]
    return crop, new_box


def augment(img):
    """Small rotation, blur, brightness change and noise so it looks less perfect."""
    img = img.rotate(random.uniform(-6, 6), expand=False, fillcolor=img.getpixel((0, 0)))
    if random.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.2)))
    arr = np.asarray(img).astype(np.float32) * random.uniform(0.8, 1.15)
    arr += np.random.normal(0, random.uniform(2, 8), arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=25, help="number of images (20-30)")
    p.add_argument("--bags", default="test_bags", help="folder of bag photos")
    p.add_argument("--out", default="synthetic_labels")
    p.add_argument("--font", help="path to a .ttf font that supports Arabic")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    font_path = find_font(args.font)
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    bag_photos = sorted(f for f in Path(args.bags).iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not bag_photos:
        raise SystemExit(f"No .jpg/.png photos found in {args.bags}/")

    rows = []
    for i in range(args.n):
        drug_en, drug_ar = DRUGS[i % len(DRUGS)]          # every drug appears evenly
        volume = random.choice(VOLUMES)

        photo = bag_photos[i % len(bag_photos)]
        bag, box = crop_to_bag(Image.open(photo).convert("RGB"))
        if box is None:
            print(f"Skipped {photo.name}: no bag found")
            continue

        # label sits on the bag, a bit narrower than the bag
        x1, y1, x2, y2 = box
        label = make_label(font_path, drug_en, drug_ar, volume, int((x2 - x1) * random.uniform(0.85, 1.0)))
        x = (x1 + x2 - label.width) // 2 + random.randint(-5, 5)
        y = int(y1 + (y2 - y1) * random.uniform(0.25, 0.4))
        bag.paste(label, (x, y))

        name = f"img_{i:02d}.jpg"
        augment(bag).save(out / name, quality=90)
        rows.append({"file": name, "drug_en": drug_en, "drug_ar": drug_ar, "volume": volume})

    with open(out / "labels.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} images + labels.csv to {out}/ (font: {font_path})")


if __name__ == "__main__":
    main()
