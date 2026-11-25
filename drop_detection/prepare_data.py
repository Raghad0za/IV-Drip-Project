"""
prepare_data.py — تجهيز داتا كشف القطرات لـ YOLOv8 (جزء جنى)

يشتغل على أي مصدر بصيغة YOLO:
  - داتا Hegde (images/ + labels/ ، نتجاهل heatmaps/)
  - تصدير Roboflow بصيغة YOLOv8 (train/valid/test كل واحد فيه images/ + labels/)

وش يسوي:
  1) يلقى كل الصور ويربط كل صورة بملف الليبل حقها
  2) يفحص الليبلات (إحداثيات بين 0 و1، أرقام الكلاسات، ملفات خربانة)
  3) يحسب الصور السلبية (بدون قطرات) وأحجام الصناديق
  4) يجمّع الفريمات اللي من نفس الفيديو في مجموعة وحدة، ويقسم 70/20/10
     بالمجموعات (عشان ما يصير تسريب: فريمات شبه متطابقة في train و test)
  5) يكتب dataset/ جاهزة + data.yaml + صورة فحص فيها الصناديق

الاستخدام (في Colab):
  !python prepare_data.py --src /content/raw --out /content/dataset
"""
import argparse, random, re, shutil, sys
from collections import Counter, defaultdict
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_label(img: Path):
    """يدور ملف الليبل: نفس المجلد، أو /images/ → /labels/"""
    cands = [img.with_suffix(".txt")]
    parts = list(img.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].lower() == "images":
            p = parts.copy(); p[i] = "labels"
            cands.append(Path(*p).with_suffix(".txt"))
            break
    for c in cands:
        if c.exists():
            return c
    return None


def group_key(img: Path):
    """مفتاح المجموعة = اسم الملف بدون رقم الفريم.
    مثال: vid03_frame_0127.jpg → vid03_frame ، و Roboflow: x_jpg.rf.abc.jpg → x"""
    s = img.stem
    s = re.sub(r"\.rf\.[0-9a-f]+$", "", s)          # لاحقة Roboflow
    s = re.sub(r"_(jpe?g|png)$", "", s, flags=re.I)
    s = re.sub(r"[\s_\-\.]*\(?\d+\)?$", "", s)        # رقم الفريم في الآخر
    return (img.parent.name if img.parent.name.lower() not in
            {"images", "train", "valid", "val", "test"} else "") + "/" + (s or "_")


def read_label(path):
    """يرجع (boxes, errors). كل box = (cls, x, y, w, h)"""
    boxes, errs = [], []
    if path is None:
        return boxes, errs  # بدون ليبل = صورة سلبية
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        p = line.split()
        try:
            c = int(float(p[0])); vals = list(map(float, p[1:]))
        except ValueError:
            errs.append(f"{path.name}:{n} قيم مو أرقام"); continue
        if len(vals) == 4:
            x, y, w, h = vals
        elif len(vals) >= 6 and len(vals) % 2 == 0:   # polygon → bbox
            xs, ys = vals[0::2], vals[1::2]
            x, y = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
            w, h = max(xs) - min(xs), max(ys) - min(ys)
        else:
            errs.append(f"{path.name}:{n} عدد القيم {len(p)}"); continue
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            errs.append(f"{path.name}:{n} إحداثيات برا [0,1]"); continue
        boxes.append((c, x, y, w, h))
    return boxes, errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="مجلد الداتا الخام (بعد فك الضغط)")
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--split", nargs=3, type=float, default=[0.7, 0.2, 0.1])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keep-classes", action="store_true",
                    help="خلّ الكلاسات كما هي (الافتراضي: كل شي يصير كلاس 0 = drop)")
    ap.add_argument("--only-class", type=int, nargs="+", default=None,
                    help="خذي هالكلاسات بس (مثلاً رقم كلاس drop) واحذفي الباقي مثل chamber/level")
    ap.add_argument("--max-neg", type=float, default=None,
                    help="أقصى نسبة للصور الفاضية (class=null)، مثلاً 0.2 — الزايد ينحذف عشوائياً")
    ap.add_argument("--no-clean", action="store_true",
                    help="لا تستبعد الصناديق المتكسّرة/البعيدة")
    ap.add_argument("--no-group", action="store_true",
                    help="تقسيم عشوائي صورة صورة (مو منصوح فيه)")
    a = ap.parse_args()
    random.seed(a.seed)

    src, out = Path(a.src), Path(a.out)
    imgs = sorted(p for p in src.rglob("*") if p.suffix.lower() in IMG_EXT
                  and "heatmap" not in str(p).lower())
    if not imgs:
        sys.exit(f"ما لقيت صور في {src}")

    # ---------- 1+2) الربط والفحص ----------
    items, all_errs, missing = [], [], 0
    cls_count, box_wh = Counter(), []
    for img in imgs:
        lab = find_label(img)
        if lab is None:
            missing += 1
        boxes, errs = read_label(lab)
        all_errs += errs
        if errs:   # نستبعد الصورة كاملة بدل ما تنحسب سلبية بالغلط
            continue
        if a.only_class is not None:
            boxes = [b for b in boxes if b[0] in a.only_class]
        items.append((img, boxes))

    # ---------- تنظيف: صناديق متكسّرة (عرض/ارتفاع ≈ 0) أو بعيدة عن مكان القطرة في نفس الفيديو ----------
    if not a.no_clean:
        by_vid = defaultdict(list)
        for img, boxes in items:
            for _, x, y, *_ in boxes: by_vid[group_key(img)].append((x, y))
        med = {k: (sorted(v[0] for v in vs)[len(vs)//2], sorted(v[1] for v in vs)[len(vs)//2])
               for k, vs in by_vid.items()}
        kept, dropped = [], []
        for img, boxes in items:
            mx, my = med.get(group_key(img), (0.5, 0.5))
            bad = any(w < 0.004 or h < 0.003 or abs(x - mx) > 0.15 or abs(y - my) > 0.25
                      for _, x, y, w, h in boxes)
            (dropped if bad else kept).append((img, boxes))
        items = kept
        if dropped:
            print(f"تنظيف: استبعدت {len(dropped)} صورة فيها صندوق متكسّر أو بعيد:")
            for img, _ in dropped[:10]: print("   ", img.name[:70])
    try:
        from PIL import Image
        size_of = lambda im: Image.open(im).size
    except ImportError:
        size_of = lambda im: (1, 1)
    for img, boxes in items:
        if boxes:
            W, H = size_of(img); s = 640 / max(W, H)
        for c, x, y, w, h in boxes:
            cls_count[c] += 1; box_wh.append((w * W * s, h * H * s))

    n_neg = sum(1 for _, b in items if not b)
    print(f"\n=== ملخص الداتا ===")
    if a.max_neg is not None and n_neg > a.max_neg * len(items):
        pos = [it for it in items if it[1]]
        neg = [it for it in items if not it[1]]
        keep = int(a.max_neg / (1 - a.max_neg) * len(pos))
        print(f"الصور الفاضية {n_neg} ({100*n_neg/len(items):.1f}%) → نخلي {keep} بس (--max-neg {a.max_neg})")
        items = sorted(pos + random.sample(neg, keep), key=lambda t: str(t[0]))
        n_neg = keep
    print(f"صور: {len(items)} | بدون ملف ليبل: {missing} | سلبية (بدون قطرات): {n_neg} "
          f"({100*n_neg/len(items):.1f}%)")
    print(f"صناديق: {sum(cls_count.values())} | الكلاسات: {dict(cls_count)}")
    if box_wh:
        ws = sorted(w for w, _ in box_wh); hs = sorted(h for _, h in box_wh)
        mid = len(ws) // 2
        print(f"حجم الصندوق بـ imgsz=640: عرض median={ws[mid]:.1f}px، ارتفاع median={hs[mid]:.1f}px "
              f"(أصغر 10%: {ws[len(ws)//10]:.1f}×{hs[len(hs)//10]:.1f}px)")
        if ws[mid] < 12:
            print("  ⚠ الصناديق صغيرة مرة على 640 — فكري في قص منطقة الغرفة (ROI) أو imgsz أكبر")
    if all_errs:
        print(f"⚠ أخطاء في {len(all_errs)} سطر (أول 10):"); [print("  ", e) for e in all_errs[:10]]
    if missing > 0.5 * len(items):
        print("⚠ أكثر من نص الصور بدون ليبل — تأكدي من مسار --src وهيكل المجلدات")

    # ---------- 4) التقسيم بالمجموعات ----------
    groups = defaultdict(list)
    for i, (img, _) in enumerate(items):
        groups[str(i) if a.no_group else group_key(img)].append(i)
    keys = sorted(groups); random.shuffle(keys)
    print(f"\nمجموعات (فيديو/تسلسل): {len(keys)}")
    if not a.no_group and len(keys) > 0.8 * len(items):
        print("  ⚠ تقريباً كل صورة مجموعة لحالها — أسماء الملفات ما تكشف الفيديو المصدر.\n"
              "    افتحي أسماء الملفات وعدّلي group_key() لو فيه نمط، وإلا التقسيم بيكون عشوائي.")
    elif len(keys) < 10:
        print("  ⚠ المجموعات قليلة — النسب الفعلية ممكن تبعد عن 70/20/10")

    tr, va, te = a.split; total = len(items); G = len(keys)
    split_of, filled = {}, {"train": 0, "val": 0, "test": 0}
    if a.no_group or G > 100:
        # كثير مجموعات: نعبي بعدد الصور
        for k in keys:
            if filled["train"] < tr * total: s_ = "train"
            elif filled["val"] < va * total: s_ = "val"
            else: s_ = "test"
            for i in groups[k]: split_of[i] = s_
            filled[s_] += len(groups[k])
    else:
        # مجموعات قليلة (فيديوهات): نقسم بعدد الفيديوهات، ومتوازن بين الخلفيات (H/P) لو موجودة
        n_te = max(1, round(te * G)); n_va = max(1, round(va * G))
        def stratum(k):
            m = re.match(r"/?\s*\d+\s*([HP])", k)
            return m.group(1) if m else "_"
        strata = defaultdict(list)
        for k in keys: strata[stratum(k)].append(k)          # keys مخلوطة بالـ seed
        order = []                                             # round-robin بين الخلفيات
        while any(strata.values()):
            for st in sorted(strata):
                if strata[st]: order.append(strata[st].pop(0))
        assign = {k: ("test" if j < n_te else "val" if j < n_te + n_va else "train")
                  for j, k in enumerate(order)}
        for k, s_ in assign.items():
            for i in groups[k]: split_of[i] = s_
            filled[s_] += len(groups[k])
        for s_ in ("train", "val", "test"):
            vids = sorted(k.strip("/") for k, v in assign.items() if v == s_)
            print(f"  {s_:5s} ({len(vids)} فيديو): " + ", ".join(v[:12] for v in vids))

    # ---------- 5) الكتابة ----------
    if out.exists(): shutil.rmtree(out)
    for s in filled:
        (out / "images" / s).mkdir(parents=True); (out / "labels" / s).mkdir(parents=True)
    seen_names = Counter()
    neg_per = Counter()
    for i, (img, boxes) in enumerate(items):
        s = split_of[i]
        name = img.stem; seen_names[name] += 1
        if seen_names[name] > 1: name = f"{name}__{seen_names[name]}"   # أسماء مكررة
        shutil.copy2(img, out / "images" / s / f"{name}{img.suffix.lower()}")
        lines = [f"{c if a.keep_classes else 0} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
                 for c, x, y, w, h in boxes]
        (out / "labels" / s / f"{name}.txt").write_text("\n".join(lines))  # فاضي = سلبية
        if not boxes: neg_per[s] += 1

    names = ([f"class{c}" for c in sorted(cls_count)] if a.keep_classes else ["drop"])
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"nc: {len(names)}\nnames: {names}\n")

    print("\n=== التقسيم ===")
    for s, n in filled.items():
        print(f"{s:5s}: {n:5d} صورة ({100*n/total:4.1f}%) | سلبية: {neg_per[s]}")
    print(f"\n✔ جاهز: {out/'data.yaml'}")

    # ---------- صورة فحص ----------
    try:
        import cv2
        tiles = []
        pos = [it for it in items if it[1]]
        for img, boxes in random.sample(pos, min(12, len(pos))):
            im = cv2.imread(str(img))
            if im is None: continue
            H, W = im.shape[:2]
            for _, x, y, w, h in boxes:
                cv2.rectangle(im, (int((x-w/2)*W), int((y-h/2)*H)),
                              (int((x+w/2)*W), int((y+h/2)*H)), (0, 255, 0), max(2, W//200))
            tiles.append(cv2.resize(im, (320, 320)))
        while len(tiles) % 4: tiles.append(tiles[0] * 0)
        if tiles:
            rows = [cv2.hconcat(tiles[i:i+4]) for i in range(0, len(tiles), 4)]
            cv2.imwrite(str(out / "check_boxes.jpg"), cv2.vconcat(rows))
            print(f"✔ صورة فحص: {out/'check_boxes.jpg'} — افتحيها وتأكدي الصناديق على القطرات")
    except ImportError:
        print("(ثبّتي opencv-python عشان صورة الفحص)")


if __name__ == "__main__":
    main()
