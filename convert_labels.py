# -*- coding: utf-8 -*-
"""
Konversi label YOLOv8-segmentation -> YOLOv8-detection.

Kenapa perlu:
    Roboflow memberi label POLYGON (x1 y1 x2 y2 ... xn yn), sementara
    main.py memakai r.boxes.xyxy -> butuh KOTAK axis-aligned.
   /train langsung dengan data polygon akan trains model segmentasi,
    bukan deteksi.

Yang dilakukan:
    polygon -> bounding box (min/max x dan y) -> "class cx cy w h"

Label asli dicadangkan ke folder _label_asli_polygon supaya tidak hilang.
"""

import io
import os
import shutil
import sys

ROOT = r"D:\Chlandestine\dataset"
BACKUP_SUFFIX = "_label_asli_polygon"

# Kotak dengan luas < ini dibuang: annotasi titik/degenerate tidak berguna
# untuk deteksi dan hanya menambah noise.
MIN_SIDE = 0.005          # sisi minimal (dalam koordinat normalized)
MIN_AREA = 1e-5


def polygon_to_box(coords):
    xs = [float(coords[i]) for i in range(0, len(coords), 2)]
    ys = [float(coords[i]) for i in range(1, len(coords), 2)]
    if not xs:
        return None
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    w, h = xmax - xmin, ymax - ymin
    if w < MIN_SIDE or h < MIN_SIDE or (w * h) < MIN_AREA:
        return None
    return ((xmin + xmax) / 2.0, (ymin + ymax) / 2.0, w, h)


def convert_split(split, stats):
    ldir = os.path.join(ROOT, split, "labels")
    if not os.path.isdir(ldir):
        return
    bdir = os.path.join(ROOT, split, BACKUP_SUFFIX)
    os.makedirs(bdir, exist_ok=True)

    for name in os.listdir(ldir):
        if not name.endswith(".txt"):
            continue
        src = os.path.join(ldir, name)
        dst = os.path.join(bdir, name)

        text = io.open(src, encoding="utf-8").read()
        if not text.strip():
            # label kosong = gambar tanpa bubble, biarkan
            stats["empty"] += 1
            shutil.copy2(src, dst)
            continue

        shutil.copy2(src, dst)

        out_lines = []
        for line in text.splitlines():
            parts = line.split()
            if not parts:
                continue
            cls = parts[0]
            coords = parts[1:]

            if len(coords) == 4:
                # sudah format box (cls cx cy w h)
                out_lines.append(" ".join([cls] + coords))
                stats["already_box"] += 1
                continue

            box = polygon_to_box(coords)
            if box is None:
                stats["dropped_degenerate"] += 1
                continue
            cx, cy, w, h = box
            out_lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            stats["converted"] += 1

        io.open(src, "w", encoding="utf-8", newline="\n").write(
            "\n".join(out_lines) + ("\n" if out_lines else ""))

    print(f"  {split:<6} -> {len(os.listdir(ldir))} label diproses")


def main():
    stats = dict(converted=0, already_box=0, dropped_degenerate=0, empty=0)
    print(f"Mengonversi label di {ROOT}\n")
    for split in ("train", "valid", "test"):
        convert_split(split, stats)

    print()
    print(f"  polygon -> box  : {stats['converted']}")
    print(f"  sudah box       : {stats['already_box']}")
    print(f"  dibuang (kecil) : {stats['dropped_degenerate']}")
    print(f"  kosong          : {stats['empty']}")
    print()
    print(f"Label asli dicadangkan ke folder '{BACKUP_SUFFIX}' per split.")

    # validasi akhir: semua label harus 5 kolom
    bad = 0
    total = 0
    for split in ("train", "valid", "test"):
        ldir = os.path.join(ROOT, split, "labels")
        if not os.path.isdir(ldir):
            continue
        for name in os.listdir(ldir):
            if not name.endswith(".txt"):
                continue
            for line in io.open(os.path.join(ldir, name), encoding="utf-8").read().splitlines():
                if not line.strip():
                    continue
                total += 1
                if len(line.split()) != 5:
                    bad += 1

    print()
    print(f"Validasi: {total} objek, {bad} baris bukan format 5 kolom")
    if bad:
        print("MASIH ADA YANG SALAH -> periksa manual.")
        return 1
    print("Semua label siap untuk 'yolo detect train'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())