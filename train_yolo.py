# -*- coding: utf-8 -*-
"""
Latih ulang detektor bubble dengan YOLOv8s.

Kenapa bukan yolov8n seperti sebelumnya?
    train10 (sekarang) memakai yolov8n -> model terkecil. mAP50-95-nya cuma
    0.476, artinya kotak batasnya kurang presisi. Ini penting karena main.py
    membuat mask elips dari bounding box YOLO untuk inpaint, jadi kotak yang
    meleset = area gambar yang terhapus kelamaan.
    yolov8s jauh lebih akurat di lokalisasi dengan biaya inferensi masih kecil.

PENTING: device dipaksa ke GPU (device=0). Tanpa itu ultralytics bisa
memilih CPU dan RTX-mu nganggur.

Cara pakai:
    1. D:\Chlandestine\venv\Scripts\python.exe D:\Chlandestine\fetch_dataset.py
    2. D:\Chlandestine\venv\Scripts\python.exe D:\Chlandestine\train_yolo.py
"""

import os
import sys

import torch
from ultralytics import YOLO

DATA_YAML = r"D:\Chlandestine\dataset\data.yaml"
PROJECT   = r"D:\Chlandestine\runs\detect"
NAME      = "train_s"


def check_dataset(path: str) -> None:
    if os.path.isfile(path):
        print(f"Dataset  : {path}")
        return
    print(f"ERROR: data.yaml tidak ditemukan di:\n  {path}")
    print()
    print("Isi folder D:\\Chlandestine\\dataset yang diharapkan:")
    root = os.path.dirname(path)
    if os.path.isdir(root):
        for name in sorted(os.listdir(root))[:15]:
            print("   ", name)
    else:
        print("   (foldernya belum ada)")
    print()
    print("Cara dapatin dataset:")
    print("  1. Buka di browser:")
    print("     https://universe.roboflow.com/manga-speech-bubble-detection/dataset/v3")
    print("  2. 'Download Dataset' -> format 'YOLOv8'")
    print("     (JANGAN pilih 'YOLOv8 Oriented Bounding Box')")
    print(r"  3. Extract ke D:\Chlandestine\dataset")
    print()
    print("Download dari script/script tidak bisa: Roboflow pakai Cloudflare")
    print("challenge JavaScript yang hanya lolos lewat browser asli.")
    sys.exit(1)


MODEL_WEIGHTS = os.environ.get("YOLO_MODEL", "yolov8s.pt")
EPOCHS = int(os.environ.get("YOLO_EPOCHS", "80"))
PLOTS = os.environ.get("YOLO_PLOTS", "0") == "1"


def matplotlib_available() -> bool:
    """matplotlib sering diblokir Windows Smart App Control (DLL unsigned)."""
    try:
        import matplotlib  # noqa: F401
        return True
    except Exception:                              # noqa: BLE001
        return False


def report_gpu() -> None:
    print("=" * 60)
    print("GPU check")
    print("=" * 60)
    print(f"  torch          : {torch.__version__}")
    print(f"  cuda available : {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("  [!] CUDA tidak tersedia -- training akan jalan di CPU dan SANGAT LAMBAT.")
        print("      Pastikan torch cu128 terpasang:")
        print("      pip install torch torchvision --index-url "
              "https://download.pytorch.org/whl/cu128")
        sys.exit(1)
    props = torch.cuda.get_device_properties(0)
    print(f"  device 0       : {props.name}")
    print(f"  VRAM           : {props.total_memory / 1024**3:.1f} GB")
    print(f"  compute        : sm_{props.major}{props.minor}")
    print("=" * 60)
    print()

    if matplotlib_available():
        print("matplotlib     : OK (plot akan dibuat)")
    else:
        print("matplotlib     : DIBLOKIR Windows Smart App Control (DLL unsigned)")
        print("                 -> plots dimatikan, training tetap jalan")
        print("                 -> Impor hasil training akan lewat results.csv,")
        print("                    bukan lewat gambar plot.")
    print()


def main() -> int:
    report_gpu()
    check_dataset(DATA_YAML)

    last = os.path.join(PROJECT, NAME, "weights", "last.pt")
    resume = os.environ.get("YOLO_RESUME", "0") == "1" and os.path.isfile(last)

    if resume:
        print("Mode      : RESUME (lanjutkan dari epoch terakhir)")
        print(f"Checkpoint: {last}")
        print(f"Sisa epoch: {max(0, EPOCHS - 16)}")
        print()
        YOLO(last).train(resume=True, device=0)
    else:
        print(f"Dataset : {DATA_YAML}")
        print(f"Model   : {MODEL_WEIGHTS}")
        print(f"Output  : {PROJECT}\\{NAME}")
        print("Device  : 0  (GPU dipaksa)")
        print(f"Epoch   : {EPOCHS}")
        print()
        YOLO(MODEL_WEIGHTS).train(
            data=DATA_YAML,
            epochs=EPOCHS,
            imgsz=640,        # samakan dengan inference di main.py (IMG_SIZE)
            batch=16,
            device=0,         # <-- paksa GPU
            project=PROJECT,
            name=NAME,
            exist_ok=True,
            patience=20,      # early stop kalau tidak ada improvement
            workers=int(os.environ.get("YOLO_WORKERS", "0")),
            cache=False,
            plots=PLOTS,      # butuh matplotlib; sering diblokir Smart App Control
            # augmentasi: webtoon itu teks melengkung & bubble varied
            degrees=5.0,
            scale=0.5,
            translate=0.1,
            fliplr=0.5,
            hsv_v=0.4,
        )

    print()
    print("=" * 60)
    print("SELESAI")
    print("=" * 60)
    best = os.path.join(PROJECT, NAME, "weights", "best.pt")
    print(f"Model terbaik : {best}")
    print()
    print("Langkah berikutnya supaya main.py memakai model ini:")
    print(f'  setx YOLO_MODEL_PATH "{best}"')
    print()
    print("Atau tanpa set env, salin best.pt ke:")
    print(r"  D:\Chlandestine\Portable\weights\best.pt")
    print(r"  D:\Chlandestine\app\weights\best.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())