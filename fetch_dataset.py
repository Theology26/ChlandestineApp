# Download dataset Roboflow untuk re-train YOLOv8.
#
# Roboflow Universe MEBUTUHKAN API key - tidak bisa diunduh tanpa itu.
# Ambil key: https://universe.roboflow.com -> Settings -> API Keys
# (free tier cukup)
#
# Cara pakai:
#   1. Buka file ini, ganti YOUR_API_KEY di bawah
#   2. Jalankan:
#      D:\Chlandestine\venv\Scripts\python.exe D:\Chlandestine\fetch_dataset.py

import os
import sys
import zipfile

# API key dibaca dari environment variable lebih dulu, supaya tidak perlu
# menaruh secret di dalam file yang nanti ikut ter-commit ke GitHub:
#     setx ROBOFLOW_API_KEY "your-key"
_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "").strip()

PROJECT = "manga-speech-bubble-detection"
MODEL = "yolov8"                  # format export biasa (bukan OBB)
# Fallback kalau nomor versi tidak bisa dideteksi. Sebaiknya biarkan saja:
# nomor Below di auto-deteksi dari API.

DEST = r"D:\Chlandestine\dataset"


def pick_newest_version(project) -> int:
    """
    Jangan hardcode nomor versi. Dataset Roboflow sering di-reupload dan
    nomornya berubah; ambil versi terbaru yang tersedia untuk key ini.
    """
    try:
        nums = []
        for v in project.versions():
            raw = getattr(v, "version", None) or getattr(v, "id", None)
            name = (getattr(v, "name", "") or "").lower()
            n = None
            if isinstance(raw, int):
                n = raw
            elif isinstance(raw, str) and raw.isdigit():
                n = int(raw)
            elif "v" in name:
                digits = "".join(ch for ch in name.split("v")[-1] if ch.isdigit())
                n = int(digits) if digits else None
            if n is not None:
                nums.append(n)
        if nums:
            print(f"Versi tersedia: {sorted(nums)} -> pakai v{max(nums)}")
            return max(nums)
    except Exception as exc:                     # noqa: BLE001
        print(f"[warn] tidak bisa daftar versi: {exc}")
    return 3


def main() -> int:
    api_key = _API_KEY
    if not api_key:
        print("ERROR: Roboflow API key belum di-set.")
        print()
        print("Cara set (tidak menaruh secret di file yang di-commit):")
        print('    setx ROBOFLOW_API_KEY "your-key-here"')
        print()
        print("Atau isi langsung di baris API_KEY pada file ini")
        print("(tapi JANGAN di-commit kalau repo-nya publik).")
        print("Ambil key di: https://universe.roboflow.com -> Settings -> API Keys")
        return 1

    try:
        from roboflow import Roboflow
    except ImportError:
        print("ERROR: paket roboflow belum terpasang.")
        print("       pip install roboflow")
        return 1

    print(f"Mengunduh {PROJECT} ({MODEL}) -> {DEST}")
    try:
        rf = Roboflow(api_key=api_key)
        workspace = rf.workspace()
        project = workspace.project(PROJECT)
        version = project.version(pick_newest_version(project))
        # Parameter bernama 'model_format' di roboflow >= 0.1.44
        # (dulu 'model_type'). Coba keduanya biar jalan di versi manapun.
        try:
            version.download(model_format=MODEL, location=DEST, overwrite=True)
        except TypeError:
            version.download(model_type=MODEL, location=DEST)
    except Exception as exc:                     # noqa: BLE001
        print(f"GAGAL: {type(exc).__name__}: {exc}")
        print()
        if isinstance(exc, (zipfile.BadZipFile,)):
            print("=" * 66)
            print("  Penyebab: Cloudflare memblokir download dari script.")
            print("  Roboflow pakai challenge JavaScript yang hanya lolos lewat")
            print("  browser asli, jadi API key yang valid pun tetap kena 403.")
            print()
            print("  DOWNLOAD MANUAL (3 klik):")
            print("    1. Buka di browser:")
            print(f"       https://universe.roboflow.com/{PROJECT}/dataset/v3")
            print("    2. Klik 'Download Dataset' -> pilih format 'YOLOv8'")
            print("       (JANGAN pilih 'YOLOv8 Oriented Bounding Box')")
            print("    3. Extract, lalu pindahkan foldernya ke:")
            print(f"       {DEST}")
            print("       (pastikan ada file data.yaml di dalamnya)")
            print()
            print("  Setelah itu jalankan:")
            print('       D:\\Chlandestine\\venv\\Scripts\\python.exe D:\\Chlandestine\\train_yolo.py')
            print("=" * 66)
            return 1
        print("Kemungkinan sebab:")
        print("  - API key salah/kedaluwarsa")
        print("  - project slug bukan '%s' (cek URL di browser)" % PROJECT)
        return 1

    data_yaml = os.path.join(DEST, "data.yaml")
    print()
    if os.path.isfile(data_yaml):
        print(f"OK. data.yaml: {data_yaml}")
        print()
        print("Langkah berikutnya (latih ulang dengan YOLOv8s):")
        print(f'  yolo detect train model=yolov8s.pt data="{data_yaml}" '
              f"epochs=80 imgsz=640 batch=16 name=train_s")
    else:
        print("PERINGATAN: data.yaml tidak ditemukan, cek isi folder:")
        for root, _dirs, files in os.walk(DEST):
            for f in files[:5]:
                print("   ", os.path.join(root, f))
    return 0


if __name__ == "__main__":
    sys.exit(main())