import os
import io
import json
import re
import uuid
import base64
import unicodedata
from datetime import datetime
from typing import List, Tuple

from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import sys
import easyocr
import torch
import requests
import gc

# ==============================
# Paths (relatif BASE_DIR agar portabel antar mesin / share)
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Saat dibekukan jadi .exe, folder program tidak bisa ditulis (dan akan
# dibongkar PyInstaller ke temp). Arahkan output ke folder yang bisa ditulis.
# CHLANDESTINE_DATA (di-set oleh Mulai.bat) selalu dihormati.
_data_dir = os.environ.get("CHLANDESTINE_DATA", "").strip()
if not _data_dir:
    if getattr(sys, "frozen", False):
        _data_dir = os.path.join(
            os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "Chlandestine")
    else:
        _data_dir = BASE_DIR

LOCAL_EASYOCR_SRC = os.path.join(BASE_DIR, "EasyOCR")
if os.path.isdir(LOCAL_EASYOCR_SRC):
    sys.path.append(LOCAL_EASYOCR_SRC)

# ==============================
# Debug Info
# ==============================
print("CUDA Version:", torch.version.cuda)
print("GPU Available:", torch.cuda.is_available())

# ==============================
# FastAPI Setup
# ==============================
app = FastAPI()

@app.get("/favicon.ico")
async def favicon():
    logo = os.path.join(BASE_DIR, "assets", "img", "logo.png")
    if os.path.isfile(logo):
        return FileResponse(logo)
    return Response(status_code=204)

_allowed_origins = [
    o.strip()
    for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=(_allowed_origins != ["*"]),
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# Paths & Models
# ==============================
def find_yolo_weights() -> str:
    """
    Cari best.pt. Urutan:
      1. env YOLO_MODEL_PATH
      2. folder 'weights' di samping .exe (mode portable)
      3. _MEIPASS / BASE_DIR (mode dev / dibekukan PyInstaller)
    """
    env = os.environ.get("YOLO_MODEL_PATH", "").strip()
    rel = os.path.join("runs", "detect", "train10", "weights", "best.pt")
    cands = []
    if env:
        cands.append(env)
    if getattr(sys, "frozen", False):
        exe_dir = os.path.dirname(sys.executable)
        cands.append(os.path.join(exe_dir, "weights", "best.pt"))
    if getattr(sys, "_MEIPASS", None):
        cands.append(os.path.join(sys._MEIPASS, rel))
        cands.append(os.path.join(sys._MEIPASS, "weights", "best.pt"))
    cands.append(os.path.join(BASE_DIR, rel))
    cands.append(os.path.join(BASE_DIR, "weights", "best.pt"))
    # layout portable: app/ dan weights/ adalah saudara, bukan anak
    cands.append(os.path.join(os.path.dirname(BASE_DIR), "weights", "best.pt"))
    # layout app/ satu level di bawah root (Portable\app -> root\weights)
    cands.append(
        os.path.join(os.path.dirname(os.path.dirname(BASE_DIR)), "weights", "best.pt"))

    for c in cands:
        if c and os.path.isfile(c):
            return c
    raise FileNotFoundError(
        "Model YOLO (best.pt) tidak ditemukan. Dicoba di:\n  "
        + "\n  ".join(cands)
        + "\nSet env YOLO_MODEL_PATH atau taruh best.pt di folder 'weights'."
    )


YOLO_MODEL_PATH = find_yolo_weights()
yolo_model = YOLO(YOLO_MODEL_PATH)

DEBUG_DIR = os.path.join(_data_dir, "debug")
os.makedirs(DEBUG_DIR, exist_ok=True)
app.mount("/debug", StaticFiles(directory=DEBUG_DIR), name="debug")

DEBUG_LOG = os.path.join(_data_dir, "ocr_debug.log")
def dlog(msg: str):
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(DEBUG_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{stamp}] {msg}\n")
    print(msg)


@app.on_event("startup")
async def _startup_checks():
    dlog(f"📁 BASE_DIR       : {BASE_DIR}")
    dlog(f"🤖 YOLO model     : {YOLO_MODEL_PATH}")
    dlog(f"🖥  EasyOCR GPU    : {use_gpu}")
    if TRANSLATE_ENABLED:
        dlog(f"🤖 Translasi lokal : {TRANSLATE_MODEL} ({NLLB_SRC} -> {NLLB_TGT})")
    else:
        dlog("⚠️  Translasi       : NONAKTIF (TRANSLATE_ENABLED=0) → pakai placeholder [TL]")
    model_dir = find_easyocr_models()
    if model_dir:
        dlog(f"🧠 EasyOCR models : {model_dir} (lokal, offline)")
    else:
        dlog("🧠 EasyOCR models : tidak ditemukan -> akan download ke cache EasyOCR")
    try:
        r = requests.get(IOPAINT_URL, timeout=3)
        dlog(f"🎨 IOPaint        : OK (HTTP {r.status_code}) {IOPAINT_URL}")
    except Exception as e:
        dlog(
            f"⚠️  IOPaint TIDAK TERHUBUNG di {IOPAINT_URL} ({type(e).__name__}). "
            "Inpaint akan dilewati, hasil keluar apa adanya."
        )

# ==============================
# Config
# ==============================
use_gpu = torch.cuda.is_available()
print(f"EasyOCR GPU mode: {use_gpu}")

MIN_LINE_CONF = float(os.environ.get("MIN_LINE_CONF", "0.30"))  # Minimal conf per baris dari EasyOCR
YOLO_CONF    = 0.49      # Minimal conf YOLO bubble
# Samakan dengan imgsz saat training. Train10 dilatih di 640 (lihat
# runs/detect/train10/args.yaml), dan mAP diukur di 640 -- inference di
# 1024 hanya memperlambat tanpa bukti membaik.
IMG_SIZE     = int(os.environ.get("IMG_SIZE", "640"))
IOPAINT_URL  = os.environ.get("IOPAINT_URL", "http://127.0.0.1:8080")
STRIDE       = 64        # Padding kelipatan 64 agar server inpaint tidak mem-pad sendiri

# Target_char_px: tinggi glyph minimum yang desirable untuk EasyOCR.
# Di bawah ini akurasi CJK/Hangul anjlok karena stroke padat terlalu kecil.
TARGET_CHAR_PX = int(os.environ.get("TARGET_CHAR_PX", "32"))
MAX_UPSCALE    = float(os.environ.get("MAX_UPSCALE", "3.0"))

# ==============================
# Deteksi Script (Unicode)
# ==============================
# Semua blok Hangul. Kalau salah satu hilang, teks KoreaUNG akan dianggap bukan Korea.
HANGUL_RANGES = (
    (0x1100, 0x11FF),   # Hangul Jamo
    (0x3130, 0x318F),   # Hangul Compatibility Jamo  (mis. 한글이写成 ㅎㅏㄴ)
    (0xA960, 0xA97F),   # Hangul Jamo Extended-A
    (0xAC00, 0xD7A3),   # Hangul Syllables            (bulkips Korea)
    (0xD7B0, 0xD7FF),   # Hangul Jamo Extended-B
    (0xFFA0, 0xFFDC),   # Halfwidth Hangul Jamo
)

# Kana = Jepang. WAJIB dipisah dari Hangul, kalau tidak teks Jepang
# akan dilaporkan sebagai Korea.
KANA_RANGES = (
    (0x3040, 0x30FF),   # Hiragana + Katakana
    (0x31F0, 0x31FF),   # Katakana Phonetic Extensions
    (0xFF66, 0xFF9D),   # Halfwidth Katakana
)

# Han = CJK/Hanja. Dipakai Mandarin & kanji Jepang, BUKAN Hangul.
HAN_RANGES = (
    (0x4E00, 0x9FFF),   # CJK Unified Ideographs
    (0x3400, 0x4DBF),   # CJK Extension A
    (0xF900, 0xFAFF),   # CJK Compatibility Ideographs
)

# Minimal porsi script untuk dianggap dominan.
SCRIPT_MIN_RATIO = float(os.environ.get("SCRIPT_MIN_RATIO", "0.15"))


def _count_in_ranges(text: str, ranges) -> int:
    n = 0
    for ch in text:
        cp = ord(ch)
        for lo, hi in ranges:
            if lo <= cp <= hi:
                n += 1
                break
    return n


def _significant(text: str) -> str:
    """Buang whitespace + simbol, sisakan huruf/angka."""
    return "".join(
        ch for ch in text
        if not ch.isspace() and unicodedata.category(ch)[0] not in ("P", "S", "C", "Z")
    )


def classify_script(text: str) -> dict:
    """
    Tentukan script dominan dari teks hasil OCR.
    Return dict: {script, lang, hangul_ratio, kana_ratio, han_ratio, latin_ratio, total}
    """
    core = _significant(text)
    total = len(core)
    out = {
        "script": "unknown",
        "lang": None,
        "hangul_ratio": 0.0,
        "kana_ratio": 0.0,
        "han_ratio": 0.0,
        "latin_ratio": 0.0,
        "total": total,
    }
    if total == 0:
        return out

    hangul = _count_in_ranges(core, HANGUL_RANGES)
    kana   = _count_in_ranges(core, KANA_RANGES)
    han    = _count_in_ranges(core, HAN_RANGES)
    latin  = sum(1 for ch in core if ch.isascii() and ch.isalpha())

    out["hangul_ratio"] = hangul / total
    out["kana_ratio"]   = kana / total
    out["han_ratio"]    = han / total
    out["latin_ratio"]  = latin / total

    # Urutan penting: Hangul dulu, karena teks Korea bisa contain Hanja.
    if out["hangul_ratio"] >= SCRIPT_MIN_RATIO:
        out["script"], out["lang"] = "hangul", "ko"
    elif out["kana_ratio"] >= SCRIPT_MIN_RATIO:
        out["script"], out["lang"] = "kana", "ja"
    elif out["han_ratio"] >= SCRIPT_MIN_RATIO:
        out["script"], out["lang"] = "han", "zh"
    elif out["latin_ratio"] >= SCRIPT_MIN_RATIO:
        out["script"], out["lang"] = "latin", "en"
    else:
        out["script"], out["lang"] = "unknown", None

    return out


def classify_results(results) -> dict:
    """Wrapper classify_script untuk [(poly, text, conf), ...] dari EasyOCR."""
    joined = " ".join(t for _, t, _ in results if t and t.strip())
    info = classify_script(joined)
    info["text"] = joined
    return info

# ==============================
# Lazy EasyOCR Loader (CACHE)
# ==============================
_easyocr_cache = {}

# Model EasyOCR (craft + korean). Dicari di folder portable dulu supaya .exe
# tidak butuh unduh, lalu fallback ke cache EasyOCR.
def _easyocr_model_dirs():
    """
    Model EasyOCR (craft + korean). Urutan:
      <exe>/ocrmodels  ->  <BASE_DIR>/ocrmodels  ->  <BASE_DIR>/EasyOCR/saved_models
    Layout portable menaruh ocrmodels sebagai saudara folder app/.
    """
    roots = []
    if getattr(sys, "frozen", False):
        roots.append(os.path.join(os.path.dirname(sys.executable), "ocrmodels"))
    roots.append(os.path.join(BASE_DIR, "ocrmodels"))
    roots.append(os.path.join(os.path.dirname(BASE_DIR), "ocrmodels"))
    # grandparent: layout di mana app/ berada satu level di bawah root,
    # mis. D:\Chlandestine\Portable\app  ->  D:\Chlandestine\ocrmodels
    roots.append(os.path.join(os.path.dirname(os.path.dirname(BASE_DIR)), "ocrmodels"))
    roots.append(os.path.join(BASE_DIR, "EasyOCR", "saved_models"))
    return [r for r in roots if os.path.isdir(r)]


def find_easyocr_models() -> str | None:
    """Folder EasyOCR yang dipakai sebagai sumber model lokal."""
    for r in _easyocr_model_dirs():
        if os.path.isfile(os.path.join(r, "craft_mlt_25k.pth")):
            return r
    return None


def get_easyocr_reader(lang_hint: str):
    """
    Muat EasyOCR reader hanya saat diperlukan, cache berdasarkan bahasa.
    lang_hint: "auto" | "ja" | "zh"/"ch" | "ko" | lainnya
    """
    # Normalisasi key cache
    key = lang_hint or "auto"
    if key == "auto":
        key = "ko"  # default fallback

    if key in _easyocr_cache:
        return _easyocr_cache[key]

    if key.startswith("ja"):
        langs = ['ja', 'en']
    elif key.startswith("zh") or key.startswith("ch"):
        langs = ['ch_sim', 'en']
    elif key.startswith("ko"):
        langs = ['ko', 'en']
    elif key.startswith("en"):
        langs = ['en']
    else:
        # fallback ke Korea jika tidak jelas
        langs = ['ko', 'en']

    dlog(f"🧠 Loading EasyOCR reader for langs={langs} (gpu={use_gpu})")
    reader_kwargs = {"gpu": use_gpu}
    model_dir = find_easyocr_models()
    if model_dir:
        reader_kwargs["user_network_directory"] = model_dir
    reader = easyocr.Reader(langs, **reader_kwargs)
    _easyocr_cache[key] = reader
    return reader

# ==============================
# Utils (BBox, Mask, IOU, Merge)
# ==============================
def clamp_bbox(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))
    if x2 <= x1:
        x2 = min(w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(h - 1, y1 + 1)
    return x1, y1, x2, y2

def polygon_to_mask(img_shape, poly, pad=0, shrink=2):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    pts = np.array(poly, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    if pad > 0:
        kernel = np.ones((pad, pad), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
    if shrink > 0:
        kernel = np.ones((shrink, shrink), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
    return mask

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter_w = max(0, xB - xA); inter_h = max(0, yB - yA)
    interArea = inter_w * inter_h
    if interArea == 0: return 0.0
    boxAArea = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

def merge_overlapping_boxes_keep_smallest(boxes: List[List[float]], iou_threshold=0.5):
    if not boxes: return []
    n = len(boxes)
    visited = [False]*n
    clusters = []
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if iou(boxes[i], boxes[j]) > iou_threshold:
                adj[i].append(j); adj[j].append(i)
    def dfs(idx, cluster):
        visited[idx] = True; cluster.append(idx)
        for nei in adj[idx]:
            if not visited[nei]: dfs(nei, cluster)
    for i in range(n):
        if not visited[i]:
            cluster=[]; dfs(i, cluster); clusters.append(cluster)
    merged=[]
    for c in clusters:
        smallest_idx=c[0]
        sx1, sy1, sx2, sy2, _ = boxes[smallest_idx]
        smallest_area=(sx2-sx1)*(sy2-sy1)
        for idx in c[1:]:
            x1,y1,x2,y2,_=boxes[idx]
            area=(x2-x1)*(y2-y1)
            if area<smallest_area:
                smallest_area=area; smallest_idx=idx
        merged.append(boxes[smallest_idx])
    return merged

def poly_in_box(poly, box, margin=10):
    x1, y1, x2, y2 = box
    for (px, py) in poly:
        if not (x1 - margin <= px <= x2 + margin and y1 - margin <= py <= y2 + margin):
            return False
    return True

# ==============================
# 🧩 Combined Preprocessing Webtoon (+ auto threshold switch)
# ==============================
def _choose_threshold(gray_blur: np.ndarray):
    """Pilih otomatis BINARY vs BINARY_INV dengan heuristik proporsi piksel hitam & edge."""
    # dua kandidat
    thr_inv = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY_INV, 15, 5)
    thr_norm = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, 15, 5)

    def score(mask):
        # hitung proporsi foreground (hitam utk BINARY, putih utk INV kita balikin ke "teks=255")
        # Normalisasi: kita ingin 0.5% - 15% area sebagai teks
        fg = mask if mask.dtype==np.uint8 else mask.astype(np.uint8)
        ratio = float(np.count_nonzero(fg)) / fg.size
        # edge density (lebih banyak edge -> cenderung huruf)
        edges = cv2.Canny(fg, 50, 150)
        edge_ratio = float(np.count_nonzero(edges)) / edges.size
        # skor: penalti jika terlalu sedikit/terlalu banyak, tambah edge
        target_low, target_high = 0.005, 0.15
        penalty = 0.0
        if ratio < target_low: penalty += (target_low - ratio) * 10
        if ratio > target_high: penalty += (ratio - target_high) * 10
        return edge_ratio - penalty

    # untuk konsistensi, samakan "teks=255" pada kedua kandidat
    # thr_inv sudah teks=255; thr_norm teks=0 → invert
    thr_norm_inv = cv2.bitwise_not(thr_norm)

    s_inv = score(thr_inv)
    s_norm = score(thr_norm_inv)
    return (thr_inv, "INV", s_inv) if s_inv >= s_norm else (thr_norm_inv, "BIN", s_norm)

def preprocess_webtoon_combined(img_bgr: np.ndarray, auto_switch: bool=True) -> np.ndarray:
    """
    Combined preprocessing untuk manhwa/webtoon:
    - Bilateral denoise (jaga edge)
    - CLAHE (tingkatkan kontras)
    - Grayscale + Adaptive Threshold (auto pilih normal/invert)
    - Morph clean
    - Sharpen
    Mengembalikan 3-channel BGR agar aman untuk EasyOCR & pipeline lain.
    """
    # 1) Denoise lembut
    denoised = cv2.bilateralFilter(img_bgr, 9, 75, 75)

    # 2) CLAHE di channel L
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab = cv2.merge((l2, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 3) Grayscale + blur
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # 4) Threshold
    if auto_switch:
        thr, mode, sc = _choose_threshold(blur)
    else:
        thr = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY_INV, 15, 5)

    # 5) Morph clean
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 6) Sharpen
    kernel_sharp = np.array([[0, -1, 0],
                             [-1, 5, -1],
                             [0, -1, 0]])
    sharpened = cv2.filter2D(cleaned, -1, kernel_sharp)

    # kembalikan ke 3-channel BGR
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

# ==============================
# Utils (OCR & Scaling)
# ==============================
def _ocr_once(img_rgb, lang_key):
    reader = get_easyocr_reader(lang_key)
    return reader.readtext(img_rgb, detail=1, paragraph=False)  # [(poly, text, conf)]


def _plan_scale(h: int, w: int) -> float:
    """
    Tentukan faktor upscale dari ukuran crop.

    Versi lama hanya upscale kalau h<20 atau w<20, padahal bubble webtoon
    jarang没见过 sekecil itu -- teks di dalamnya tetap kecil dan tidak pernah
    di-upscale. Sekarang patokannya tinggi glyph minimum (TARGET_CHAR_PX),
    dan crop bubble diasumsikan ~1-3 baris.
    """
    if h <= 0 or w <= 0:
        return 1.0
    est_char_px = h / 2.0          # asumsi bubble ~2 baris teks
    if est_char_px >= TARGET_CHAR_PX:
        return 1.0
    scale = TARGET_CHAR_PX / est_char_px
    return float(min(max(scale, 1.0), MAX_UPSCALE))


def safe_easyocr(img_bgr, lang_hint="auto"):
    """
    Jalankan EasyOCR dengan auto-upscale + deteksi script dua tahap.

    Tahap 1: baca pakai reader default (Korea).
    Tahap 2: kalau hasil baca didominasi script lain (kana/Han/latin),
             ulangi dengan reader bahasa itu -- ini yang dulu tidak ada,
             sehingga manga Jepang/Cina dipaksa dibaca sebagai Korea.

    Return: ([(poly, text, conf), ...] koordinat ukuran ASLI), scale_x, scale_y, script_info
    """
    if img_bgr is None or img_bgr.size == 0:
        return [], 1.0, 1.0, {"script": "unknown", "lang": None, "total": 0}
    h, w = img_bgr.shape[:2]
    if h < 5 or w < 5:
        return [], 1.0, 1.0, {"script": "unknown", "lang": None, "total": 0}

    scale = _plan_scale(h, w)
    scale_x = scale_y = scale
    work = img_bgr
    if scale != 1.0:
        work = cv2.resize(work, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    img_rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)

    def _to_orig(results):
        out = []
        for poly, text, conf in results:
            if scale != 1.0:
                poly = [(p[0] / scale, p[1] / scale) for p in poly]
            out.append((poly, text, conf))
        return out

    try:
        results = _ocr_once(img_rgb, lang_hint)
    except Exception as e:
        dlog(f"EasyOCR exception: {e}")
        return [], scale_x, scale_y, {"script": "unknown", "lang": None, "total": 0}

    info = classify_results(results)

    # Retry sekali kalau script dominant bukan Hangul tapi reader-nya Korea.
    if lang_hint == "auto" and info["total"] > 0:
        target = info["lang"]
        if target and target != "ko":
            dlog(
                f"🔎 script terdeteksi '{info['script']}' "
                f"(hangul={info['hangul_ratio']:.2f} kana={info['kana_ratio']:.2f} "
                f"han={info['han_ratio']:.2f} latin={info['latin_ratio']:.2f}) "
                f"→ retry OCR pakai '{target}'"
            )
            try:
                retry = _ocr_once(img_rgb, target)
                retry_info = classify_results(retry)
                # Terima retry hanya kalau hasil retry lebih "meyakinkan":
                # script-nya cocok dengan bahasa yang dipakai.
                if retry_info["total"] > 0 and retry_info["script"] == info["script"]:
                    results = retry
                    info = retry_info
                    info["retried"] = True
            except Exception as e:
                dlog(f"EasyOCR retry ({target}) gagal: {e}")

    if scale != 1.0:
        dlog(f"🔍 OCR upscale x{scale:.2f} (crop {w}x{h})")

    return _to_orig(results), scale_x, scale_y, info

def fake_translate(text: str) -> str:
    return f"[TL] {text}"


# ==============================
# Translasi Korea -> Indonesia
# NLLB-200 (Meta) jalan LOKAL di GPU — tanpa API key, tanpa biaya,
# tidak ada data yang keluar dari mesin.
# ==============================
TRANSLATE_ENABLED = os.environ.get("TRANSLATE_ENABLED", "1").strip().lower() not in ("0", "false", "no", "off")
TRANSLATE_MODEL  = os.environ.get("TRANSLATE_MODEL", "facebook/nllb-200-distilled-1.3B")
# Cache model DI DISK LOKAL. Kalau diisi path di Z:/ (network share) download
# 2.4GB akan tersendat. Kosongkan -> pakai default HF (~/.cache/huggingface).
TRANSLATE_CACHE  = os.environ.get("TRANSLATE_CACHE_DIR", "").strip()
NLLB_SRC = "kor_Latn"    # Korean
NLLB_TGT = "ind_Latn"    # Indonesian
TRANSLATE_BATCH   = int(os.environ.get("TRANSLATE_BATCH", "8"))
TRANSLATE_BEAMS   = int(os.environ.get("TRANSLATE_BEAMS", "4"))
TRANSLATE_FP16    = os.environ.get("TRANSLATE_FP16", "1").strip().lower() not in ("0", "false", "no", "off")

# --- Inpainting ---
# "opencv" (default): cepat, tanpa model, hasil paling rata untuk bubble datar.
# "iopaint": pakai server IOPaint (model manga) -- perlu yang lebih kompleks
#            di belakang bubble, tapi bisa mengarang tekstur.
INPAINT_METHOD       = os.environ.get("INPAINT_METHOD", "opencv").strip().lower()
INPAINT_OPENCV_RADIUS = float(os.environ.get("INPAINT_OPENCV_RADIUS", "5"))
INPAINT_OPENCV_MODE   = os.environ.get("INPAINT_OPENCV_MODE", "ns").strip().lower()
# Mask isian bubble: elips shrunk terhadap bbox + inset agar garis bubble aman.
# Angka ini hasil sweep pada bubble nyata (lihat catatan di process_image).
BUBBLE_FILL_SCALE = float(os.environ.get("BUBBLE_FILL_SCALE", "0.72"))
BUBBLE_FILL_INSET = int(os.environ.get("BUBBLE_FILL_INSET", "16"))
# Erosi tambahan supaya mask tidak pernah menyentuh garis bubble.
BUBBLE_FILL_ERODE = int(os.environ.get("BUBBLE_FILL_ERODE", "3"))
# Rasio tinggi/lebar yang dianggap "garis", bukan teks. EasyOCR sering salah baca
# garis penghubung bubble sebagai teks; kalau di-inpaint hasilnya coretan jelek.
GLOBAL_TAIL_ASPECT = float(os.environ.get("GLOBAL_TAIL_ASPECT", "2.5"))
MAX_NEW_TOKENS    = int(os.environ.get("TRANSLATE_MAX_NEW_TOKENS", "256"))

# Folder tempat model translasi ditaruh manual (dipakai kalau nama model
# bukan path yang ada, mis. karena Windows Defender memblokir auto-download).
# Urutan: folder di samping .exe (portable) -> D:\Chlandestine\models -> Z:\models
def _default_model_dirs() -> str:
    roots = []
    if getattr(sys, "frozen", False):
        # jalan dari Chlandestine.exe -> cari di folder yang sama
        roots.append(os.path.join(os.path.dirname(sys.executable), "models"))
    roots.append(os.path.join(BASE_DIR, "models"))
    roots.append(os.path.join(os.path.dirname(BASE_DIR), "models"))
    roots.append(r"D:\Chlandestine\models")   # fallback terakhir
    return ";".join(roots)


LOCAL_MODEL_DIRS = [
    d.strip() for d in os.environ.get(
        "TRANSLATE_MODEL_DIR", _default_model_dirs()
    ).split(";") if d.strip()
]


def resolve_translate_model(name: str) -> str:
    """
    Kalau `name` sudah folder yang valid -> pakai langsung.
    Kalau tidak, cari di LOCAL_MODEL_DIRS. Kalau ketemu folder yang di dalamnya
    ada satu sub-folder (hasil extract zip), turun satu level.
    """
    if os.path.isdir(name):
        return name

    # repo id "facebook/nllb-xxx" -> coba juga nama folder polosnya,
    # supaya hasil download manual ke models\nllb-xxx ketemu.
    leaf = name.rstrip("/").split("/")[-1]

    for base_dir in LOCAL_MODEL_DIRS:
        for sub in (name, leaf):
            cand = os.path.join(base_dir, *sub.split("/"))
            if os.path.isdir(cand):
                # handle hasil extract zip: <models>/<nama>/<nama>/config.json
                if not os.path.isfile(os.path.join(cand, "config.json")):
                    subs = [
                        os.path.join(cand, d) for d in os.listdir(cand)
                        if os.path.isfile(os.path.join(cand, d, "config.json"))
                    ]
                    if len(subs) == 1:
                        cand = subs[0]
                if os.path.isfile(os.path.join(cand, "config.json")):
                    return cand

    return name

_translate_cache: dict[str, str] = {}
_translator = None
_translator_tried = False


def get_translator():
    """Muat model translasi lokal. None kalau gagal / dinonaktifkan."""
    global _translator, _translator_tried
    if _translator_tried:
        return _translator
    _translator_tried = True

    if not TRANSLATE_ENABLED:
        dlog("ℹ️ TRANSLATE_ENABLED=0 → translasi dilewati")
        return None

    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        cache_kw = {"cache_dir": TRANSLATE_CACHE} if TRANSLATE_CACHE else {}
        if TRANSLATE_CACHE:
            os.makedirs(TRANSLATE_CACHE, exist_ok=True)

        model_name = resolve_translate_model(TRANSLATE_MODEL)
        is_local = os.path.isdir(model_name)
        # Kalau ketemu folder lokal, paksa OFFLINE supaya tidak pernah
        # menyentuh HuggingFace sama sekali.
        if is_local:
            cache_kw["local_files_only"] = True
            dlog(f"📁 Model translasi dari folder lokal (offline): {model_name}")
        else:
            dlog(f"⚠️  '{model_name}' bukan folder lokal, akan coba download dari HuggingFace...")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        # fp16 di GPU: VRAM turun ~2x (1.3B = 5.2GB -> 2.6GB) dan lebih cepat.
        use_fp16 = device == "cuda" and TRANSLATE_FP16
        if use_fp16:
            cache_kw["torch_dtype"] = torch.float16

        t0 = datetime.now()
        dlog(f"🤖 Memuat model translasi '{model_name}' ({'fp16' if use_fp16 else 'fp32'}) ...")

        tok = AutoTokenizer.from_pretrained(
            model_name, src_lang=NLLB_SRC, **cache_kw
        )
        mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name, **cache_kw)
        tok.tgt_lang = NLLB_TGT
        mdl.resize_token_embeddings(len(tok))

        mdl.to(device).eval()

        bos_id = tok.convert_tokens_to_ids(NLLB_TGT)
        _translator = (tok, mdl, device, bos_id)
        dlog(
            f"✅ Model translasi siap di {device} dalam "
            f"{(datetime.now() - t0).total_seconds():.1f}s"
        )
    except Exception as e:
        dlog(f"❌ Gagal memuat model translasi: {type(e).__name__}: {e}")
        _translator = None
    return _translator


# Sapaan Korea. NLLB menerjemahkan harfiah, mis. 형 -> "Saudara".
# Karena outputnya sudah bahasa Indonesia, kata Korea tidak ada lagi di
# hasil terjemahan -- jadi kata KOREA di sumber dipakai sebagai trigger,
# lalu terjemahan harfiahnya yang diganti.
# (format: kata Korea -> (terjemahan harfiah yang mungkin muncul, bentuk natural))
_SAPAAN_HARFIAH = ["Saudara", "saudara"]
_SAPAAN_FIX = {
    "형": (_SAPAAN_HARFIAH + ["Kakak laki-laki", "Abang"], "Kakak"),
    "오빠": (_SAPAAN_HARFIAH + ["Kakak laki-laki", "Abang"], "Kakak"),
    "누나": (_SAPAAN_HARFIAH + ["Kakak perempuan"], "Kakak"),
    "언니": (_SAPAAN_HARFIAH + ["Kakak perempuan"], "Kakak"),
    "동생": (_SAPAAN_HARFIAH, "Adik"),
    "선배": (["Senior", "senior"], "Kakak senior"),
}

# Frasa kaku khas hasil MT yang sering muncul di dialog.
_FRASA_KAKU = {
    "Jangan sekali-kali": "Jangan boleh",
    "Sekali-kali tidak": "Tak boleh",
    "tidak dapat dihindari": "pasti terjadi",
}


def polish_translation(src_ko: str, text: str) -> str:
    """
    Rapikan output translator:
      - sapaan Korea yang diterjemahkan harfiah ("형" -> "Saudara" -> "Kakak")
      - register formal "Anda" -> "kamu" (dialog manga biasanya informal)
      - pengulangan kata ("makanlah dan makanlah", "rumah kami adalah rumah kami")
      - frasa kaku khas hasil machine translation
    """
    if not text:
        return text
    out = text.strip()

    # 1. sapaan -- trigger dari kata Korea di teks sumber
    if src_ko:
        for ko, (bad_forms, good) in _SAPAAN_FIX.items():
            if ko in src_ko:
                for b in bad_forms:
                    out = re.sub(rf"\b{re.escape(b)}\b", good, out)

    # 2. register formal -> informal
    out = re.sub(r"\bAnda\b", "kamu", out)

    # 3. frasa kaku MT -> natural
    for kaku, natural in _FRASA_KAKU.items():
        out = re.sub(rf"\b{re.escape(kaku)}\b", natural, out)

    # 4. buang pengulangan: "X adalah X", "X dan X", "X atau X", "X, X"
    out = re.sub(
        r"\b([\w-]+(?:\s+[\w-]+)?)\s+(?:adalah|dan|atau|itu)\s+(?:[\w-]+\s+)?\1\b",
        r"\1", out, flags=re.IGNORECASE)
    out = re.sub(r"\b([\w-]{3,})\s*,\s*\1\b", r"\1", out, flags=re.IGNORECASE)

    # 5. rapikan spasi / tanda baca
    out = re.sub(r"\s{2,}", " ", out).strip()
    out = re.sub(r"\s+([,.!?])", r"\1", out)
    return out


def translate_ko_to_id(texts: list[str]) -> dict[str, str]:
    """
    Terjemahkan banyak teks Korea ke Indonesia dengan model lokal.

    Batch (bukan per-teks) supaya cepat di GPU.
    Cache in-memory supaya teks yang sama tidak di-inference dua kali.
    Return: {teks_asal: terjemahan}
    """
    uniq, seen = [], set()
    for t in texts or []:
        t = (t or "").strip()
        if t and t not in seen:
            seen.add(t)
            uniq.append(t)
    if not uniq:
        return {}

    result = {t: _translate_cache[t] for t in uniq if t in _translate_cache}
    todo = [t for t in uniq if t not in _translate_cache]
    if not todo:
        return result

    tr = get_translator()
    if tr is None:
        return result
    tok, mdl, device, bos_id = tr

    try:
        for i in range(0, len(todo), TRANSLATE_BATCH):
            chunk = todo[i:i + TRANSLATE_BATCH]
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=512)
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                gen = mdl.generate(
                    **enc,
                    forced_bos_token_id=bos_id,
                    max_new_tokens=MAX_NEW_TOKENS,
                    num_beams=TRANSLATE_BEAMS,
                )
            decoded = tok.batch_decode(gen, skip_special_tokens=True)
            for src, dst in zip(chunk, decoded):
                if dst and dst.strip():
                    _translate_cache[src] = polish_translation(src, dst)
    except Exception as e:
        dlog(f"❌ Translasi gagal: {type(e).__name__}: {e}")
        return result

    result.update({t: _translate_cache[t] for t in todo if t in _translate_cache})
    dlog(f"🌐 Terjemahkan {len(result)}/{len(uniq)} baris ke Indonesia (lokal)")
    return result


def apply_translations(bubbles: list[dict]) -> int:
    """Isi field 'translation' pada bubble & lines. Return jumlah baris terjemahan."""
    texts = []
    for b in bubbles:
        for ln in b.get("lines", []):
            if (ln.get("text") or "").strip():
                texts.append(ln["text"])
    if not texts:
        return 0

    mapping = translate_ko_to_id(texts)
    if not mapping:
        return 0

    filled = 0
    for b in bubbles:
        joined = []
        for ln in b.get("lines", []):
            dst = mapping.get(ln.get("text"))
            if dst:
                ln["translation"] = dst
                joined.append(dst)
                filled += 1
        if joined:
            b["translation"] = "\n".join(joined)
    return filled

# ==============================
# Utils (Padding/Unpadding & Size Guard)
# ==============================
def _pad_to_stride(img: np.ndarray, mask: np.ndarray|None = None, stride: int = STRIDE):
    """
    Pad img (dan mask jika ada) ke kelipatan stride, padding simetris.
    Gambar: BORDER_REFLECT_101, Mask: BORDER_CONSTANT(0).
    Return: (img_padded, mask_padded_or_None, (top, bottom, left, right))
    """
    h, w = img.shape[:2]
    pad_h = (stride - (h % stride)) % stride
    pad_w = (stride - (w % stride)) % stride

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    img_p = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REFLECT_101)
    m_p = None
    if mask is not None:
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = (mask > 127).astype(np.uint8) * 255
        m_p = cv2.copyMakeBorder(mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    return img_p, m_p, (top, bottom, left, right)

def _unpad(img: np.ndarray, pads: tuple[int,int,int,int]) -> np.ndarray:
    top, bottom, left, right = pads
    h, w = img.shape[:2]
    y2 = h - bottom if bottom > 0 else h
    x2 = w - right  if right  > 0 else w
    return img[top:y2, left:x2]

def _ensure_same_size(out_img: np.ndarray, ref_img: np.ndarray, debug_prefix: str|None=None):
    rh, rw = ref_img.shape[:2]
    if out_img is None:
        return ref_img
    oh, ow = out_img.shape[:2]
    if (oh, ow) != (rh, rw):
        dlog(f"⚠ IOPaint returned size {ow}x{oh}, expected {rw}x{rh}. Force-resizing back.")
        if debug_prefix:
            cv2.imwrite(os.path.join(DEBUG_DIR, f"{debug_prefix}mismatch{ow}x{oh}.png"), out_img)
        out_img = cv2.resize(out_img, (rw, rh), interpolation=cv2.INTER_LINEAR)
    return out_img

# ==============================
# IOPaint Integration
# ==============================
def _decode_iopaint_response(resp: requests.Response, dbg_prefix: str):
    ct = resp.headers.get("Content-Type", "")
    content = resp.content
    if "application/json" in ct.lower():
        try:
            data = resp.json()
            b64 = data.get("image")
            if not b64:
                dlog(f"IOPaint JSON tanpa 'image': {str(data)[:250]}")
                return None
            raw = base64.b64decode(b64)
            arr = np.frombuffer(raw, np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception as e:
            dlog(f"Parse JSON IOPaint gagal: {e}")
            return None
    if content and (content[:8] == b"\x89PNG\r\n\x1a\n" or content[:3] == b"\xff\xd8\xff"):
        arr = np.frombuffer(content, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    dlog(f"IOPaint balasan bukan gambar/JSON (status {resp.status_code}, ct={ct})")
    return None

def inpaint_opencv(img_bgr: np.ndarray, mask: np.ndarray, debug_prefix=None):
    """
    Isi mask dengan piksel sekitar memakai algoritma Telea/NS dari OpenCV.

    Untuk bubble manga (warna datar) ini jauh lebih bagus daripada model
    generatif, diuji pada contoh nyata:
        std dev area teks  ->  IOPaint manga : 13.2
                               OpenCV       :  1.1
    Plus jauh lebih cepat (tanpa HTTP, tanpa muat model) dan tidak butuh
    server apa pun yang jalan.
    """
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = (mask > 127).astype(np.uint8) * 255
    if cv2.countNonZero(mask) == 0:
        return img_bgr
    flag = cv2.INPAINT_NS if INPAINT_OPENCV_MODE == "ns" else cv2.INPAINT_TELEA
    out = cv2.inpaint(img_bgr, mask, INPAINT_OPENCV_RADIUS, flag)
    if debug_prefix:
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{debug_prefix}_inpaint.png"), out)
    return out


def inpaint_image(img_bgr: np.ndarray, mask: np.ndarray, debug_prefix=None):
    """Pilih mesin inpaint sesuai INPAINT_METHOD, dengan fallback aman."""
    if INPAINT_METHOD == "iopaint":
        out = inpaint_with_iopaint_crop(img_bgr, mask, debug_prefix)
        # Kalau IOPaint tidak aktif/berhasil, jangan diam-diam mengembalikan hasil apa adanya.
        if out is not None and np.array_equal(out, img_bgr) and cv2.countNonZero(
                (mask > 127).astype(np.uint8) * 255) > 0:
            dlog("    IOPaint tidak tersedia -> pakai OpenCV inpaint")
            return inpaint_opencv(img_bgr, mask, debug_prefix)
        return out
    return inpaint_opencv(img_bgr, mask, debug_prefix)


def inpaint_with_iopaint_crop(img_bgr: np.ndarray, mask: np.ndarray, debug_prefix=None):
    """
    Crop area minimal dari mask, kirim hanya patch ke IOPaint → tempelkan kembali.
    Hemat memori & bandwidth.
    """
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = (mask > 127).astype(np.uint8) * 255

    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return img_bgr  # tidak ada area untuk di-inpaint

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())

    crop_img = img_bgr[y1:y2+1, x1:x2+1]
    crop_mask = mask[y1:y2+1, x1:x2+1]

    crop_img_p, crop_mask_p, pads = _pad_to_stride(crop_img, crop_mask, stride=STRIDE)

    ok_img, img_buf = cv2.imencode(".png", crop_img_p)
    ok_msk, mask_buf = cv2.imencode(".png", crop_mask_p)
    if not ok_img or not ok_msk:
        dlog("❌ encode PNG gagal (crop image/mask)")
        return img_bgr

    payload = {
        "image": base64.b64encode(img_buf).decode("utf-8"),
        "mask": base64.b64encode(mask_buf).decode("utf-8"),
        "hd_strategy": "Original",
        "sd_keep_unmasked_area": True
    }

    try:
        r = requests.post(f"{IOPAINT_URL}/api/v1/inpaint", json=payload, timeout=120)
        if r.status_code != 200:
            dlog(f"❌ IOPaint crop HTTP {r.status_code}: {r.text[:300]}")
            return img_bgr

        img_out_padded = _decode_iopaint_response(r, debug_prefix or "resp_crop")
        if img_out_padded is None:
            dlog("❌ decode crop hasil IOPaint gagal → pakai gambar lama")
            return img_bgr

        img_out_patch = _unpad(img_out_padded, pads)
        img_out_patch = _ensure_same_size(img_out_patch, crop_img, debug_prefix)

        result = img_bgr.copy()
        result[y1:y2+1, x1:x2+1] = img_out_patch

        if debug_prefix:
            cv2.imwrite(os.path.join(DEBUG_DIR, f"{debug_prefix}_crop_inpaint.png"), img_out_patch)

        return result

    except Exception as e:
        dlog(f"❌ iopaint crop error: {e}")
        return img_bgr

# ==============================
# API Endpoint
# ==============================
# ==============================
# Core Pipeline (reusable)
# ==============================
def process_image(
    cv_img: np.ndarray,
    *,
    shot_id: str | None = None,
    lang: str = "auto",
    pad: int = 10,
    shrink: int = 0,
    draw_debug: bool = True,
    min_bubble_conf: float = 0.1,
    preprocess_combined: bool = False,
    preprocess_auto_switch: bool = True,
    do_translate: bool = True,
    save_debug: bool = True,
) -> dict:
    """
    Pipeline lengkap: YOLO -> EasyOCR -> translasi -> IOPaint.

    Input  : cv_img (BGR numpy array)
    Output : dict { bubbles, total_bubbles, script, replaced_file, ... }

    Dipakai oleh endpoint FastAPI DAN desktop app, supaya tidak ada dua
    versi logika yang bisa berbeda.
    """
    if shot_id is None:
        shot_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]

    H, W = cv_img.shape[:2]
    dlog(f"📐 Input image size: {W}x{H}")

    debug_paths = []

    # Simpan input asli
    inp_name = f"{shot_id}_input.png"
    if save_debug:
        cv2.imwrite(os.path.join(DEBUG_DIR, inp_name), cv_img)
        debug_paths.append(f"/debug/{inp_name}")

    # ===== Preprocess hanya untuk OCR (bukan untuk YOLO/inpaint)
    ocr_base_img = cv_img
    preprocessed_path = None
    if preprocess_combined:
        ocr_base_img = preprocess_webtoon_combined(cv_img, auto_switch=preprocess_auto_switch)
        pre_path = f"{shot_id}_preprocessed.png"
        cv2.imwrite(os.path.join(DEBUG_DIR, pre_path), ocr_base_img)
        debug_paths.append(f"/debug/{pre_path}")
        preprocessed_path = f"/debug/{pre_path}"
        dlog(f"🧹 Preprocessed image saved: {pre_path}")

    # ==============================
    # YOLO Detection (gunakan gambar asli)
    # ==============================
    results = yolo_model.predict(cv_img, imgsz=IMG_SIZE, conf=YOLO_CONF, verbose=False)
    raw_boxes = []
    for r in results:
        if getattr(r, "boxes", None) is None:
            continue
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = getattr(r.boxes, "conf", None)
        confs = confs.cpu().numpy() if confs is not None else None
        for i, box in enumerate(xyxy):
            x1, y1, x2, y2 = map(int, box[:4])
            conf = float(confs[i]) if confs is not None else 1.0
            raw_boxes.append([x1, y1, x2, y2, conf])

    filtered_boxes = merge_overlapping_boxes_keep_smallest(raw_boxes, 0.5)

    # ==============================
    # Global OCR (teks di luar bubble) — pakai gambar preprocess bila aktif
    # ==============================
    global_ocr_results, _, _, global_script = safe_easyocr(ocr_base_img, lang_hint=lang)
    dlog(
        f"🌐 Global OCR script='{global_script.get('script')}' "
        f"total={global_script.get('total')} "
        f"hangul={global_script.get('hangul_ratio', 0):.2f} "
        f"kana={global_script.get('kana_ratio', 0):.2f} "
        f"han={global_script.get('han_ratio', 0):.2f} "
        f"latin={global_script.get('latin_ratio', 0):.2f}"
    )
    global_text_bubbles = []
    global_masks = []
    for gi, (poly, text, c) in enumerate(global_ocr_results):
        if c < MIN_LINE_CONF or not text.strip():
            continue
        # Skip jika poligon masuk ke box bubble (berdasarkan deteksi YOLO dari gambar asli)
        if any(poly_in_box(poly, (bx1, by1, bx2, by2)) for (bx1, by1, bx2, by2, _) in filtered_boxes):
            continue

        x, y, w, h = cv2.boundingRect(np.array(poly, np.int32))
        x1, y1, x2, y2 = clamp_bbox(x, y, x + w, y + h, W, H)
        conf = float(c)
        if conf < min_bubble_conf:
            continue

        line_script = classify_script(text.strip())
        line_entry = {
            "text": text.strip(),
            "translation": fake_translate(text.strip()),
            "bbox": [x1, y1, x2, y2],
            "conf": conf,
            "script": line_script["script"],
            "hangul_ratio": round(line_script["hangul_ratio"], 4),
        }

        global_text_bubbles.append({
            "id": f"g{gi}",
            "bbox": [x1, y1, x2, y2],
            "conf": conf,
            "raw_text": text.strip(),
            "translation": line_entry["translation"],
            "script": line_script["script"],
            "lines": [line_entry]
        })

        # Mask untuk teks di luar bubble.
        # Jangan pakai rect penuh: EasyOCR sering salah membaca garis junction/
        # garis penghubung bubble sebagai teks, lalu OpenCV mencolok gambar
        # yang jauh di sekitarnya (coretan vertikal di dalam bubble).
        cxg, cyg = (x1 + x2) // 2, (y1 + y2) // 2
        inside_bubble = any(
            bx1 - 6 <= cxg <= bx2 + 6 and by1 - 6 <= cyg <= by2 + 6
            for (bx1, by1, bx2, by2, _) in filtered_boxes
        )
        if inside_bubble:
            dlog(f"    lewati global {gi}: pusatnya di dalam bubble "
                 f"(kemungkinan garis outline, bukan teks)")
            continue

        bw_, bh_ = x2 - x1, y2 - y1
        if bh_ > 0 and bw_ > 0 and (bh_ / bw_) > GLOBAL_TAIL_ASPECT:
            dlog(f"    lewati global {gi}: bentuk {bw_}x{bh_} terlalu "
                 f"tinggi/vertikal (kemungkinan garis bubble)")
            continue
        # inset supaya sumber isian tetap di sekitar teks
        ix = max(1, int(bw_ * 0.08))
        iy = max(1, int(bh_ * 0.12))
        sub_mask = np.zeros(cv_img.shape[:2], dtype=np.uint8)
        cv2.rectangle(sub_mask, (x1 + ix, y1 + iy), (x2 - ix, y2 - iy), 255, -1)
        global_masks.append(sub_mask)

        crop_name = f"{shot_id}global{gi}_crop.png"
        if save_debug:
            cv2.imwrite(os.path.join(DEBUG_DIR, crop_name), cv_img[y1:y2, x1:x2])
            debug_paths.append(f"/debug/{crop_name}")

    # ==============================
    # Bubble OCR (YOLO) — crop dari gambar preprocess bila aktif
    # ==============================
    inpaint_img = cv_img.copy()
    bubbles_out = []
    all_masks = []

    for bubble_id, (x1, y1, x2, y2, yolo_c) in enumerate(filtered_boxes):
        x1, y1, x2, y2 = clamp_bbox(x1, y1, x2, y2, W, H)
        # crop OCR dari basis OCR (preprocessed bila aktif)
        crop_for_ocr = ocr_base_img[y1:y2, x1:x2]
        ocr_results, _, _, bubble_script = safe_easyocr(crop_for_ocr, lang_hint=lang)

        lines_keep = []
        line_confs = []
        line_weights = []
        line_entries = []
        ocr_mask = np.zeros(cv_img.shape[:2], dtype=np.uint8)

        dbg_line = cv_img.copy()

        for (poly, text, c) in ocr_results:
            if c >= MIN_LINE_CONF and text.strip():
                txt = text.strip()
                line_script = classify_script(txt)
                lines_keep.append(txt)
                line_confs.append(float(c))
                # Panjang baris jadi bobot: baris 1-2 karakter lebih sering
                # noise daripada benar, jangan dianggap seberat baris panjang.
                line_weights.append(max(1, len(txt)))

                # koordinat global (crop → global) — poly masih di koordinat crop_for_ocr
                poly_g = [(p[0] + x1, p[1] + y1) for p in poly]
                sub_mask = polygon_to_mask(cv_img.shape, poly_g, pad=pad, shrink=shrink)
                ocr_mask = cv2.bitwise_or(ocr_mask, sub_mask)

                bx, by, bw, bh = cv2.boundingRect(np.array(poly_g, np.int32))
                line_entries.append({
                    "text": txt,
                    "translation": fake_translate(txt),
                    "bbox": [int(bx), int(by), int(bx + bw), int(by + bh)],
                    "conf": float(c),
                    "script": line_script["script"],
                    "hangul_ratio": round(line_script["hangul_ratio"], 4),
                })

                # Debug polygon+conf
                pts = np.array(poly_g, np.int32)
                color = (0, 255, 0) if c >= 0.75 else ((0, 255, 255) if c >= 0.5 else (0, 0, 255))
                cv2.polylines(dbg_line, [pts], True, color, 2)
                label = f"{txt} ({c:.2f})"
                px, py = pts[0]
                cv2.putText(dbg_line, label, (px, max(0, py - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        dbg_line_name = f"{shot_id}bubble{bubble_id}_ocr_lines.png"
        if save_debug:
            cv2.imwrite(os.path.join(DEBUG_DIR, dbg_line_name), dbg_line)
            debug_paths.append(f"/debug/{dbg_line_name}")

        if line_confs:
            wsum = float(sum(line_weights))
            avg_line_conf = float(sum(c * w for c, w in zip(line_confs, line_weights)) / wsum)
        else:
            avg_line_conf = 0.0
        bubble_conf = 0.5 * float(yolo_c) + 0.5 * avg_line_conf

        # Ellipse mask untuk area bubble (mencegah terlalu luas)
        bubble_w = x2 - x1
        bubble_h = y2 - y1
        area = bubble_w * bubble_h
        if area < 20000: shrink_factor = 0.7
        elif area < 60000: shrink_factor = 0.65
        elif area < 120000: shrink_factor = 0.5
        else: shrink_factor = 0.40

        ellipse_mask = np.zeros(cv_img.shape[:2], dtype=np.uint8)
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        axes = (int(bubble_w * shrink_factor / 2), int(bubble_h * shrink_factor / 2))
        cv2.ellipse(ellipse_mask, center, axes, 0, 0, 360, 255, -1)

        if lines_keep:
            # Mask = ELIPS DIKECILKAN, bukan area teks.
            #
            # Diuji pada bubble nyata (std dev area teks / border error):
            #   elips penuh 0.90        -> 22.3 / 0.0   (isi tidak rata)
            #   teks dilated 27         ->  1.1 / 1.5   (border tergerus)
            #   elips 0.72 + inset 10   ->  0.5 / 0.0   (rata, border utuh)  <-- dipakai
            #
            # Elips dikecilkan supaya OpenCV memakaiHANYA warna di dalam bubble
            # sebagai sumber isian, dan garis bubble tidak pernah masuk mask.
            inset = BUBBLE_FILL_INSET
            fill_axes = (max(4, int(bubble_w * BUBBLE_FILL_SCALE / 2) - inset),
                         max(4, int(bubble_h * BUBBLE_FILL_SCALE / 2) - inset))
            fill_mask = np.zeros(cv_img.shape[:2], dtype=np.uint8)
            cv2.ellipse(fill_mask, center, fill_axes, 0, 0, 360, 255, -1)
            if BUBBLE_FILL_ERODE > 0:
                fill_mask = cv2.erode(
                    fill_mask,
                    np.ones((BUBBLE_FILL_ERODE, BUBBLE_FILL_ERODE), np.uint8),
                    iterations=1)

            all_masks.append((fill_mask, f"{shot_id}bubble{bubble_id}_mask"))

            if bubble_conf >= min_bubble_conf:
                bubbles_out.append({
                    "id": f"y{bubble_id}",
                    "bbox": [x1, y1, x2, y2],
                    "conf": round(bubble_conf, 4),
                    "raw_text": "\n".join(lines_keep),
                    "translation": fake_translate(" ".join(lines_keep)),
                    "script": bubble_script.get("script", "unknown"),
                    "ocr_lang": bubble_script.get("lang"),
                    "lines": line_entries
                })

        crop_name = f"{shot_id}bubble{bubble_id}_crop.png"
        if save_debug:
            cv2.imwrite(os.path.join(DEBUG_DIR, crop_name), cv_img[y1:y2, x1:x2])  # untuk referensi asli
            debug_paths.append(f"/debug/{crop_name}")

    # ==============================
    # Inpainting — gambar asli
    # ==============================
    dlog(f"🎨 Start inpainting (method={INPAINT_METHOD})...")
    dlog(f"    Before inpaint: {inpaint_img.shape[1]}x{inpaint_img.shape[0]}")

    # Global mask dulu
    for idx, m in enumerate(global_masks):
        before = inpaint_img.shape
        inpaint_img = inpaint_image(inpaint_img, m, f"{shot_id}global{idx}")
        after = inpaint_img.shape
        dlog(f"    Global mask {idx}: {before[1]}x{before[0]} -> {after[1]}x{after[0]}")

    # Lalu bubble mask
    for mask, dbg_prefix in all_masks:
        before = inpaint_img.shape
        inpaint_img = inpaint_image(inpaint_img, mask, dbg_prefix)
        after = inpaint_img.shape
        dlog(f"    Bubble {dbg_prefix}: {before[1]}x{before[0]} -> {after[1]}x{after[0]}")

    dlog(f"    After inpaint:  {inpaint_img.shape[1]}x{inpaint_img.shape[0]}")

    all_bubbles = bubbles_out + global_text_bubbles

    # ==============================
    # Translasi Korea -> Indonesia
    # ==============================
    if do_translate:
        _t0 = datetime.now()
        n_translated = apply_translations(all_bubbles)
        dlog(
            f"🌐 Translasi: {n_translated} baris dalam "
            f"{(datetime.now() - _t0).total_seconds():.1f}s"
        )
    else:
        n_translated = 0

    # ==============================
    # Debug Overlay Box Conf
    # ==============================
    dbg_overlay = inpaint_img.copy()
    for b in all_bubbles:
        x1, y1, x2, y2 = b["bbox"]
        conf = b.get("conf", 0.0)
        color = (0, 200, 0) if conf >= 0.75 else ((0, 200, 200) if conf >= 0.6 else (0, 0, 255))
        cv2.rectangle(dbg_overlay, (x1, y1), (x2, y2), color, 2)
        label = f"{b['id']} | conf={conf:.2f}"
        cv2.putText(dbg_overlay, label, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    dbg_overlay_name = f"{shot_id}_overlay_conf.png"
    if save_debug:
        cv2.imwrite(os.path.join(DEBUG_DIR, dbg_overlay_name), dbg_overlay)
        debug_paths.append(f"/debug/{dbg_overlay_name}")

    # Debug OCR box (semua bubble) — tampilkan di atas gambar asli (bukan preprocess)
    dbg_ocr = cv_img.copy()
    for b in all_bubbles:
        x1, y1, x2, y2 = b["bbox"]
        conf = b.get("conf", 0.0)
        color = (0, 255, 0) if conf >= 0.75 else ((0, 255, 255) if conf >= 0.5 else (0, 0, 255))
        cv2.rectangle(dbg_ocr, (x1, y1), (x2, y2), color, 2)
        label = f"{b['id']} conf={conf:.2f}"
        cv2.putText(dbg_ocr, label, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    dbg_ocr_name = f"{shot_id}_ocr_boxes.png"
    if save_debug:
        cv2.imwrite(os.path.join(DEBUG_DIR, dbg_ocr_name), dbg_ocr)
        debug_paths.append(f"/debug/{dbg_ocr_name}")

    # Simpan hasil akhir
    replaced_filename = f"replaced_{shot_id}.png"
    Image.fromarray(inpaint_img[:, :, ::-1]).save(os.path.join(DEBUG_DIR, replaced_filename))

    # ==============================
    # Log & JSON Debug
    # ==============================
    ocr_log_path = os.path.join(DEBUG_DIR, f"{shot_id}_ocr_log.txt")
    with open(ocr_log_path, "w", encoding="utf-8") as f:
        f.write(f"OCR DEBUG LOG — {shot_id}\n")
        f.write(f"Gambar: {W}x{H}\n\n")
        for b in all_bubbles:
            x1, y1, x2, y2 = b["bbox"]
            raw_text = b.get("raw_text", "")
            translation = b.get("translation", "")
            conf = b.get("conf", 0.0)
            f.write(f"[{b['id']}] BBox=({x1},{y1},{x2},{y2}) conf={conf:.2f}\n")
            f.write(f"Raw Text: {raw_text}\n")
            f.write(f"Translation: {translation}\n")
            f.write("-" * 40 + "\n")
    dlog(f"📝 OCR log saved: {ocr_log_path}")
    debug_paths.append(f"/debug/{os.path.basename(ocr_log_path)}")

    ocr_json_path = os.path.join(DEBUG_DIR, f"{shot_id}_ocr_debug.json")
    with open(ocr_json_path, "w", encoding="utf-8") as jf:
        json.dump({"bubbles": all_bubbles, "size": [W, H]}, jf, ensure_ascii=False, indent=2)
    debug_paths.append(f"/debug/{os.path.basename(ocr_json_path)}")

    # ==============================
    # Memory hygiene
    # ==============================
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    resp = {
        "replaced_path": f"/debug/{replaced_filename}",
        "replaced_file": os.path.join(DEBUG_DIR, replaced_filename),
        "total_bubbles": len(all_bubbles),
        "translated_lines": n_translated,
        "translation_enabled": get_translator() is not None,
        "translation_model": TRANSLATE_MODEL,
        "bubbles": all_bubbles,
        "script": {
            "detected": global_script.get("script"),
            "ocr_lang": global_script.get("lang"),
            "hangul_ratio": round(global_script.get("hangul_ratio", 0.0), 4),
            "kana_ratio": round(global_script.get("kana_ratio", 0.0), 4),
            "han_ratio": round(global_script.get("han_ratio", 0.0), 4),
            "latin_ratio": round(global_script.get("latin_ratio", 0.0), 4),
        },
        "debug_paths": debug_paths
    }
    if preprocessed_path:
        resp["preprocessed_path"] = preprocessed_path

    return resp

# ==============================
# API Endpoint
# ==============================
@app.post("/ocr_bubble")
async def ocr_bubble(
    file: UploadFile = File(...),
    lang: str = Query(default="auto"),
    pad: int = Query(default=10),
    shrink: int = Query(default=0),
    draw_debug: bool = Query(default=True),
    min_bubble_conf: float = Query(default=0.1),
    preprocess_combined: bool = Query(default=False),
    preprocess_auto_switch: bool = Query(default=True)
):
    """
    Mengembalikan:
      - replaced_path: path gambar hasil inpaint di /debug
      - bubbles: list bubble {id, bbox, conf, raw_text, translation, script, lines:[...]}
      - debug_paths: daftar file debug
      - (opsional) preprocessed_path jika preprocessing aktif
    """
    contents = await file.read()
    shot_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]

    image = Image.open(io.BytesIO(contents)).convert("RGB")
    cv_img = np.array(image)[:, :, ::-1].copy()

    return process_image(
        cv_img,
        shot_id=shot_id,
        lang=lang,
        pad=pad,
        shrink=shrink,
        draw_debug=draw_debug,
        min_bubble_conf=min_bubble_conf,
        preprocess_combined=preprocess_combined,
        preprocess_auto_switch=preprocess_auto_switch,
    )
