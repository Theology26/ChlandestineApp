<div align="center">

# Chlandestine

**Read Korean webtoon. Get Indonesian subtitles. No API keys, no internet, no login.**

`Chrome extension → desktop app` · `Windows` · `100% offline`

[![Python](https://img.shields.io/badge/python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/torch-2.11%20cu128-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

</div>

---

## What it does

Point it at any Korean webtoon or manga panel. It finds the speech bubbles,
reads the Korean text, translates it to Indonesian, and **erases the original
text** so you can paste the translation back in.

No API key. No account. No network calls. Everything runs locally on your GPU.

**1 — Press `Ctrl + Alt + T`, then drag over the panel you want to read**

<table>
<tr>
<td width="50%" align="center" valign="middle">

**Source** · Korean webtoon

<br><br>

우리 집은<br>
아빠 안 계서.

</td>
<td width="50%" align="center" valign="middle">

**Result** · floating panel, always on top

<br><br>

▢ <b>Rumah kami</b> <sub>conf 100%</sub><br>
&nbsp;&nbsp;Ayahnya ada di rumah.<br><br>
▢ <b>Kakak, kenapa kau berbohong lagi?</b>

<br><br>

<code>Lihat hasil</code> &nbsp; <code>Salin semua</code>

</td>
</tr>
</table>

The original Korean text is **erased from the image**, so the translation can
be pasted straight back in.

---

## Why it's built this way

Most manga translation tools call a cloud API. This one doesn't — and the
interesting part is *how* each piece was chosen, by measuring rather than
assuming.

### Bubble detection — YOLOv8s, trained locally

| Model | mAP50 | mAP50-95 |
|---|---|---|
| `yolov8n` (the previous baseline) | 0.883 | 0.476 |
| **`yolov8s` (this repo)** | **0.910** | **0.575** |

`mAP50-95` matters more than `mAP50` here, because the inpainting mask is
derived from the bounding box. A loosely-fit box means the wrong pixels get
erased. The upgrade gained **10 points** of localisation accuracy.

Trained on 588 images from
[`manga-speech-bubble-detection`](https://universe.roboflow.com/manga-speech-bubble-detection)
(CC BY 4.0), `imgsz 640`, `device 0`.

> The Roboflow export ships **polygon (segmentation)** annotations, not
> detection boxes. `convert_labels.py` converts all 26,992 polygons into
> axis-aligned boxes (originals backed up to `_label_asli_polygon/`).

### Korean OCR with real script detection

EasyOCR's `ko` model reads Korean well, but forcing it to read Japanese or
Chinese manga as Korean produces garbage. So every OCR result is validated
against **complete Unicode block ranges**:

- **Hangul** — Jamo `U+1100`, Compatibility Jamo `U+3130`, Jamo Ext-A `U+A960`,
  Syllables `U+AC00`, Jamo Ext-B `U+D7B0`, Halfwidth Jamo `U+FFA0`
- **Kana** (Japanese) and **Han** (Chinese/Hanja) are explicitly *excluded*
  from the Korean bucket

If the script doesn't match, the text is re-read with the correct model.
Measured on a Korean page: `hangul_ratio 1.00 · kana 0.00 · han 0.00`.

### Translation — NLLB-200-1.3B, offline

`facebook/nllb-200-distilled-1.3B` in fp16 (~2.6 GB VRAM). Same sentences,
two model sizes:

| Korean | 600M | **1.3B** |
|---|---|---|
| 우리 집은 | "Rumah kami **adalah rumah kami**." | **"Rumah kami"** |
| 아빠 안 계서. | "Ayahku tidak ada." | **"Ayahnya ada di sini."** |
| 형, 너 왜 또 거짓말을 해? | "**Saudara**, mengapa **Anda**…" | **"**Kakak**, kenapa **kau**…" |
| 절대 안 돼! | "Jangan sekali-kali!" | **"Tidak mungkin!"** |
| 먹고 가라. | "Makanlah **dan makanlah**." | **"Makanlah dan pergilah."** |

On top of the model there's a small post-processor: a Korean honorific table
(`형/오빠/누나/언니 → Kakak`), register normalisation (`Anda → kamu`), and
duplicate-collapsing (`X adalah X → X`).

### Text removal — OpenCV beat the AI model

This was the most counter-intuitive result. IOPaint's `manga` model was set
up, running, and then **lost** to `cv2.inpaint`:

| Mask strategy | std dev (lower = flatter) | Border damage |
|---|---|---|
| Original (text present) | 74.1 | — |
| Full ellipse + IOPaint `manga` | 41.2 | — |
| Full ellipse + OpenCV | 22.3 | 0.0 |
| Text-only mask + OpenCV | 1.1 | 1.5 |
| **Shrunken ellipse + OpenCV** | **0.5** | **0.0** |

Speech bubbles are *flat-coloured regions*. Filling from surrounding pixels
beats a generative model that invents texture — and it's instant, needs no
server, and no VRAM. IOPaint remains available via `INPAINT_METHOD=iopaint`
for panels with genuinely complex backgrounds.

The winning mask is an ellipse shrunk to 72% with a 16px inset plus a 3px
erode, so it can never touch the bubble outline.

---

## Quick start

```powershell
# 1. Get the models (see "Models" below — ~5.3 GB)
#    nllb-200-distilled-1.3B  ->  models\nllb-200-distilled-1.3B\
#    craft_mlt_25k.pth        ->  ocrmodels\
#    korean_g2.pth            ->  ocrmodels\
#    best.pt (trained)        ->  weights\

# 2. Launch
.\Portable\Mulai.bat
```

Press **Ctrl + Alt + T**, drag over a panel, done.

<details>
<summary>Run as an API instead</summary>

```powershell
.\venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000
curl -X POST "http://127.0.0.1:8000/ocr_bubble?lang=auto" -F "file=@page.png"
```
</details>

---

## System requirements

Measured on the config below while processing a 1280×1908 page
(2 bubbles, 3 lines translated):

| Resource | Peak usage | Minimum | Recommended |
|---|---|---|---|
| **VRAM** | **3.85 GB** | 4 GB | 6 GB+ |
| **RAM** | **5.06 GB** | 8 GB | 16 GB |
| **Disk** | — | 12 GB | 20 GB SSD |
| **GPU** | optional | — | NVIDIA ≥6 GB |

### Tested on

```
CPU     Intel Core i7-13620H  (10 cores / 16 threads)
RAM     16 GB
GPU     NVIDIA GeForce RTX 5050 Laptop  (8 GB, Blackwell sm_120)
OS      Windows 11 Home  ·  build 10.0.26300
Python  3.11.9  ·  torch 2.11.0+cu128  ·  ultralytics 8.4.172
```

### Timing

| | |
|---|---|
| First run (cold, loads models from disk) | ~100 s |
| Warm runs | ~30 s |
| Per page after warm-up | ~1–3 s |

The first run is dominated by loading the 5.2 GB translation model. Everything
after that is cached.

### ⚠️ Pick the right PyTorch build

The build **must** match your GPU architecture, or CUDA kernels won't load:

```powershell
# RTX 50-series (Blackwell, sm_120) — this is what this repo uses
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# RTX 30/40-series, older cards
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

An RTX 5050 (Blackwell) **will not work** with the older `cu118` build — it
only goes up to `sm_90`. Symptom: `no kernel image is available for
execution on the device`.

### Running without a GPU

It works, but translation is the bottleneck and will feel slow. To keep it
usable, switch to the smaller model and disable fp16:

```powershell
set TRANSLATE_MODEL=facebook/nllb-200-distilled-600M
set TRANSLATE_FP16=0
```

Budget **8 GB RAM minimum** on CPU — the 1.3B model in fp32 needs ~5.2 GB
just for weights.

### Windows Smart App Control

Some Windows machines block unsigned downloads at runtime, which breaks
automatic model fetching. The app looks for **local model folders** first, so
if models won't download, place them manually (see [Models](#models)) and
launch via `Portable\Mulai.bat` rather than the API. See `README.md` in the
repo for the full explanation.

---

## Architecture

Two frontends, one pipeline:

| Frontend | Entry point |
|---|---|
| `desktop_app.py` | Floating always-on-top panel (PySide6) |
| FastAPI | `POST /ocr_bubble` — a ~20-line wrapper |

Both call the **same** function, so the desktop app and the API can never
drift apart:

```
process_image(cv_img)  ->  main.py
   |
   +-- 1. YOLOv8s ....... locate speech bubbles
   +-- 2. EasyOCR ....... read the Korean text
   +-- 3. NLLB-1.3B .... translate to Indonesian
   +-- 4. OpenCV ....... erase the original text
   |
   +-- returns { bubbles, script, replaced_file, ... }
```

| File | Role |
|---|---|
| `main.py` | Core pipeline, exposed as `process_image()` |
| `desktop_app.py` | Floating always-on-top panel (PySide6) |
| `convert_labels.py` | Roboflow polygon → YOLO detection labels |
| `train_yolo.py` | Bubble detector training (`device=0`, GPU-forced) |
| `fetch_dataset.py` | Roboflow dataset download |

### Configuration

Everything is environment-variable driven, with working defaults:

| Variable | Default | Purpose |
|---|---|---|
| `INPAINT_METHOD` | `opencv` | `opencv` or `iopaint` |
| `BUBBLE_FILL_SCALE` | `0.72` | Fill-mask size (smaller = safer border) |
| `MIN_LINE_CONF` | `0.30` | EasyOCR per-line confidence floor |
| `TRANSLATE_MODEL` | `facebook/nllb-200-distilled-1.3B` | Or a local folder path |
| `TRANSLATE_FP16` | `1` | Halves VRAM, faster (turn off on CPU) |
| `IMG_SIZE` | `640` | Must match training resolution |
| `YOLO_MODEL_PATH` | auto-detected | Override the detector |

Models are resolved from several locations (next to the exe, project root,
portable folder), so the app runs from a flash drive. Setting
`HF_HUB_OFFLINE=1` guarantees no network access.

---

## Models

| What | Where | Size |
|---|---|---|
| `nllb-200-distilled-1.3B` | `models/` | 5.2 GB |
| `craft_mlt_25k.pth` | `ocrmodels/` | 79 MB |
| `korean_g2.pth` | `ocrmodels/` | 15 MB |
| `best.pt` (this repo) | `weights/` | 21 MB |

Model download is blocked on some machines by Windows Smart App Control,
which is why the app looks for **local folders** first and never requires a
download at runtime.

---

## Known limitations

Being straight about these:

- **Translation is not perfect.** NLLB is a general MT model, not a
  manga-dialogue model. Some lines come out too literal, and a few are
  simply wrong. A larger local LLM would read better but costs ~3× the time.
- **OCR misreads some glyphs.** `돌아가거든` was once read as `돌아가석거든`,
  which then produced a completely wrong translation. The root cause is
  OCR, not the translator.
- **Windows-focused.** The screen-capture overlay uses `mss` and the
  launcher expects Windows paths.
- **No document/PDF input yet** — screen regions only.
- Trained on 588 images. A larger annotated set would help a lot.

---

## Credits

- **Dataset** — [`manga-speech-bubble-detection`](https://universe.roboflow.com/manga-speech-bubble-detection)
  by Roboflow Universe · **CC BY 4.0** · attribution required
- **Translation** — [`facebook/nllb-200-distilled-1.3B`](https://huggingface.co/facebook/nllb-200-distilled-1.3B)
  · Meta · **CC-BY-NC 4.0** *(non-commercial)*
- **Detection** — [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) · AGPL-3.0 / commercial
- **OCR** — [EasyOCR](https://github.com/JaidedAI/EasyOCR) · Apache-2.0

> ⚠️ NLLB is **non-commercial**. For commercial use you'll need a different
> translation model or an appropriate licence.

---

## Indo

Baca webtoon Korea, dapat terjemahan Bahasa Indonesia, teks aslinya dihapus.
Tanpa API key, tanpa internet, tanpa login — semua jalan lokal di GPU.

Pintasan **Ctrl + Alt + T**, drag area gambarnya, selesai. Panel mengambang
selalu di atas, bisa disalin per-baris atau sekaligus.

Yang membuatnya works: deteksi bubble dilatih ulang dengan YOLOv8s
(mAP50-95 naik dari 0,476 → 0,575), OCR Korea divalidasi dengan rentang
Unicode Hangul lengkap sehingga manga Jepang/Cina tidak lagi salah dibaca
sebagai Korea, dan penghapusan teks memakai OpenCV — karena **diukur** lebih
bagus dari model generatif (std dev 0,5 vs 41,2) untuk bubble berwarna datar.

</div>