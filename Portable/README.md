# Chlandestine — Portable

Baca + terjemahkan webtoon Korea langsung dari layar. Tidak perlu login,
tidak perlu internet, tidak perlu install Python.

## Cara pakai

Klik dua kali **`Mulai.bat`**, lalu tekan **Ctrl+Alt+T** dan drag area
gambar yang mau dibaca.

Jendela hitam `Mulai.bat` boleh ditutup setelah panel mengambang muncul.

## Kenapa bukan `.exe`?

`Chlandestine.exe` **sudah pernah saya build** (ada di
`D:\Chlandestine\app\Chlandestine\`), tapi Windows **Smart App Control**
menolaknya:

```
Chlandestine.exe          -> NotSigned        (diblokir)
venv\Scripts\python.exe   -> Valid (signed PSF) (lolos)
```

Smart App Control memblokir executable yang tidak signed dan belum punya
reputasi. `Mulai.bat` memakai `python.exe` resmi yang sudah signed, jadi
jalan tanpa perlu mematikan antivirus atau mengubah kebijakan apa pun.

Kalau nanti mau `.exe` betulan, opsinya (berbutuh hak admin):
1. Sign exe dengan sertifikat code-signing, atau
2. Matikan Smart App Control di
   Settings → Windows Security → App & browser control → Smart App Control,
   atau
3. Tambahkan folder ini ke exclusions Microsoft Defender.

## Struktur folder

```
Mulai.bat                        launcher (klik ini)
app\main.py                      pipeline OCR + translasi
app\desktop_app.py                UI panel mengambang
models\nllb-200-distilled-1.3B\  model translasi Korea -> Indonesia (5,1 GB)
ocrmodels\craft_mlt_25k.pth       model deteksi teks EasyOCR
ocrmodels\korean_g2.pth           model baca aksara Korea
weights\best.pt                  model deteksi bubble YOLO
```

Model **tidak** diunduh saat dijalankan — semuanya sudah ada di folder ini
dan aplikasi dipaksa offline (`HF_HUB_OFFLINE=1`).

## Hasil & log

Semua output ditulis ke:

```
%LOCALAPPDATA%\Chlandestine\debug\
%LOCALAPPDATA%\Chlandestine\ocr_debug.log
```

Jadi folder aplikasi boleh ditaruh di lokasi read-only.

## Kebutuhan

- Windows 10/11
- GPU NVIDIA (dibangun & diuji di RTX 5050, sm_120, torch 2.11 + cu128)
- RAM 8 GB, disk ±5,5 GB untuk folder ini
- Untuk gambar "bersih" (teks dihilang): jalankan IOPaint di
  `http://127.0.0.1:8080`. Kalau tidak, gambar keluar apa adanya.

## Troubleshooting

| Gejala | Penyebab & solusi |
|---|---|
| "Python tidak ditemukan" | Install Python 3.11, atau taruh folder `runtime\Scripts\python.exe` di samping `Mulai.bat` |
| Tekan Ctrl+Alt+T tidak kerja | Hotkey global butuh izin admin. Ganti pakai tombol **Tangkap Layar** di panel |
| Teks Korea terbaca sampah | Turunkan sharpen/pre-process, atau naikkan `MIN_LINE_CONF` di `app\main.py` |
| Terjemahan kosong | Cek `ocrm_debug.log`, cari baris `Model translasi siap` |
| Lambat | Loading model pertama ~60 detik (dari disk). Setelah itu di-cache RAM |