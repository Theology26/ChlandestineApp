@echo off
REM ===================================================================
REM  IOPaint server - mesin inpaint untuk menghapus teks manga
REM
REM  main.py mengirim permintaan ke http://127.0.0.1:8080/api/v1/inpaint
REM  Kalau server ini mati, gambar keluar apa adanya (tidak error).
REM ===================================================================
setlocal

set "IOPAINT_DIR=D:\Chlandestine\venv-iopaint"
set "IOPAINT_MODEL=manga"
set "IOPAINT_URL=http://127.0.0.1:8080"

echo.
echo  IOPaint - server inpaint
echo  Model : %IOPAINT_MODEL%
echo  Alamat: %IOPAINT_URL%
echo.
echo  Model diunduh otomatis saat pertama kali jalan.
echo  Tutup jendela ini untuk menghentikan server.
echo.

"%IOPAINT_DIR%\Scripts\python.exe" -m iopaint start ^
    --host 127.0.0.1 ^
    --port 8080 ^
    --model %IOPAINT_MODEL% ^
    --no-inbrowser

echo.
echo  IOPaint berhenti.
pause