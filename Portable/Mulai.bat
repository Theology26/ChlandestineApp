@echo off
REM ===================================================================
REM  Chlandestine - launcher
REM  Jalankan: klik dua kali file ini (Mulai.bat)
REM
REM  Kenapa bukan .exe?
REM  Windows Smart App Control memblokir executable yang tidak signed.
REM  Launcher ini memakai python.exe resmi yang sudah signed, jadi jalan
REM  tanpa perlu mematikan antivirus / mengubah kebijakan apa pun.
REM ===================================================================
setlocal

REM --- lokasi folder ini ---
set "APPDIR=%~dp0"
REM hapus trailing backslash
if "%APPDIR:~-1%"=="\" set "APPDIR=%APPDIR:~0,-1%"

REM --- cari python ---
REM 1) venv lokal (XE, hasil portable)   2) python di sistem
set "PY="
if exist "%APPDIR%\runtime\Scripts\python.exe" (
    set "PY=%APPDIR%\runtime\Scripts\python.exe"
)
if not defined PY (
    for %%P in (
        "%LOCALAPPDATA%\Chlandestine\runtime\Scripts\python.exe"
        "D:\Chlandestine\venv\Scripts\python.exe"
    ) do (
        if exist %%P set "PY=%%~P"
    )
)
if not defined PY (
    where python.exe >nul 2>nul && set "PY=python.exe"
)

if not defined PY (
    echo.
    echo  [ERROR] Python tidak ditemukan.
    echo  Install Python 3.11 atau letakkan folder 'runtime' di samping file ini.
    echo.
    pause
    exit /b 1
)

echo.
echo  Chlandestine
echo  Python : %PY%
echo  Model  : %APPDIR%\models
echo.
echo  Tekan Ctrl+Alt+T untuk mulai menangkap layar.
echo  Jendela ini boleh ditutup setelah panel muncul.
echo.

REM selalu offline supaya tidak ada permintaan jaringan ke HuggingFace
set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"

REM hasil (gambar bersih, log) ditulis ke folder data pengguna, bukan ke
REM folder aplikasi -- supaya folder ini bisa ditaruh di lokasi read-only.
if not defined CHLANDESTINE_DATA set "CHLANDESTINE_DATA=%LOCALAPPDATA%\Chlandestine"

"%PY%" "%APPDIR%\app\desktop_app.py"
set "RC=%ERRORLEVEL%"

if not "%RC%"=="0" (
    echo.
    echo  [ERROR] Aplikasi berhenti dengan kode %RC%.
    echo  Cek log: %%LOCALAPPDATA%%\Chlandestine\ocr_debug.log
    pause
)
endlocal