"""
Chlandestine Desktop — floating overlay untuk baca + terjemahkan webtoon Korea.

Alur kerja (mirip extension Chrome):
    1. User baca webtoon di browser
    2. Tekan Ctrl+Alt+T (atau tombol Tangkap)
    3. Drag Kotak di layar → region itu di-capture
    4. YOLO deteksi bubble → EasyOCR baca Korea → NLLB terjemahkan ke Indonesia
    5. Panel mengambang menampilkan gambar bersih + daftar terjemahan

Menjalankan:
    D:\\Chlandestine\\venv\\Scripts\\python.exe -m uvicorn main:app   (tidak perlu)
    D:\\Chlandestine\\venv\\Scripts\\python.exe desktop_app.py
"""

import sys
import os
import time
import threading

import numpy as np
from PySide6.QtCore import (
    Qt, QThread, Signal, QTimer, QPoint, QRect, QSettings, QSize
)
from PySide6.QtGui import (
    QGuiApplication, QPainter, QColor, QPen, QFont, QCursor, QGuiApplication as _QA,
    QIcon, QPixmap, QImage
)
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFrame, QVBoxLayout, QHBoxLayout,
    QScrollArea, QSizePolicy, QCheckBox, QComboBox, QApplication as QApp
)

# main.py di folder yang sama.
# Sengaja TIDAK di-import di sini: main.py memuat model YOLO (berat ~5 detik),
# dan import di level modul akan menunda tampilnya UI. Import dilakukan di
# dalam worker thread saat proses pertama kali berjalan.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

APP_NAME = "Chlandestine"
HOTKEY = "<ctrl>+<alt>+t"

# ==============================
# Tema
# ==============================
C = {
    "bg":        "#15171c",
    "surface":   "#1d2027",
    "surface2":  "#252932",
    "hover":     "#2f343f",
    "border":    "#313641",
    "text":      "#e9ecf1",
    "dim":       "#98a1b2",
    "faint":     "#6b7484",
    "accent":    "#ff5470",
    "accent_hi": "#ff7690",
    "good":      "#3ddc97",
    "warn":      "#ffc857",
    "bad":       "#ff5470",
}

QSS = f"""
QWidget {{
    background: {C['bg']};
    color: {C['text']};
    font-family: "Segoe UI", "Malgun Gothic", sans-serif;
    font-size: 13px;
}}
QFrame#card {{
    background: {C['surface']};
    border: 1px solid {C['border']};
    border-radius: 10px;
}}
QFrame#titlebar {{
    background: {C['surface']};
    border: none;
    border-bottom: 1px solid {C['border']};
    border-top-left-radius: 12px;
    border-top-right-radius: 12px;
}}
QPushButton {{
    background: {C['surface2']};
    border: 1px solid {C['border']};
    border-radius: 8px;
    padding: 7px 12px;
    color: {C['text']};
}}
QPushButton:hover {{ background: {C['hover']}; }}
QPushButton:pressed {{ background: {C['border']}; }}
QPushButton:disabled {{ color: {C['faint']}; border-color: {C['border']}; }}
QPushButton#primary {{
    background: {C['accent']};
    border: none;
    border-radius: 8px;
    padding: 10px 16px;
    font-weight: 600;
    color: #ffffff;
}}
QPushButton#primary:hover {{ background: {C['accent_hi']}; }}
QPushButton#primary:disabled {{ background: {C['border']}; color: {C['faint']}; }}
QPushButton#ghost {{
    background: transparent;
    border: none;
    padding: 4px 8px;
    color: {C['dim']};
    font-size: 15px;
}}
QPushButton#ghost:hover {{ color: {C['text']}; background: {C['hover']}; }}
QScrollArea {{ border: none; background: transparent; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QScrollBar:vertical {{
    background: transparent; width: 10px; margin: 2px;
}}
QScrollBar::handle:vertical {{
    background: {C['border']}; border-radius: 5px; min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{ background: {C['faint']}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
QCheckBox {{ color: {C['dim']}; spacing: 7px; }}
QCheckBox::indicator {{
    width: 15px; height: 15px;
    border: 1px solid {C['border']};
    border-radius: 4px;
    background: {C['surface2']};
}}
QCheckBox::indicator:checked {{ background: {C['accent']}; border-color: {C['accent']}; }}
QComboBox {{
    background: {C['surface2']};
    border: 1px solid {C['border']};
    border-radius: 7px;
    padding: 5px 9px;
    min-width: 78px;
}}
QComboBox:hover {{ border-color: {C['faint']}; }}
QComboBox QAbstractItemView {{
    background: {C['surface2']};
    border: 1px solid {C['border']};
    selection-background-color: {C['accent']};
    outline: none;
}}
"""


def conf_color(conf: float) -> str:
    if conf >= 0.75:
        return C["good"]
    if conf >= 0.50:
        return C["warn"]
    return C["bad"]


# ==============================
# Overlay pilih region
# ==============================
class RegionSelector(QWidget):
    """Layar penuh semi-transparan; user drag untuk memilih area."""

    picked = Signal(int, int, int, int)   # x, y, w, h  (logical Qt coords)
    cancelled = Signal()

    def __init__(self):
        super().__init__(
            None,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setMouseTracking(True)

        self._origin = None
        self._current = None
        self._done = False

        screen = QGuiApplication.primaryScreen()
        self.setGeometry(screen.virtualGeometry())

    # ---------- gambar ----------
    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        full = QRect(0, 0, self.width(), self.height())
        if self._current is None:
            p.fillRect(full, QColor(8, 9, 12, 205))
            self._draw_hint(p)
            return

        sel = self._norm_rect()
        # redupkan di luar area terpilih (4 persegi, bukan bolong)
        p.fillRect(QRect(0, 0, self.width(), sel.top()), QColor(8, 9, 12, 205))
        p.fillRect(QRect(0, sel.bottom() + 1, self.width(), self.height()), QColor(8, 9, 12, 205))
        p.fillRect(QRect(0, sel.top(), sel.left(), sel.height()), QColor(8, 9, 12, 205))
        p.fillRect(QRect(sel.right() + 1, sel.top(), self.width(), sel.height()), QColor(8, 9, 12, 205))

        # garis tepi + ukuran
        p.setPen(QPen(QColor(C["accent"]), 2))
        p.drawRect(sel)

        label = f"{sel.width()} × {sel.height()}"
        fm = p.fontMetrics()
        tw = fm.horizontalAdvance(label) + 16
        ly = sel.top() - 28
        if ly < 4:
            ly = sel.top() + 6
        p.fillRect(QRect(sel.left(), ly, tw, 22), QColor(C["accent"]))
        p.setPen(QColor("#ffffff"))
        f = QFont("Segoe UI", 10)
        f.setWeight(QFont.Weight.DemiBold)
        p.setFont(f)
        p.drawText(QRect(sel.left(), ly, tw, 22), Qt.AlignmentFlag.AlignCenter, label)

    def _draw_hint(self, p):
        f = QFont("Segoe UI", 14)
        f.setWeight(QFont.Weight.DemiBold)
        p.setFont(f)
        p.setPen(QColor("#ffffff"))
        p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                   "Drag untuk memilih area  ·  Esc untuk batal")
        f2 = QFont("Segoe UI", 10)
        p.setFont(f2)
        p.setPen(QColor(C["dim"]))
        p.drawText(self.rect().adjusted(0, 34, 0, 0), Qt.AlignmentFlag.AlignCenter,
                   "Tarik hanya area bubble-nya supaya lebih cepat & akurat")

    # ---------- input ----------
    def _norm_rect(self) -> QRect:
        r = QRect(self._origin, self._current).normalized()
        return r

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._origin = e.position().toPoint()
            self._current = self._origin
            self.update()

    def mouseMoveEvent(self, e):
        if self._origin is not None:
            self._current = e.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, e):
        if e.button() != Qt.MouseButton.LeftButton or self._origin is None:
            return
        self._current = e.position().toPoint()
        r = self._norm_rect()
        self.close()
        if r.width() >= 24 and r.height() >= 24:
            self.picked.emit(r.x(), r.y(), r.width(), r.height())
        else:
            self.cancelled.emit()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Escape:
            self.close()
            self.cancelled.emit()

    def closeEvent(self, e):
        if not self._done and self._origin is None:
            self._done = True
        super().closeEvent(e)


# ==============================
# Worker OCR
# ==============================
class OcrWorker(QThread):
    status = Signal(str)
    done = Signal(dict)
    failed = Signal(str)

    def __init__(self, bgr, options, parent=None):
        super().__init__(parent)
        self._bgr = bgr
        self._opt = options
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            self.status.emit("Memuat model…")
            t0 = time.time()
            import main as engine

            self.status.emit("Mendeteksi bubble & membaca teks…")
            t1 = time.time()
            result = engine.process_image(self._bgr, save_debug=False, **self._opt)

            self.status.emit("Selesai")
            result["_timing"] = {"total": time.time() - t0, "ocr": time.time() - t1}
            self.done.emit(result)
        except Exception as exc:                      # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}")


# ==============================
# Kartu satu bubble
# ==============================
class BubbleCard(QFrame):
    clicked = Signal()

    def __init__(self, bubble: dict, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 11)
        lay.setSpacing(5)

        # --- baris atas: badge ---
        top = QHBoxLayout()
        top.setSpacing(6)

        conf = float(bubble.get("conf", 0.0))
        badge = QLabel(f"{conf:.0%}")
        badge.setStyleSheet(
            f"background: {C['surface2']}; color: {conf_color(conf)};"
            f"border: 1px solid {conf_color(conf)}; border-radius: 5px;"
            f"padding: 1px 6px; font-size: 11px; font-weight: 600;"
        )
        top.addWidget(badge)

        script = bubble.get("script")
        if script and script != "unknown":
            sl = QLabel(script)
            sl.setStyleSheet(f"color: {C['faint']}; font-size: 11px;")
            top.addWidget(sl)

        top.addStretch(1)
        lid = QLabel(str(bubble.get("id", "")))
        lid.setStyleSheet(f"color: {C['faint']}; font-size: 11px;")
        top.addWidget(lid)
        lay.addLayout(top)

        # --- teks Korea (asli) ---
        src = (bubble.get("raw_text") or "").strip()
        if src:
            ko = QLabel(src.replace("\n", "  "))
            ko.setWordWrap(True)
            ko.setStyleSheet(f"color: {C['dim']}; font-size: 12px;")
            lay.addWidget(ko)

        # --- terjemahan Indonesia ---
        tr = (bubble.get("translation") or "").strip()
        # simpan payload asli supaya "Salin semua" tidak reliant ke QLabel
        self.payload = tr.replace("[TL] ", "").strip() or src
        if tr and tr != f"[TL] {src}":
            idl = QLabel(tr.replace("\n", "  "))
            idl.setWordWrap(True)
            idl.setStyleSheet(f"color: {C['text']}; font-size: 14px; font-weight: 600;")
            lay.addWidget(idl)
        else:
            none = QLabel("(belum ada terjemahan)")
            none.setStyleSheet(f"color: {C['faint']}; font-size: 12px; font-style: italic;")
            lay.addWidget(none)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mouseReleaseEvent(e)


# ==============================
# Panel utama (floating)
# ==============================
class MainWindow(QWidget):
    def __init__(self):
        super().__init__(
            None,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool,
        )
        self.setWindowTitle(APP_NAME)
        self.setMinimumSize(400, 520)
        self.resize(452, 760)

        self.settings = QSettings("Chlandestine", "Desktop")
        self._worker = None
        self._selector = None
        self._before_pix = None
        self._after_pix = None
        self._show_after = False

        self._build_ui()
        self._restore_geometry()
        self._start_hotkey()

    # ---------------- UI ----------------
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(9)

        root.addWidget(self._build_titlebar())

        # tombol capture
        self.btn_capture = QPushButton("Tangkap Layar")
        self.btn_capture.setObjectName("primary")
        self.btn_capture.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_capture.clicked.connect(self.begin_capture)
        root.addWidget(self.btn_capture)

        # baris kontrol
        controls = QHBoxLayout()
        controls.setSpacing(7)

        self.chk_translate = QCheckBox("Terjemahkan")
        self.chk_translate.setChecked(bool(self.settings.value("translate", True, bool)))
        self.chk_translate.toggled.connect(
            lambda v: self.settings.setValue("translate", v))

        self.cmb_lang = QComboBox()
        self.cmb_lang.addItems(["Otomatis", "Korea", "Jepang", "Mandarin"])
        self.cmb_lang.setCurrentIndex(int(self.settings.value("lang", 0, int)))
        self.cmb_lang.currentIndexChanged.connect(
            lambda i: self.settings.setValue("lang", i))

        self.chk_pre = QCheckBox("Pre-process")
        self.chk_pre.setChecked(bool(self.settings.value("preprocess", False, bool)))

        controls.addWidget(self.chk_translate)
        controls.addWidget(self.cmb_lang)
        controls.addStretch(1)
        controls.addWidget(self.chk_pre)
        root.addLayout(controls)

        # preview
        self.preview = QLabel("Tekan Ctrl+Alt+T lalu drag area gambar")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumHeight(190)
        self.preview.setWordWrap(True)
        self.preview.setStyleSheet(
            f"background: {C['surface']}; border: 1px dashed {C['border']};"
            f"border-radius: 10px; color: {C['faint']}; padding: 10px;"
        )
        root.addWidget(self.preview)

        self.btn_toggle = QPushButton("Lihat hasil")
        self.btn_toggle.setVisible(False)
        self.btn_toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_toggle.clicked.connect(self._toggle_preview)
        root.addWidget(self.btn_toggle)

        # ringkasan
        self.summary = QLabel("")
        self.summary.setStyleSheet(f"color: {C['dim']}; font-size: 12px;")
        root.addWidget(self.summary)

        # daftar bubble
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        holder = QWidget()
        self.list_layout = QVBoxLayout(holder)
        self.list_layout.setContentsMargins(0, 0, 4, 0)
        self.list_layout.setSpacing(7)
        self.list_layout.addStretch(1)
        self.scroll.setWidget(holder)
        self.scroll.setStyleSheet("background: transparent;")
        root.addWidget(self.scroll, 1)

        # footer
        footer = QHBoxLayout()
        self.status = QLabel(f"Ctrl+Alt+T · siap")
        self.status.setStyleSheet(f"color: {C['faint']}; font-size: 11px;")
        footer.addWidget(self.status)
        footer.addStretch(1)

        self.btn_copy = QPushButton("Salin semua")
        self.btn_copy.setVisible(False)
        self.btn_copy.clicked.connect(self._copy_all)
        footer.addWidget(self.btn_copy)
        root.addLayout(footer)

    def _build_titlebar(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("titlebar")
        bar.setFixedHeight(42)

        lay = QHBoxLayout(bar)
        lay.setContentsMargins(12, 0, 8, 0)
        lay.setSpacing(8)

        self.dot = QLabel("●")
        self.dot.setStyleSheet(f"color: {C['good']}; font-size: 11px;")
        lay.addWidget(self.dot)

        title = QLabel(APP_NAME)
        f = QFont("Segoe UI", 12)
        f.setWeight(QFont.Weight.DemiBold)
        title.setFont(f)
        lay.addWidget(title)

        lay.addStretch(1)

        pin = QPushButton("—")
        pin.setObjectName("ghost")
        pin.setToolTip("Sembunyikan (tombol tetap di taskbar)")
        pin.clicked.connect(self.hide)
        lay.addWidget(pin)

        close = QPushButton("✕")
        close.setObjectName("ghost")
        close.setToolTip("Keluar")
        close.clicked.connect(self._quit)
        lay.addWidget(close)

        bar._drag_start = None
        bar.mousePressEvent = lambda e: self._bar_press(bar, e)
        bar.mouseMoveEvent = lambda e: self._bar_move(bar, e)
        return bar

    def _bar_press(self, bar, e):
        if e.button() == Qt.MouseButton.LeftButton:
            bar._drag_start = e.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def _bar_move(self, bar, e):
        if getattr(bar, "_drag_start", None) is not None:
            self.move(e.globalPosition().toPoint() - bar._drag_start)

    # ---------------- hotkey ----------------
    def _start_hotkey(self):
        def listen():
            try:
                from pynput import keyboard
                hm = keyboard.GlobalHotKeys({HOTKEY: self._on_hotkey})
                hm.start()
            except Exception:
                # tanpa pynput / tanpa admin -> tombol tetap bisa dipakai
                pass

        threading.Thread(target=listen, daemon=True).start()

    def _on_hotkey(self):
        QTimer.singleShot(0, self.begin_capture)

    # ---------------- capture ----------------
    def begin_capture(self):
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
        self.hide()
        QTimer.singleShot(220, self._open_selector)

    def _open_selector(self):
        self._selector = RegionSelector()
        self._selector.picked.connect(self._on_picked)
        self._selector.cancelled.connect(self._show_again)
        self._selector.showFullScreen()
        self._selector.raise_()
        self._selector.activateWindow()
        QGuiApplication.setOverrideCursor(QCursor(Qt.CursorShape.CrossCursor))

    def _show_again(self):
        QGuiApplication.restoreOverrideCursor()
        self.show()
        self.raise_()
        self.activateWindow()

    def _on_picked(self, x, y, w, h):
        QGuiApplication.restoreOverrideCursor()
        QTimer.singleShot(180, lambda: self._grab_and_run(x, y, w, h))

    def _grab_and_run(self, x, y, w, h):
        try:
            import mss
        except ImportError:
            self._set_status("modul mss belum terpasang", bad=True)
            self._show_again()
            return

        try:
            dpr = self.screen().devicePixelRatio() or 1.0
            region = {
                "left": int(x * dpr), "top": int(y * dpr),
                "width": max(1, int(w * dpr)), "height": max(1, int(h * dpr)),
            }
            with mss.mss() as sct:
                shot = sct.grab(region)
            arr = np.frombuffer(shot.bgra, dtype=np.uint8).reshape(shot.height, shot.width, 4)
            bgr = np.ascontiguousarray(arr[:, :, :3])
        except Exception as exc:                        # noqa: BLE001
            self._set_status(f"Gagal capture: {exc}", bad=True)
            self._show_again()
            return

        self._show_again()
        self._before_pix = self._np_to_pixmap(bgr)

        langs = ["auto", "ko", "ja", "zh"]
        options = {
            "lang": langs[self.cmb_lang.currentIndex()],
            "do_translate": self.chk_translate.isChecked(),
            "preprocess_combined": self.chk_pre.isChecked(),
            "min_bubble_conf": 0.1,
        }

        self.btn_capture.setEnabled(False)
        self.btn_capture.setText("Memproses…")
        self.dot.setStyleSheet(f"color: {C['warn']}; font-size: 11px;")
        self._set_status("Mendeteksi bubble…")

        self._worker = OcrWorker(bgr, options, self)
        self._worker.status.connect(self._set_status)
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    # ---------------- hasil ----------------
    def _on_done(self, res: dict):
        self.btn_capture.setEnabled(True)
        self.btn_capture.setText("Tangkap Layar")
        self.dot.setStyleSheet(f"color: {C['good']}; font-size: 11px;")

        bubbles = res.get("bubbles", [])
        self._clear_list()

        for b in bubbles:
            card = BubbleCard(b)
            card.clicked.connect(lambda b=b: self._copy_one(b))
            self.list_layout.insertWidget(self.list_layout.count() - 1, card)

        replaced = res.get("replaced_file")
        if replaced and os.path.isfile(replaced):
            self._after_pix = QPixmap(replaced)
            self._show_after = True
            self._render_preview()
            self.btn_toggle.setVisible(True)

        script = (res.get("script") or {}).get("detected") or "—"
        n_tr = res.get("translated_lines", 0)
        t = res.get("_timing", {})
        self.summary.setText(
            f"{len(bubbles)} bubble · script: {script} · {n_tr} baris terjemahan · "
            f"{t.get('total', 0):.1f}s"
        )
        self.summary.setVisible(True)
        self.btn_copy.setVisible(bool(bubbles))
        self._set_status(f"Selesai dalam {t.get('total', 0):.1f}s")

    def _on_failed(self, msg: str):
        self.btn_capture.setEnabled(True)
        self.btn_capture.setText("Tangkap Layar")
        self.dot.setStyleSheet(f"color: {C['bad']}; font-size: 11px;")
        self._set_status(f"Gagal: {msg}", bad=True)

    def _set_status(self, text: str, bad: bool = False):
        self.status.setText(text)
        self.status.setStyleSheet(
            f"color: {C['bad'] if bad else C['faint']}; font-size: 11px;"
        )

    # ---------------- preview ----------------
    def _np_to_pixmap(self, bgr: np.ndarray) -> QPixmap:
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        h, w, _ = rgb.shape
        img = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(img.copy())

    def _toggle_preview(self):
        if self._before_pix is None:
            return
        self._show_after = not self._show_after
        self._render_preview()

    def _render_preview(self):
        src = self._after_pix if (self._show_after and self._after_pix) else self._before_pix
        if src is None:
            return
        self.btn_toggle.setText("Lihat asli" if self._show_after else "Lihat hasil")
        self.preview.setStyleSheet(
            f"background: {C['surface']}; border: 1px solid {C['border']};"
            f"border-radius: 10px; padding: 6px;"
        )
        self.preview.setPixmap(
            src.scaled(
                self.preview.width() - 16,
                self.preview.height() - 16,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._before_pix is not None:
            self._render_preview()

    # ---------------- clipboard ----------------
    def _copy_one(self, b: dict):
        txt = (b.get("translation") or b.get("raw_text") or "").strip()
        txt = txt.replace("[TL] ", "")
        if txt:
            QApp.clipboard().setText(txt)
            self._set_status("Tersalin")

    def _copy_all(self):
        parts = []
        for i in range(self.list_layout.count() - 1):
            w = self.list_layout.itemAt(i).widget()
            if isinstance(w, BubbleCard):
                txt = (getattr(w, "payload", "") or "").strip()
                if txt:
                    parts.append(txt)
        text = "\n".join(parts)
        if text:
            QApp.clipboard().setText(text)
            self._set_status(f"{len(parts)} baris tersalin")

    def _clear_list(self):
        while self.list_layout.count() > 1:
            item = self.list_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

    # ---------------- geometry & quit ----------------
    def _restore_geometry(self):
        g = self.settings.value("geometry")
        if g:
            self.restoreGeometry(g)
        else:
            scr = QGuiApplication.primaryScreen().availableGeometry()
            self.move(scr.right() - self.width() - 28, scr.top() + 60)
        self.show()

    def closeEvent(self, e):
        self.settings.setValue("geometry", self.saveGeometry())
        super().closeEvent(e)

    def _quit(self):
        self.settings.setValue("geometry", self.saveGeometry())
        QApp.quit()


# ==============================
def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setStyle("Fusion")
    app.setStyleSheet(QSS)

    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()