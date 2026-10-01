"""Görüntü yardımcıları: yükleme, ölçekleme, düzleştirerek kırpma ve okuma öncesi görünüm dönüşümleri."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def load_image(path: Path):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def scale_to(img, side: int):
    h, w = img.shape[:2]
    s = side / max(h, w)
    if abs(s - 1) < 0.05:
        return img, 1.0
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC), s


def crop(img, poly) -> np.ndarray:
    """Dörtgen kutuyu düzleştirerek kırpar (eğik yazı da yatay hale gelir)."""
    poly = np.asarray(poly, dtype=np.float32)
    w = int(max(np.linalg.norm(poly[0] - poly[1]), np.linalg.norm(poly[2] - poly[3])))
    h = int(max(np.linalg.norm(poly[0] - poly[3]), np.linalg.norm(poly[1] - poly[2])))
    w, h = max(w, 4), max(h, 4)
    m = cv2.getPerspectiveTransform(poly, np.float32([[0, 0], [w, 0], [w, h], [0, h]]))
    out = cv2.warpPerspective(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return np.rot90(out) if h > 1.5 * w and len(poly) == 4 else out


def clahe(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4)).apply(gray)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def invert_otsu(img):
    """İkili + ters: renkli rozet/daire içindeki açık renkli rakamları koyu-üstü-açık yazıya çevirir."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(255 - bw, cv2.COLOR_GRAY2BGR)
