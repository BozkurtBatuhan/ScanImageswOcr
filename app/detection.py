from __future__ import annotations

import math

import cv2
import numpy as np
import pyclipper


def det_resize(img, side: int):
    src = img.shape[:2]
    if sum(src) < 64:
        pad = np.zeros((max(32, src[0]), max(32, src[1]), 3), np.uint8)
        pad[:src[0], :src[1]] = img
        img = pad
    h, w = img.shape[:2]
    ratio = side / max(h, w) if max(h, w) > side else 1.0
    rh = max(int(round(int(h * ratio) / 32) * 32), 32)
    rw = max(int(round(int(w * ratio) / 32) * 32), 32)
    return img if (rh, rw) == (h, w) else cv2.resize(img, (rw, rh)), src


def mini_box(contour) -> tuple[np.ndarray, float]:
    rect = cv2.minAreaRect(contour)
    p = sorted(list(cv2.boxPoints(rect)), key=lambda x: x[0])
    a, d = (0, 1) if p[1][1] > p[0][1] else (1, 0)
    b, c = (2, 3) if p[3][1] > p[2][1] else (3, 2)
    return np.array([p[a], p[b], p[c], p[d]]), min(rect[1])


def box_score(pred, box) -> float:
    h, w = pred.shape
    box = box.copy()
    x0 = max(0, min(math.floor(box[:, 0].min()), w - 1))
    x1 = max(0, min(math.ceil(box[:, 0].max()), w - 1))
    y0 = max(0, min(math.floor(box[:, 1].min()), h - 1))
    y1 = max(0, min(math.ceil(box[:, 1].max()), h - 1))
    mask = np.zeros((y1 - y0 + 1, x1 - x0 + 1), np.uint8)
    box[:, 0] -= x0
    box[:, 1] -= y0
    cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
    return cv2.mean(pred[y0:y1 + 1, x0:x1 + 1], mask)[0]


def unclip(box, ratio: float) -> np.ndarray:
    dist = cv2.contourArea(box) * ratio / cv2.arcLength(box, True)
    off = pyclipper.PyclipperOffset()
    off.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    try:
        return np.array(off.Execute(dist))
    except ValueError:
        return np.array(off.Execute(dist)[0])


def db_boxes(pred, thresh: float, box_thresh: float, unclip_ratio: float, max_candidates: int,
             dest_w: int, dest_h: int) -> list[np.ndarray]:
    h, w = pred.shape
    sx, sy = dest_w / w, dest_h / h
    contours, _ = cv2.findContours(((pred > thresh) * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for contour in contours[:max_candidates]:
        pts, sside = mini_box(contour)
        if sside < 3:
            continue
        if box_score(pred, pts.reshape(-1, 2)) < box_thresh:
            continue
        box, sside = mini_box(unclip(pts, unclip_ratio).reshape(-1, 1, 2))
        if sside < 5:
            continue
        box[:, 0] = np.clip(np.round(box[:, 0] * sx), 0, dest_w)
        box[:, 1] = np.clip(np.round(box[:, 1] * sy), 0, dest_h)
        out.append(box.astype(np.int16))
    return out


def faint_boxes(pred, dest_w: int, dest_h: int, lo: float = 0.08, max_h: int = 10) -> list[np.ndarray]:
    h, w = pred.shape
    sx, sy = dest_w / w, dest_h / h
    n, lab, stats, _ = cv2.connectedComponentsWithStats((pred > lo).astype(np.uint8))
    out = []
    for k in range(1, n):
        x, y, bw, bh, area = stats[k]
        if area < 15 or bw < 2.5 * bh or bh > max_h:
            continue
        if pred[y:y + bh, x:x + bw][lab[y:y + bh, x:x + bw] == k].mean() >= 0.3:
            continue
        out.append(np.float32([[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]]) * [sx, sy])
    return out
