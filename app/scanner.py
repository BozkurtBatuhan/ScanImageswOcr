from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.geometry import Rect, iou, joined, lines_of, merge, roi
from app.imaging import clahe, crop, invert_otsu, load_image, scale_to
from app.ocr import CROP_SIDE, DET_SIDE, MIN_DET_SIDE, OCR
from app.phones import digitish, extract, fragment
from app.text import normalize

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEADLINE_MS = int(os.getenv("OCR_DEADLINE_MS", "500"))
SMALL_TEXT_PX = 12
UNSURE_SCORE = 0.8
ZOOM_MX = 8
HI_SIDE = int(os.getenv("OCR_HI_SIDE", "960"))
HI_BELOW = float(os.getenv("OCR_HI_BELOW", "0.4"))


@dataclass
class Result:
    phones: list[str] = field(default_factory=list)
    has_phone: bool = False
    stage: str = ""
    suspicious: bool = False
    timed_out: bool = False
    ms: int = 0
    timings: dict[str, int] = field(default_factory=dict)
    error: str = ""


def zoom_regions(polys: list[np.ndarray], texts: list[str], unsure: set[int], faint: list[np.ndarray], shape,
                 det_scale: float) -> list[Rect]:
    seeds = [i for i, (p, t) in enumerate(zip(polys, texts))
             if i in unsure or fragment(t)
             or (np.ptp(p[:, 1]) * det_scale < SMALL_TEXT_PX and any(c.isdigit() for c in normalize(t)))]
    rects = [(*roi(polys, [i], shape, mx=ZOOM_MX), i in unsure) for i in seeds]
    rects += [(*roi(faint, [i], shape, mx=ZOOM_MX), False) for i in range(len(faint))]
    return merge([r for r in rects if r[2] > r[0] and r[3] > r[1]])


def scan(img, ocr: OCR, deadline_ms: int = DEADLINE_MS) -> Result:
    t0 = time.perf_counter()
    r = Result()
    last = [t0]
    step = ["A"]

    def lap(name: str) -> None:
        now = time.perf_counter()
        r.timings[name] = r.timings.get(name, 0) + round((now - last[0]) * 1000)
        last[0], step[0] = now, name

    def late() -> bool:
        if deadline_ms > 0 and (time.perf_counter() - t0) * 1000 > deadline_ms:
            r.timed_out = True
        return r.timed_out

    def done(phones: list[str], stage: str = "") -> Result:
        lap(step[0])
        r.phones, r.has_phone, r.stage = phones, bool(phones), stage if phones else ""
        r.suspicious = (r.suspicious or r.timed_out) and not phones
        r.ms = round((time.perf_counter() - t0) * 1000)
        return r

    def detect_read(base, fn, side: int) -> list[str]:
        polys = ocr.detect(fn(base) if fn else base, side=side)
        if not polys:
            return []
        raw = [crop(base, p) for p in polys]
        ln = lines_of(polys)
        reads = ocr.read(raw)
        texts = [t for t, _ in reads]
        out = extract(joined(texts, ln))[0]
        if out or fn is None:
            return out
        # Filtreli görünümden yalnızca belirsiz ya da rakamlı kutular yeniden okunur; güvenle okunmuş
        # rakamsız kutunun (ör. "SATILIK", "12") filtreli okuması aynı ya da daha kötü çıkıyor.
        redo = [i for i, (t, sc) in enumerate(reads) if sc < UNSURE_SCORE or digitish(t)]
        if not redo:
            return []
        for i, (t, _) in zip(redo, ocr.read([fn(raw[i]) for i in redo])):
            texts[i] = t
        return extract(joined(texts, ln))[0]

    native = max(img.shape[:2])
    src, _ = scale_to(img, min(max(native, DET_SIDE), CROP_SIDE))
    side = min(max(native, MIN_DET_SIDE), DET_SIDE)

    det_scale = side / max(src.shape[:2])
    polys, faint = ocr.detect(src, side=side, faint=True)
    crops = [crop(src, p) for p in polys]
    reads = ocr.read(crops)
    texts = [t for t, _ in reads]
    lines = lines_of(polys)
    held, strong = extract(joined(texts, lines))
    unsure = {i for i, (t, sc) in enumerate(reads) if sc < UNSURE_SCORE and digitish(t)}
    phones = extract(joined(["" if i in unsure else t for i, t in enumerate(texts)], lines))[0] if unsure else held
    held = [p for p in held if p not in phones]
    lap("A")
    found_in = "full" if phones else ""
    keep_held = False
    step[0] = "Z"
    regions = zoom_regions(polys, texts, unsure, faint, src.shape, det_scale)
    for x0, y0, x1, y1, has_unsure in regions:
        if late():
            keep_held = True
            break
        region = src[y0:y1, x0:x1]
        got = detect_read(region, clahe, min(max(region.shape[:2]) * 2, DET_SIDE))
        keep_held |= has_unsure and not got
        new = [p for p in got if p not in phones]
        phones += new
        found_in = found_in or ("zoom" if new else "")
    if keep_held and held:
        phones += [p for p in held if p not in phones]
        found_in = found_in or "full"
    if regions:
        lap("Z")

    hi = min(max(src.shape[:2]), HI_SIDE)
    can_hi = HI_SIDE > 0 and hi > 1.3 * side

    def hi_pass() -> list[str]:
        hp = ocr.detect(src, side=hi)
        ht: list[str | None] = [None] * len(hp)
        for j, p in enumerate(hp):
            k = max(range(len(polys)), key=lambda k: iou(p, polys[k]), default=None)
            if k is not None and k not in unsure and iou(p, polys[k]) > 0.5:
                ht[j] = texts[k]
        need = [j for j, t in enumerate(ht) if t is None]
        for j, (t, _) in zip(need, ocr.read([crop(src, hp[j]) for j in need])):
            ht[j] = t
        return extract(joined(ht, lines_of(hp)))[0]

    if can_hi and not phones and det_scale < HI_BELOW and not late():
        step[0] = "H"
        if phones := hi_pass():
            found_in = "hi"
        lap("H")
        can_hi = False
    if phones:
        return done(phones, found_in)

    r.suspicious = strong
    if strong:
        step[0] = "C"
        passes = []
        cand = [i for i, t in enumerate(texts) if digitish(t) or extract(t)[1]]
        if cand:
            x0, y0, x1, y1 = roi(polys, cand, src.shape)
            region = src[y0:y1, x0:x1]
            rside = min(max(region.shape[:2]) * 2, DET_SIDE)
            if region.size:
                passes += [("roi", region, clahe, rside), ("roi_inv", region, invert_otsu, rside)]
        passes.append(("inv", src, invert_otsu, MIN_DET_SIDE))
        for name, base, fn, sd in passes:
            if late():
                break
            if phones := detect_read(base, fn, sd):
                return done(phones, name)
        if can_hi and not late():
            lap("C")
            step[0] = "H"
            if phones := hi_pass():
                return done(phones, "hi")
    return done([])


def scan_file(path: Path, ocr: OCR) -> Result:
    img = load_image(path)
    if img is None:
        return Result(error="görsel okunamadı")
    return scan(img, ocr)


def collect(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for p in map(Path, paths):
        files += sorted(f for f in p.rglob("*") if f.suffix.lower() in IMAGE_EXTS) if p.is_dir() else [p]
    return files
