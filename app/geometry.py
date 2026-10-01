"""Kutu/bölge geometrisi: satır gruplama, ilgi bölgesi (ROI) çıkarma, çakışma ölçme ve bölge birleştirme."""
from __future__ import annotations

import numpy as np

Rect = tuple[int, int, int, int, bool]  # x0, y0, x1, y1, bölgede emin olunmayan okuma var mı


def lines_of(polys: list[np.ndarray]) -> list[list[int]]:
    """Kutuları satırlara böler, satır içinde soldan sağa sıralar. Aynı satırda olup arası
    açık kutular (> 2 yazı yüksekliği) ayrı satır sayılır: yalnızca yan yana parçalar birleşsin."""
    if not polys:
        return []
    ys = [p[:, 1].mean() for p in polys]
    hs = [max(np.ptp(p[:, 1]), 1) for p in polys]
    idx = sorted(range(len(polys)), key=lambda i: ys[i])
    rows: list[list[int]] = [[idx[0]]]
    for i in idx[1:]:
        j = rows[-1][-1]
        (rows[-1].append(i) if abs(ys[i] - ys[j]) < 0.5 * min(hs[i], hs[j]) else rows.append([i]))
    out: list[list[int]] = []
    for row in rows:
        row.sort(key=lambda k: polys[k][:, 0].min())
        out.append([row[0]])
        for a, b in zip(row, row[1:]):
            gap = polys[b][:, 0].min() - polys[a][:, 0].max()
            (out[-1].append(b) if gap < 2 * max(hs[a], hs[b]) else out.append([b]))
    return out


def joined(texts: list[str], lines: list[list[int]]) -> str:
    return "\n".join(" ".join(texts[i] for i in ln) for ln in lines)


def roi(polys: list[np.ndarray], idx: list[int], shape, mx: float = 4, my: float = 3) -> tuple[int, int, int, int]:
    """Şüpheli kutuları kapsayan bölge, çevresiyle birlikte (numara genelde ipucu yazısının hemen altında/üstünde)."""
    pts = np.concatenate([polys[i] for i in idx])
    h = max(np.median([np.ptp(polys[i][:, 1]) for i in idx]), 8)
    x0, y0 = pts.min(0) - [mx * h, my * h]
    x1, y1 = pts.max(0) + [mx * h, my * h]
    H, W = shape[:2]
    return int(max(x0, 0)), int(max(y0, 0)), int(min(x1, W)), int(min(y1, H))


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """İki kutunun eksen hizalı çerçevelerinin kesişim / birleşim oranı."""
    (ax0, ay0), (ax1, ay1) = a.min(0), a.max(0)
    (bx0, by0), (bx1, by1) = b.min(0), b.max(0)
    inter = max(0, min(ax1, bx1) - max(ax0, bx0)) * max(0, min(ay1, by1) - max(ay0, by0))
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union > 0 else 0.0


def merge(rects: list[Rect]) -> list[Rect]:
    """Çakışan bölgeleri birleştirir (aynı yere iki kez bakılmasın); son alan: bölgede emin olunmayan okuma var."""
    out: list[list] = []
    for r in sorted(rects):
        for m in out:
            if r[0] < m[2] and r[2] > m[0] and r[1] < m[3] and r[3] > m[1]:
                m[:] = min(m[0], r[0]), min(m[1], r[1]), max(m[2], r[2]), max(m[3], r[3]), m[4] or r[4]
                break
        else:
            out.append(list(r))
    return [tuple(m) for m in out] if len(out) == len(rects) else merge([tuple(m) for m in out])
