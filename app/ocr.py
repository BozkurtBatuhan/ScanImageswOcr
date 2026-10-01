from __future__ import annotations

import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from app.detection import db_boxes, det_resize, faint_boxes
from app.imaging import scale_to

DET_MODEL = os.getenv("OCR_DET_MODEL", "PP-OCRv6_small_det")
REC_MODEL = os.getenv("OCR_REC_MODEL", "PP-OCRv6_small_rec")
MODEL_DIR = Path(os.getenv("OCR_MODEL_DIR", str(Path(__file__).resolve().parent.parent / "models")))
DET_SIDE = int(os.getenv("OCR_DET_SIDE", "640"))
MIN_DET_SIDE = 640
CROP_SIDE = 2000
CPU_THREADS = int(os.getenv("OCR_CPU_THREADS", "4"))
REC_BATCH = 8
MAX_REC_WIDTH = 3200


def load_model(model_dir: Path, name: str, threads: int):
    d = model_dir / name
    if not (d / "inference.onnx").exists():
        raise FileNotFoundError(f"{d}/inference.onnx yok; önce üretin: python -m app.export_models {model_dir}")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(str(d / "inference.onnx"), sess_options=opts, providers=["CPUExecutionProvider"])
    return session, json.loads((d / "config.json").read_text(encoding="utf-8"))


class OCR:

    def __init__(self, det_model: str = DET_MODEL, rec_model: str = REC_MODEL, threads: int = CPU_THREADS,
                 model_dir: Path = MODEL_DIR):
        model_dir = Path(model_dir)
        self.det, self.det_cfg = load_model(model_dir, det_model, threads)
        self.rec, rec_cfg = load_model(model_dir, rec_model, threads)
        c = self.det_cfg
        self.alpha = np.float32([c["scale"] / s for s in c["std"]])
        self.beta = np.float32([-m / s for m, s in zip(c["mean"], c["std"])])
        self.chars = ["blank"] + rec_cfg["characters"] + [" "]
        _, self.rec_h, self.rec_w = rec_cfg["image_shape"]
        warm = np.full((64, 256, 3), 255, np.uint8)
        cv2.putText(warm, "0532", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
        self.detect(warm)
        self.read([warm])

    def detect(self, img, side: int = DET_SIDE, thresh: float = 0.2, box_thresh: float = 0.4,
               faint: bool = False):
        small, s = scale_to(img, side)
        x, (h, w) = det_resize(small, side)
        x = (x.astype(np.float32) * self.alpha + self.beta).transpose(2, 0, 1)[None]
        pred = self.det.run(None, {self.det.get_inputs()[0].name: np.ascontiguousarray(x)})[0][0, 0]
        boxes = db_boxes(pred, thresh, box_thresh, self.det_cfg["unclip_ratio"], self.det_cfg["max_candidates"], w, h)
        boxes = [b.astype(np.float32) / s for b in boxes]
        if not faint:
            return boxes
        return boxes, [b.astype(np.float32) / s for b in faint_boxes(pred, w, h)]

    def rec_input(self, img) -> np.ndarray:
        h, w = img.shape[:2]
        width = int(self.rec_h * max(self.rec_w / self.rec_h, w / h))
        if width > MAX_REC_WIDTH:
            rw = width = MAX_REC_WIDTH
        else:
            rw = min(int(math.ceil(self.rec_h * w / h)), width)
        x = cv2.resize(np.ascontiguousarray(img), (rw, self.rec_h)).astype(np.float32).transpose(2, 0, 1) / 255
        out = np.zeros((3, self.rec_h, width), np.float32)
        out[:, :, :rw] = (x - 0.5) / 0.5
        return out

    def decode(self, pred) -> list[tuple[str, float]]:
        out = []
        for idx, prob in zip(pred.argmax(-1), pred.max(-1)):
            keep = np.ones(len(idx), bool)
            keep[1:] = idx[1:] != idx[:-1]
            keep &= idx != 0
            out.append(("".join(self.chars[i] for i in idx[keep]), float(np.mean(prob[keep])) if keep.any() else 0.0))
        return out

    def read(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        order = sorted(range(len(crops)), key=lambda i: crops[i].shape[1] / crops[i].shape[0])
        out: list[tuple[str, float]] = [("", 0.0)] * len(crops)
        name = self.rec.get_inputs()[0].name
        for k in range(0, len(order), REC_BATCH):
            idx = order[k:k + REC_BATCH]
            xs = [self.rec_input(crops[i]) for i in idx]
            width = max(x.shape[2] for x in xs)
            batch = np.stack([np.pad(x, ((0, 0), (0, 0), (0, width - x.shape[2]))) for x in xs])
            for i, r in zip(idx, self.decode(self.rec.run(None, {name: batch})[0])):
                out[i] = r
        return out
