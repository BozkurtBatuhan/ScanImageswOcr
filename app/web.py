"""FastAPI uygulaması: fotoğraf yükleme + telefon numarası tespiti.

Her uvicorn worker'ı (ayrı süreç) kendi OCR örneğini yükler ve aynı anda tek görsel tarar;
eşzamanlılık worker sayısıyla (WEB_CONCURRENCY / --workers) ölçeklenir.

/api/scan-listing bir ilanın fotoğraflarını sırayla tarar, ilk numarada durur.

Koruma: OCR_MAX_MB'den büyük dosya taranmaz (413 / dosya bazında hata). Worker'ın tarayıcısı
OCR_QUEUE_TIMEOUT_S içinde boşalmazsa 503 + Retry-After döner (yük dengeleyici başka worker'a yönlendirsin)."""
from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from app.ocr import OCR
from app.scanner import Result, scan

TEMPLATE = (Path(__file__).parent / "templates" / "index.html").read_text(encoding="utf-8")
MAX_BYTES = int(float(os.getenv("OCR_MAX_MB", "20")) * (1 << 20))
QUEUE_TIMEOUT_S = float(os.getenv("OCR_QUEUE_TIMEOUT_S", "10"))
ocr: OCR | None = None
lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ocr
    ocr = OCR()
    yield


app = FastAPI(title="İlan Görsel Telefon Kontrolü", lifespan=lifespan)


class TooLarge(Exception):
    pass


def too_large_message() -> str:
    return f"dosya {MAX_BYTES / (1 << 20):.3g} MB sınırını aşıyor"


async def read_upload(file: UploadFile) -> bytes:
    data = await file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise TooLarge
    return data


async def scan_bytes(data: bytes) -> Result:
    img = await asyncio.to_thread(cv2.imdecode, np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return Result(error="görsel okunamadı")
    try:
        await asyncio.wait_for(lock.acquire(), QUEUE_TIMEOUT_S)
    except asyncio.TimeoutError:
        raise HTTPException(503, "tarayıcı meşgul, tekrar deneyin", headers={"Retry-After": "1"})
    try:
        return await asyncio.to_thread(scan, img, ocr)
    finally:
        lock.release()


@app.post("/api/scan")
async def scan_upload(file: UploadFile = File(...)) -> dict:
    try:
        data = await read_upload(file)
    except TooLarge:
        raise HTTPException(413, too_large_message())
    result = await scan_bytes(data)
    return {"file": file.filename, **asdict(result)}


@app.post("/api/scan-listing")
async def scan_listing(files: list[UploadFile] = File(...)) -> dict:
    """Bir ilanın fotoğrafları, gönderilen sırayla. İlk numara bulunan fotoğrafta durur;
    kalanlar taranmaz ("skipped")."""
    t0 = time.perf_counter()
    results: list[dict] = []
    hit: dict | None = None
    for f in files:
        if hit:
            results.append({"file": f.filename, "skipped": True})
            continue
        try:
            result = await scan_bytes(await read_upload(f))
        except TooLarge:
            result = Result(error=too_large_message())
        results.append({"file": f.filename, "skipped": False, **asdict(result)})
        if result.has_phone:
            hit = results[-1]
    return {
        "has_phone": hit is not None,
        "phones": hit["phones"] if hit else [],
        "found_in": hit["file"] if hit else None,
        "suspicious": any(r.get("suspicious") for r in results),
        "scanned": sum(not r["skipped"] for r in results),
        "skipped": sum(r["skipped"] for r in results),
        "ms": round((time.perf_counter() - t0) * 1000),
        "results": results,
    }


@app.get("/health")
def health():
    """Yük dengeleyici kontrolü: model yüklendiyse 200."""
    if ocr is None:
        return JSONResponse({"status": "loading"}, status_code=503)
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return TEMPLATE
