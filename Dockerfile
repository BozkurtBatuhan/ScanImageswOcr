# 1) Model aşaması: resmi PP-OCR modellerini indirip ONNX'e çevirir (paddle yalnızca burada).
# Her zaman amd64: paddle2onnx bağımlılıklarının (onnxoptimizer) arm64 Linux paketi yok. ONNX dosyaları
# mimariden bağımsız, çalışma imajı hedef mimaride (x86 / Graviton) kurulur.
FROM --platform=linux/amd64 python:3.11-slim AS models

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /srv
COPY requirements.txt requirements-export.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-export.txt
COPY app ./app
RUN python -m app.export_models /models

# 2) Çalışma aşaması: yalnızca onnxruntime + OpenCV.
FROM python:3.11-slim

WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scan.py .
COPY --from=models /models ./models

# Modellerin yüklendiğini build sırasında doğrular.
RUN python -c "from app.ocr import OCR; OCR()"

# Worker sayısı verilmezse çekirdek sayısı / OCR_CPU_THREADS (en az 1). CPU limiti olan ortamlarda
# (K8s) nproc makinenin tüm çekirdeklerini görebilir; orada WEB_CONCURRENCY'yi açıkça verin.
ENV OCR_CPU_THREADS=4

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"
CMD ["sh", "-c", "W=${WEB_CONCURRENCY:-$(( $(nproc) / OCR_CPU_THREADS ))}; [ \"$W\" -ge 1 ] || W=1; exec uvicorn app.web:app --host 0.0.0.0 --port 8000 --workers $W"]
