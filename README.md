# İlan Görsellerinde Telefon Numarası Tespiti

İlan fotoğraflarına yazılmış telefon numaralarını OCR ile bulan servis (PaddleOCR PP-OCRv6 small, ONNX Runtime, CPU).

## Kurulum

Python 3.11.

```bash
python3.11 -m venv venv311
venv311/bin/pip install -r requirements.txt -r requirements-export.txt
venv311/bin/python -m app.export_models          # modelleri indirip ONNX'e çevirir -> models/
```

## Çalıştırma

```bash
# Komut satırı
venv311/bin/python scan.py foto.jpg
venv311/bin/python scan.py klasor/ --json

# Web arayüzü ve API (http://localhost:8000)
OCR_CPU_THREADS=4 venv311/bin/uvicorn app.web:app --host 127.0.0.1 --port 8000 --workers 2
```

### Docker

```bash
docker build -t ocr-phone .
docker run -d --name ocr-phone --restart unless-stopped \
  -p 127.0.0.1:8000:8000 \
  -e WEB_CONCURRENCY=2 -e OCR_CPU_THREADS=4 \
  ocr-phone

curl http://localhost:8000/health                # {"status":"ok"}
```

Durdurmak: `docker stop ocr-phone` · yeniden başlatmak: `docker start ocr-phone`


## API

```bash
# Tek görsel
curl -F "file=@foto.jpg" http://localhost:8000/api/scan

# Bir ilanın fotoğrafları (numara bulunan ilk fotoğrafta durur)
curl -F "files=@1.jpg" -F "files=@2.jpg" -F "files=@3.jpg" http://localhost:8000/api/scan-listing

# Sağlık kontrolü
curl http://localhost:8000/health
```
