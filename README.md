# İlan Görsellerinde Telefon Numarası Tespiti

İlan fotoğraflarına yazılmış Türkiye telefon numaralarını OCR ile bulan servis. PaddleOCR PP-OCRv6 small modelleri
ONNX Runtime ile yalnızca CPU'da çalışır; çalışma zamanında Paddle ya da GPU gerekmez. Web arayüzü, HTTP API ve
komut satırı aracı içerir.

Her görsel için sonuç:

| Durum | Anlamı |
|---|---|
| `NUMARA` | Numara bulundu; standart biçimde döner (`05321234567`) |
| `ŞÜPHELİ` | Numara bulunamadı ama şüphe var (ipucu kelime, eksik numara, süre sınırı); moderatöre gider |
| `TEMİZ` | Numara izi yok |
| `HATA` | Görsel okunamadı |

## Nasıl çalışır

Tarama ucuzdan pahalıya kademelidir (`app/scanner.py`):

- **A · Hızlı tarama:** her görselde 640 px'te metin tespiti + okuma.
- **Z · Yakın bakış:** küçük, soluk ya da düşük güvenle okunmuş yazıların bölgesi 2 kat büyütülüp yeniden taranır.
  Numara bulunmuş olsa da çalışır; görseldeki diğer numaralar da çıkar.
- **H · Yüksek çözünürlük:** numara yoksa, uzun kenarı 1600 px'i aşan fotoğraflarda 960 px'te ikinci tespit.
- **C · Güçlü şüphe:** "arayın", "whatsapp", "tel" gibi ipucu ya da eksik numara varsa şüpheli bölge CLAHE ve
  ters-ikili görünümlerle yeniden taranır.

Görsel başına süre sınırı vardır (`OCR_DEADLINE_MS`, varsayılan 500 ms). A her zaman tamamlanır; sınır aşılınca kalan
kademeler atlanır ve numara bulunmadıysa sonuç `ŞÜPHELİ` olur.

Okunan metinden numaralar `app/phones.py` ile çıkarılır: cep (`05…`, `+90 5…`, baştaki 0 eksik), sabit hat
(`02/03/04…`), `0850/0800`, `444 XXXX`; bitişik yazılmış iki numara ayrılır. Fiyat, m², ilan / vergi / portföy /
yetki belge numaraları elenir. Türkçe karakterler, rakama benzeyen harfler (O→0, l→1, S→5), daire içi rakamlar ve
yazıyla sayılar ("sıfır beş yüz otuz iki") normalleştirilir (`app/text.py`).

## Kurulum

Python 3.11.

```bash
python3.11 -m venv venv311
venv311/bin/pip install -r requirements.txt -r requirements-export.txt
venv311/bin/python -m app.export_models          # modelleri indirip ONNX'e çevirir -> models/
```

`requirements-export.txt` (paddle, paddlex, paddle2onnx) yalnızca model dışa aktarma için gerekir. Modeller
Hugging Face'teki resmi `PaddlePaddle/PP-OCRv6_small_det` ve `PaddlePaddle/PP-OCRv6_small_rec` depolarından iner.

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

İmaj iki aşamalıdır: ilk aşama modelleri ONNX'e çevirir (paddle2onnx'in arm64 Linux paketi olmadığı için
`linux/amd64`'e sabit; Apple Silicon'da emülasyonla çalışır, ilk build uzun sürebilir). Çalışma imajı yalnızca
ONNX Runtime + OpenCV içerir ve hedef mimaride kurulur. Mac'te Docker Desktop'a en az 8 CPU ayırın.

### İnternete açmak (ngrok)

```bash
ngrok http 8000 --basic-auth="demo:GucluBirSifre123"
```

Serviste kimlik doğrulama yoktur; dışarı açarken mutlaka `--basic-auth` kullanın.

## API

### `POST /api/scan` — tek görsel

```bash
curl -F "file=@foto.jpg" http://localhost:8000/api/scan
```

```json
{"file": "foto.jpg", "phones": ["05321234567"], "has_phone": true, "stage": "full",
 "suspicious": false, "timed_out": false, "ms": 86, "timings": {"A": 80, "Z": 6}, "error": ""}
```

- `phones`: görseldeki tüm numaralar. `stage`: ilk numaranın bulunduğu kademe (`full`, `zoom`, `hi`, `roi`,
  `roi_inv`, `inv`).
- `ms`: yalnızca tarama süresi (tüm kademeler, tespit + okuma); yükleme, görsel çözme ve kuyrukta bekleme hariç.

### `POST /api/scan-listing` — bir ilanın fotoğrafları

```bash
curl -F "files=@1.jpg" -F "files=@2.jpg" -F "files=@3.jpg" http://localhost:8000/api/scan-listing
```

Fotoğraflar gönderilen sırayla taranır; numara bulunan ilk fotoğrafta durulur, kalanlar `skipped` olur. Yanıt:
`has_phone`, `phones` ve `found_in` (ilk numaralı fotoğraf), `suspicious` (taranan herhangi biri şüpheliyse),
`scanned`, `skipped`, `ms` (isteğin toplam süresi) ve fotoğraf başına `results`.

### `GET /health`

Model yüklendiyse `200 {"status": "ok"}`; yük dengeleyici / Docker HEALTHCHECK için.

### Koruma

- `OCR_MAX_MB`'den büyük dosya taranmaz: tek görselde `413`, ilan taramasında dosya bazında `error`.
- Her worker aynı anda tek görsel tarar. Worker `OCR_QUEUE_TIMEOUT_S` içinde boşalmazsa `503` + `Retry-After: 1`.
  İlan taramasında bu durum isteğin tamamını düşürür.

## Ortam değişkenleri

| Değişken | Varsayılan | Açıklama |
|---|---|---|
| `OCR_DET_MODEL` | `PP-OCRv6_small_det` | Tespit modeli |
| `OCR_REC_MODEL` | `PP-OCRv6_small_rec` | Okuma modeli |
| `OCR_MODEL_DIR` | `models/` | ONNX modellerin klasörü (Docker'da `/srv/models`) |
| `OCR_DET_SIDE` | `640` | A kademesinde tespit girdisinin uzun kenarı (px) |
| `OCR_HI_SIDE` | `960` | H kademesinin uzun kenarı; `0` = kapalı |
| `OCR_HI_BELOW` | `0.4` | H, tespit ölçeği bunun altındaysa çalışır (≈ 1600 px üstü fotoğraflar) |
| `OCR_DEADLINE_MS` | `500` | Görsel başı süre sınırı; `0` = sınırsız |
| `OCR_CPU_THREADS` | `4` | Worker başına CPU thread sayısı |
| `OCR_MAX_MB` | `20` | Kabul edilen en büyük dosya |
| `OCR_QUEUE_TIMEOUT_S` | `10` | Worker'ı bekleme süresi; aşılınca `503` |
| `WEB_CONCURRENCY` | `nproc / OCR_CPU_THREADS` | Docker'da uvicorn worker sayısı; CPU limitli ortamlarda (K8s) açıkça verin |

## Performans

Apple M1 Pro (10 çekirdek), CPU, 121 etiketli test görseli (26 numaralı, 95 numarasız):

| Yakalanan görsel | Numara recall | Yanlış alarm | Ort / p50 / p95 / max | CPU / görsel | Tepe RAM |
|---|---|---|---|---|---|
| 25/26 | 28/29 | 0/95 | 115 / 86 / 251 / 379 ms | 0,50 sn | ~615 MB |

Kaçırılan tek görsel desenli zemin üstüne yarı saydam yazı; tespit modeli yazıyı göremiyor.

Kapasite (yalnızca tarama, tüm worker'lar dolu):

| Worker × thread | Görsel / sn | Görsel başı süre |
|---|---|---|
| 2 × 4 | ~14,5 | ~135 ms |
| 5 × 2 | ~19 | ~255 ms |
| 10 × 1 | ~22 | ~440–480 ms (500 ms sınırına yakın) |

Eşzamanlı yük testi (2 × 4, en yavaş görsellerle): 24 eşzamanlı istek 503 almadan en fazla ~3 sn'de yanıtlanır;
503 ~85 eşzamanlı istekten sonra başlar.

### Model seçimi

Akış değiştirilmeden 17 alternatif aynı koşullarda (500 ms süre sınırı, aynı test setleri) denendi: 10 PaddleOCR
model kombinasyonu (PP-OCRv4/v5/v6; tiny, mobile, small, medium, server; latin/en okuma) ve 7 OCR paketi (RapidOCR,
OpenOCR, ddddocr, CnOCR, OnnxTR, GLM-OCR, Florence-2). Hiçbiri PP-OCRv6 small'un numara recall'unu geçmedi.
Hızlı modeller 5–9 numara kaçırıyor; büyük modeller ve VLM'ler 3–16 kat yavaş ve doğruluk kazandırmıyor. Aynı
recall'a ulaşan tek alternatif (v6 small tespit + v6 medium okuma) 2 kat yavaş.

## Değerlendirme

`testdata/` (etiketli test görselleri ve `evaluate.py`) ile `ProdData/` gizlilik nedeniyle depoda yok
(`.gitignore`). Klasör mevcutsa:

```bash
venv311/bin/python testdata/evaluate.py -v
```

Yakalanan / kaçırılan / eksik-yanlış numara, numara bazında recall, yanlış alarm, süre (ort / p50 / p95 / max), CPU,
RAM ve kademe dağılımını raporlar.

## Dosyalar

| Dosya | İçerik |
|---|---|
| `app/scanner.py` | Kademeli tarama akışı (A / Z / H / C), süre sınırı, `Result` |
| `app/ocr.py` | `OCR` sınıfı: model yükleme, tespit, okuma (ONNX Runtime) |
| `app/detection.py` | Tespit ön/son işleme (DB kutu çıkarma, soluk iz) |
| `app/phones.py` | Metinden numara çıkarma ve standart biçime çevirme |
| `app/text.py` | Türkçe normalizasyon, rakam benzeri harfler, yazıyla sayılar |
| `app/geometry.py`, `app/imaging.py` | Satır gruplama, bölge hesapları, kırpma ve görüntü filtreleri |
| `app/web.py`, `app/templates/index.html` | FastAPI servisi ve web arayüzü |
| `app/cli.py`, `scan.py` | Komut satırı aracı |
| `app/export_models.py` | Resmi PaddleOCR modellerini ONNX'e çevirir |
| `Dockerfile` | İki aşamalı CPU imajı |

## Bilinen sınırlar

- Desenli zemin üstüne yarı saydam yazı tespit edilemiyor.
- Çok küçük görsellerde (ör. 64×48) rakamlar okunamaz; taramaya orijinal fotoğraf verilmeli.
- Numara bulunan görsellerde H ve C çalışmaz; yalnızca H'nin görebileceği ikinci bir numara aranmaz.
- İlan taramasında ilk numaralı fotoğrafta durulur; sonraki fotoğraflardaki farklı numaralar dönmez.
- Test verisi az; gerçek kaçırma oranı moderatörün reddettiği ilanlarla ölçülmeli.
