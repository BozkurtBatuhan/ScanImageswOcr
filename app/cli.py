from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from app.ocr import OCR
from app.scanner import collect, scan_file


def main() -> None:
    ap = argparse.ArgumentParser(description="İlan görsellerinde telefon numarası tespiti")
    ap.add_argument("paths", nargs="+", help="görsel dosyaları veya klasörler")
    ap.add_argument("--json", action="store_true", help="JSON çıktı ver")
    args = ap.parse_args()

    ocr = OCR()
    rows = []
    for f in collect(args.paths):
        r = scan_file(f, ocr)
        rows.append({"file": str(f), **asdict(r)})
        if not args.json:
            status = "HATA" if r.error else "NUMARA" if r.has_phone else "ŞÜPHELİ" if r.suspicious else "TEMİZ"
            print(f"{status:8} {r.ms:5} ms  {f.name}  {', '.join(r.phones) or r.error}")
    if args.json:
        json.dump(rows, sys.stdout, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
