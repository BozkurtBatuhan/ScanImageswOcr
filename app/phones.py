"""OCR metninden Türkiye telefon numaralarını çıkarıp standart biçime (05321234567) çevirir."""
from __future__ import annotations

import re

from app.text import normalize

RUN = re.compile(r"\d(?:(?:[^\w\n]|_){0,3}\d){5,}")
LOOSE_RUN = re.compile(r"\d(?:[^a-z0-9\n]{0,8}\d){9,}")
MOBILE_PREFIXED = re.compile(r"(?:90|0)5\d{9}")
MOBILE_CORE = re.compile(r"5\d{9}$")
LANDLINE = re.compile(r"0[234]\d{9}")
LANDLINE_CORE = re.compile(r"[234]\d{9}$")
SPECIAL = re.compile(r"0?8\d{9}")
CALL_CENTER = re.compile(r"444\d{4}")
MULTI = re.compile(r"(?:90|0)5\d{9}|0[234]\d{9}|08\d{9}")
NON_PHONE_CTX = re.compile(r"(yetki|belge|vergi|ilan\s*no|ilan\s*numara|portfoy|ref)[^0-9]*$")
PRICE_AFTER = re.compile(r"\s*(tl|try|₺|\$|€|eur|usd|m\s*2)\b")
THOUSANDS = re.compile(r"\d{1,3}(?:\s*[.,]\s*\d{3})+")
ENCLOSED = re.compile(r"[\u2460-\u24ff\u2776-\u2793]")
HINT_STEMS = ("aray", "numara", "telefon", "iletisim", "whatsap", "watsap", "wp", "tel", "gsm")


def classify(d: str) -> str | None:
    """Rakam dizisini standart numaraya çevirir (cep/sabit: 0XXXXXXXXXX, 444: 444XXXX); telefon değilse None."""
    if 14 <= len(d) <= 24 and len(ms := MULTI.findall(d)) >= 2:
        return ",".join("0" + m[-10:] for m in ms)
    if m := MOBILE_PREFIXED.search(d):
        return "0" + m.group()[-10:]
    if 10 <= len(d) <= 12 and MOBILE_CORE.search(d):
        return "0" + d[-10:]
    if (m := LANDLINE.search(d)) and len(d) <= 12:
        return m.group()
    if 10 <= len(d) <= 11 and LANDLINE_CORE.search(d):
        return "0" + d[-10:]
    if SPECIAL.fullmatch(d):
        return "0" + d[-10:]
    if CALL_CENTER.fullmatch(d):
        return d
    return None


def not_phone(norm: str, m: re.Match, d: str) -> bool:
    """0/90 ile başlamayan dizi fiyat ya da belge/vergi/ilan no gibi görünüyorsa telefon sayma."""
    if d.startswith(("0", "90")):
        return False
    return bool(NON_PHONE_CTX.search(norm[max(m.start() - 30, 0):m.start()])
                or THOUSANDS.fullmatch(m.group(0).strip()) or PRICE_AFTER.match(norm, m.end()))


def extract(text: str) -> tuple[list[str], bool]:
    """(bulunan numaralar, şüpheli mi). Şüpheli: numaraya benzeyen dizi/ipucu kelime var ama numara yok.
    text: satırları \n ile ayrılmış OCR metni; numara satır sınırını aşmaz."""
    norm = "\n".join(normalize(line) for line in text.split("\n"))
    found: list[str] = []
    near = False
    for m in RUN.finditer(norm):
        d = re.sub(r"\D", "", m.group(0))
        if not_phone(norm, m, d):
            continue
        if num := classify(d):
            found += num.split(",")
        elif len(d) >= 7:
            near = True

    if not found:
        for m in LOOSE_RUN.finditer(norm):
            d = re.sub(r"\D", "", m.group(0))
            if 10 <= len(d) <= 13 and not not_phone(norm, m, d) and (num := classify(d)):
                found += num.split(",")

    hint = any(w.startswith(HINT_STEMS) for w in re.findall(r"[a-z]+", norm)) or ENCLOSED.search(text)
    return list(dict.fromkeys(found)), bool(near or hint) and not found


def digitish(text: str) -> bool:
    """Kutu yeniden okunmaya değer mi: rakam (ya da rakama benzeyen daire içi vb.) içeriyor."""
    return sum(c.isdigit() for c in normalize(text)) >= 3


def fragment(text: str) -> bool:
    """Numaranın başı gibi görünen ama tek başına numara olmayan parça: '0212', '0532 12', '+90 5'."""
    d = re.sub(r"\D", "", normalize(text))
    return 3 <= len(d) <= 9 and d.startswith(("0", "5", "90")) and not extract(text)[0]
