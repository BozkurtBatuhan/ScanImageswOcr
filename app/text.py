"""Türkçe metin normalizasyonu: OCR çıktısındaki harf/rakam karışıklıklarını ve
yazıyla yazılmış sayıları ('sıfır beş yüz otuz iki') rakama çevirir."""
from __future__ import annotations

import re
import unicodedata

_FOLD = str.maketrans("ıİğĞüÜşŞöÖçÇ", "iigguussoocc")
_CIRCLED_TEN = str.maketrans(dict.fromkeys("⑩⓾❿➉➓", "0"))

UNITS = {"bir": 1, "iki": 2, "uc": 3, "dort": 4, "bes": 5,
         "alti": 6, "yedi": 7, "sekiz": 8, "dokuz": 9}
TENS = {"on": 10, "yirmi": 20, "otuz": 30, "kirk": 40, "elli": 50,
        "altmis": 60, "yetmis": 70, "seksen": 80, "doksan": 90}

_LOOK = str.maketrans({"O": "0", "o": "0", "D": "0", "I": "1", "l": "1", "|": "1", "Z": "2", "z": "2",
                       "S": "5", "s": "5", "G": "6", "b": "6", "T": "7", "B": "8", "g": "9", "q": "9"})
_LC = "0-9OoDIl|ZzSsGbTBgq"
_LOOK_RUN = re.compile(rf"(?<![A-Za-z0-9])[{_LC}][{_LC}\s.\-_()/*+,·•]{{7,}}[{_LC}](?![A-Za-z])")
_NUM_WORDS = sorted({"sifir", "yuz", *UNITS, *TENS}, key=len, reverse=True)


def fold(s: str) -> str:
    return s.translate(_FOLD).lower()


def unify_digits(text: str) -> str:
    """Daire içi/tam genişlik/dingbat rakamları (⑤, ０, ❺, ➄) ASCII rakama çevirir."""
    text = unicodedata.normalize("NFKC", text.translate(_CIRCLED_TEN))
    return "".join(str(unicodedata.digit(c)) if not c.isascii() and unicodedata.digit(c, None) is not None
                   else c for c in text)


def fix_lookalikes(text: str) -> str:
    """Rakam görünümlü koşularda O->0, l->1, S->5 gibi OCR hatalarını düzeltir."""
    def repl(m: re.Match) -> str:
        run = m.group(0)
        if sum(c.isdigit() for c in run) < 5:
            return run
        return run.translate(_LOOK)
    return _LOOK_RUN.sub(repl, text)


def split_glued(tok: str) -> list[str] | None:
    """OCR'ın bitişik okuduğu sayı kelimelerini ayırır: 'yuzotuz' -> ['yuz', 'otuz']. Tamamı ayrılmazsa None."""
    if not tok:
        return []
    for w in _NUM_WORDS:
        if tok.startswith(w) and (rest := split_glued(tok[len(w):])) is not None:
            return [w, *rest]
    return None


def words_to_digits(s: str) -> str:
    """'sıfır beş yüz otuz iki' -> '0 532'. 'beş üç iki' -> '5 3 2'. (s zaten fold edilmiş olmalı)"""
    out: list[str] = []
    h = t = 0
    u: int | None = None

    def flush() -> None:
        nonlocal h, t, u
        if h or t or u is not None:
            out.append(str(h + t + (u or 0)))
        h = t = 0
        u = None

    toks: list[str] = []
    for tok in re.findall(r"[a-z]+|\d+|\S", s):
        parts = split_glued(tok) if tok.isalpha() and len(tok) > 5 else None
        toks += parts or [tok]
    for tok in toks:
        if tok == "sifir":
            flush(); out.append("0")
        elif tok in UNITS:
            if u is not None:
                flush()
            u = UNITS[tok]
        elif tok == "yuz":
            if u is not None and not h and not t:
                h, u = u * 100, None
            elif not (h or t or u is not None):
                h = 100
            else:
                flush(); h = 100
        elif tok in TENS:
            if t or u is not None:
                flush()
            t = TENS[tok]
        else:
            flush(); out.append(tok)
    flush()
    return " ".join(out)


def normalize(text: str) -> str:
    return words_to_digits(fold(fix_lookalikes(unify_digits(text))))
