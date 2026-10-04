# -*- coding: utf-8 -*-
"""Bon metric co ban, moi loai truong dung mot loai.

CHON SAI LOAI METRIC = SAI KET LUAN. Hai vi du that:

  '166.000' vs '160.000'   (SAI 6.000 dong)
      ANLS    = 0.857   <- THUONG DIEM CAO cho cau tra loi SAI TIEN
      numeric = 0       <- dung
  '285000' vs '128000'     (SAI hoan toan)
      ANLS    = 0.667   <- VAN tren nguong 0.5, van duoc diem
      numeric = 0       <- dung
  -> truong TIEN va SO phai dung numeric_match

  '12/03/2021 14:32' vs '2021-03-12 14:32'   (DOC DUNG, chi doi dinh dang)
      ANLS         = 0.500  <- MAT MOT NUA diem du tra loi DUNG
      norm_date_em = 1      <- dung
  -> truong NGAY/GIO phai dung norm_date_em

  (cac con so tren do bang chinh module nay, khong phai uoc luong)
"""
import re
import unicodedata

try:
    from rapidfuzz.distance import Levenshtein

    def _lev(a: str, b: str) -> int:
        return Levenshtein.distance(a, b)
except ImportError:                                   # fallback thuan Python
    def _lev(a: str, b: str) -> int:
        if a == b:
            return 0
        if not a:
            return len(b)
        if not b:
            return len(a)
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                               prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]


def nfc(s) -> str:
    """BAT BUOC truoc moi phep so sanh.

    Ky tu tieng Viet co dau co HAI cach ma hoa Unicode:
        NFC: 'ế' = 1 codepoint U+1EBF
        NFD: 'ế' = 'e' + dau mu + dau sac
    Hien thi GIONG HET nhau nhung Levenshtein tinh ra KHAC.
    Bo buoc nay = ANLS tut ma khong ai hieu vi sao.
    """
    return unicodedata.normalize("NFC", "" if s is None else str(s)).strip()


def anls(pred, gt, thr: float = 0.5) -> float:
    """Average Normalized Levenshtein Similarity (chuan DocVQA).
    Dung cho TEXT DAI: merchant, address, item name."""
    p, g = nfc(pred).lower(), nfc(gt).lower()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    sim = 1.0 - _lev(g, p) / max(len(g), len(p))
    return sim if sim >= thr else 0.0


def numeric_match(pred, gt) -> float:
    """So bang GIA TRI sau khi bo moi ky tu khong phai chu so.
    Dung cho TIEN va SO: total, quantity, unit_price, amount."""
    f = lambda s: re.sub(r"[^\d]", "", nfc(s))
    a, b = f(pred), f(gt)
    return float(bool(b) and a == b)


def norm_date_em(pred, gt) -> float:
    """So tap chu so (da sap xep) -> bat duoc 12/03/2021 == 2021-03-12.
    Dung cho NGAY va GIO."""
    f = lambda s: sorted(re.findall(r"\d+", nfc(s)))
    gg = f(gt)
    return float(bool(gg) and f(pred) == gg)


def exact(pred, gt) -> float:
    """So bang tuyet doi sau NFC + lower. Dung cho MA: invoice_no, tax_code."""
    return float(nfc(pred).lower() == nfc(gt).lower())
