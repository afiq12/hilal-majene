"""Uji regresi ringan untuk hilal_majene (jalankan: python -m pytest tests/ atau python tests/test_checks.py).

Setiap uji mereproduksi satu angka pembanding dari literatur yang dikutip naskah:
  T1  luminansi sabit K&S 1991 vs Sultan (2007) Tabel I (dalam 10 %)
  T2  ambang kontras Crumey (2014) vs pembacaan Blackwell oleh Sultan (38 vs 37)
  T3  langit senja jangkar Sultan pada h_sun = -5 deg (sekitar 1.5 cd m^-2)
  T4  skala Yallop q: batas zona A/B dan contoh di TN 69
  T5  ekstingsi: komponen Rayleigh + ozon + uap air pada RH 80 %, T 28 C, paras laut
  T6  geometri efemeris: ijtima' pertama 2026 (18 Jan 2026 19:52 UTC menurut almanak)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import contrast as ct                      # noqa: E402
import atmosphere as at                    # noqa: E402

ARCMIN = np.radians(1 / 60)


def test_T1_crescent_luminance_vs_sultan2007():
    # Sultan 2007 Tabel I: alpha = 171.5 deg, L_cres ~ 2.1e8 nL di luar atmosfer
    elong = 180 - 171.5
    f = 0.5 * (1 - np.cos(np.radians(elong)))
    L0, _ = ct.crescent_luminance_cdm2(171.5, f, 15.5, 384400, 0.0, 1.0)
    L0_nl = L0 / ct.NL_TO_CDM2
    assert abs(L0_nl / 2.1e8 - 1) < 0.12, L0_nl


def test_T2_crumey_threshold_vs_blackwell_reading():
    # Sultan 2007: sabit lebar 0.16', latar 0.86 cd m^-2, C_th (Blackwell) = 37
    # Sultan membaca kurva Blackwell untuk target berdiameter 0.16'; Crumey eq. 41 memberi 38.0
    B = 0.86
    area = np.pi * (0.08 * ARCMIN) ** 2
    Cth = ct.crumey_threshold_contrast(B, area)
    assert 35 < Cth < 41, Cth


def test_T3_twilight_sky_sultan_anchor():
    ct.SKY_MODEL = "sultan07"
    B = ct.sky_brightness_twilight_cdm2(-5.0, 88.0, 7.0, 0.5, 15.0)
    assert 1.2 < B < 1.9, B          # 10^(8.074 - 2.395) nL = 4.8e5 nL = 1.5 cd m^-2


def test_T4_yallop_q_scale():
    # Lebar 0.3': ARCV pada q = 0 adalah 11.8371 - 6.3226*0.3 + 0.7319*0.09 - 0.1018*0.027
    W = 0.3
    arcv0 = 11.8371 - 6.3226 * W + 0.7319 * W ** 2 - 0.1018 * W ** 3
    q, zone = ct.yallop_q(arcv0, W)
    assert abs(q) < 1e-9 and zone.startswith("B")
    q2, zone2 = ct.yallop_q(arcv0 + 2.2, W)
    assert zone2.startswith("A"), (q2, zone2)


def test_T5_extinction_components():
    e = at.extinction_from_met(80.0, 28.0, 10.0, -3.54, 200, aod_550=0.25)
    assert 0.10 < e.k_rayleigh < 0.12, e.k_rayleigh     # Schaefer 1993: 0.1066 di paras laut (550 nm)
    assert 0.010 < e.k_ozone < 0.025, e.k_ozone
    assert 0.08 < e.k_water < 0.25, e.k_water
    assert 0.40 < e.k_total < 0.65, e.k_total


def test_T6_first_conjunction_2026():
    import ephemeris as ep
    ts, eph, _ = ep.load_ephemeris()
    t = ep.new_moons(ts, eph, 2026, 2026)[0]
    # Almanak: Bulan baru 18 Jan 2026 19:52 UTC
    assert t.utc_strftime("%Y-%m-%d") == "2026-01-18", t.utc_iso()
    hh = int(t.utc_strftime("%H"))
    assert 18 <= hh <= 21, t.utc_iso()


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print("PASS", name)
            except AssertionError as exc:
                fails += 1
                print("FAIL", name, exc)
    sys.exit(1 if fails else 0)
