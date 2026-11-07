"""
contrast.py — Kerangka model kontras hilal terhadap langit senja (pita V).

Rantai perhitungan (semua luminansi dalam cd m⁻², iluminansi dalam lux):
  1. Kecerahan langit senja di posisi Bulan  B_sky(h_sun, z_moon, ρ, k, X)      — Schaefer 1993/1998 (VISLIMIT 2280–2290)
  2. Luminansi permukaan sabit               L_cres = E_moon / Ω_cres            — magnitudo Bulan fungsi sudut fase (Krisciunas & Schaefer 1991)
  3. Pelemahan oleh ekstingsi                 × 10^(−0.4 k X)                     — k dari atmosphere.py (RH, T, AOD reanalisis)
  4. Kontras                                  C = L_cres,obs / B_sky              — definisi Hoffman (2003)
  5. Ambang mata                              (a) Hecht/Blackwell via Schaefer: iluminansi ambang titik TH(B_sky)
                                              (b) Crumey (2014): R(B) rezim Ricco (konstanta diisi dari makalah) — HOOK
     Sabit setebal W' ≪ resolusi mata diperlakukan sebagai sumber-garis: iluminansi per elemen resolusi
     E_elem = L_cres,obs × W' × θ_res, dibandingkan dengan F × TH.  (pendekatan Schaefer 1988)
  6. Indeks visibilitas fisis  V_phys = log10( E_elem / (F × TH) )  → ≥ 0 "terlihat secara fisis"
  7. Kriteria pembanding: MABIMS (3°, 6,4°), Yallop q (1997), Odeh V (2006)

CATATAN VERIFIKASI (hal yang harus dicek terhadap makalah asli sebelum dipakai di naskah):
  [V1] Hukum fase Bulan K&S 1991 pada α > 150° adalah ekstrapolasi; bandingkan dengan Schaefer 1991 dan Sultan 2007
       (luminansi sabit terukur). Parameter `crescent_sb_factor` disediakan untuk kalibrasi.
  [V2] Konstanta Crumey (2014) eq. 23–25, 28 di bawah disalin dari ringkasan daring; VERIFIKASI dari PDF makalah,
       termasuk satuan R (cd m⁻² × satuan luas sudut) sebelum mengaktifkan `threshold='crumey'`.
  [V3] Rumus senja Schaefer berlaku kira-kira untuk h_sun antara −4° dan −18°; untuk h_sun > −4° (langit masih
       sangat terang) model ini meremehkan kecerahan — tandai hasil sebagai 'di luar rentang'.
  [V4] Efek Danjon (pemendekan sabit) belum dimodelkan; pengaruhnya pada panjang sabit, bukan kontras per elemen.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

# ---------------------------------------------------------------- Konstanta (VISLIMIT.BAS pita V = indeks 3)
MO_V = -11.05           # titik nol fotometrik pita V (Schaefer)
MS_V = -26.74           # magnitudo Matahari pita V
BO_V = 1.0e-13          # kecerahan langit malam dasar (satuan internal VISLIMIT)
FC_TO_LUX = 10.7639
INTERNAL_TO_NL = 1.0 / 1.11e-15   # satuan internal VISLIMIT → nanolambert
NL_TO_CDM2 = 3.183e-6             # 1 nL = 10⁻⁹ lambert = 10⁻⁹ × 10⁴/π cd m⁻²
LUX_MAG0 = 2.54e-6                # iluminansi bintang V = 0 (lux)
ARCMIN = np.pi / (180.0 * 60.0)
SKY_SCALE = 1.0   # faktor skala kecerahan langit senja (uji sensitivitas; Patat dkk. 2006 di Paranal ≈ 0,2–0,3 × Schaefer pada h_sun = −5°)


# ---------------------------------------------------------------- 1. Kecerahan langit senja
SKY_MODEL = "sultan07"   # "schaefer98" (VISLIMIT) atau "sultan07" (log-linear dari luminansi senja terukur Sultan 2007, geometri hilal)


def _sky_schaefer98_nl(h_sun_deg, z_moon_deg, rho_deg, k, X_moon):
    """VISLIMIT 2280–2290 (Schaefer 1998), nL."""
    rho = max(rho_deg, 0.5)
    bt = 10 ** (-0.4 * (MS_V - MO_V + 32.5 - h_sun_deg - z_moon_deg / (360.0 * k)))
    bt *= (100.0 / rho) * (1.0 - 10 ** (-0.4 * k * X_moon))
    return bt * INTERNAL_TO_NL


def _sky_sultan07_nl(h_sun_deg, z_moon_deg, rho_deg, k, X_moon):
    """Luminansi langit senja di arah hilal, dijangkarkan pada pengukuran Sultan (2007, Obs. 127, 53) Tabel I:
    L_B pada tinggi Bulan 2°, DAZ = 0, Sana'a 2000 m, untuk h_sun = −2° … −6°:
        log10 L_B[nL] = 8.074 + 0.479 h_sun   (rms 0,018 dex; ekstrapolasi di luar −2…−6 ditandai [V3]).
    Ketergantungan sudut terhadap geometri Sultan (ρ_S = 2° − h_sun, z_S = 88°) diambil dari bentuk Schaefer 1998
    yang dinormalkan: faktor (ρ_S/ρ) dan 10^{0.4 (z − 88)/(360 k)}."""
    rho = max(rho_deg, 0.5)
    rho_S = 2.0 - h_sun_deg
    base = 10 ** (8.074 + 0.479 * h_sun_deg)
    ang = (rho_S / rho) * 10 ** (0.4 * (z_moon_deg - 88.0) / (360.0 * k))
    return base * ang


def sky_brightness_twilight_cdm2(h_sun_deg: float, z_moon_deg: float, rho_deg: float,
                                 k: float, X_moon: float, year: float = 2015.0) -> float:
    """Kecerahan langit di arah Bulan (cd m⁻²): senja (model terpilih SKY_MODEL) + langit malam dasar.
    h_sun < 0 (derajat), z_moon = jarak zenit Bulan (derajat), rho = jarak sudut Bulan–Matahari (derajat)."""
    if SKY_MODEL == "schaefer98":
        bt = _sky_schaefer98_nl(h_sun_deg, z_moon_deg, rho_deg, k, X_moon)
    else:
        bt = _sky_sultan07_nl(h_sun_deg, z_moon_deg, rho_deg, k, X_moon)
    sz = np.sin(np.radians(z_moon_deg))
    bn = BO_V * (1 + 0.3 * np.cos(2 * np.pi * (year - 1992) / 11.0))
    bn *= (0.4 + 0.6 / np.sqrt(1.0 - 0.96 * sz ** 2))
    bn *= 10 ** (-0.4 * k * X_moon)
    b_nl = bt * SKY_SCALE + bn * INTERNAL_TO_NL
    return float(b_nl * NL_TO_CDM2)


# ---------------------------------------------------------------- 2–3. Luminansi sabit
def moon_magnitude_ks91(phase_angle_deg: float, dist_km: float) -> float:
    """Krisciunas & Schaefer (1991, PASP 103, 1033) eq. 9: m = −12.73 + 0.026|α| + 4×10⁻⁹ α⁴ (α derajat),
    dikoreksi jarak Bulan terhadap jarak rata-rata 384 400 km. [V1]"""
    a = abs(phase_angle_deg)
    m = -12.73 + 0.026 * a + 4.0e-9 * a ** 4
    m += 5.0 * np.log10(dist_km / 384400.0)
    return float(m)


def crescent_luminance_cdm2(phase_angle_deg: float, illum_frac: float, sd_arcmin: float, dist_km: float,
                            k: float, X_moon: float, crescent_sb_factor: float = 1.0) -> tuple[float, float]:
    """Kembalikan (L_cres_di_luar_atmosfer, L_cres_teramati) dalam cd m⁻².
    L = E/Ω dengan Ω_cres = f_illum × π SD² (sr). crescent_sb_factor = faktor kalibrasi empiris [V1]."""
    m = moon_magnitude_ks91(phase_angle_deg, dist_km)
    E = LUX_MAG0 * 10 ** (-0.4 * m)                     # lux, di luar atmosfer
    omega = max(illum_frac, 1e-6) * np.pi * (sd_arcmin * ARCMIN) ** 2
    L0 = E / omega * crescent_sb_factor
    L_obs = L0 * 10 ** (-0.4 * k * X_moon)
    return float(L0), float(L_obs)


# ---------------------------------------------------------------- 5. Ambang mata
def threshold_point_lux_schaefer(B_sky_cdm2: float) -> float:
    """Iluminansi ambang sumber titik (lux) — Hecht (1947) dengan konstanta Schaefer (VISLIMIT 280–310).
    Dua rezim: gelap (B < 1500 nL) dan terang."""
    bl = B_sky_cdm2 / NL_TO_CDM2  # nL
    if bl < 1500.0:
        c1, c2 = 10 ** -9.8, 10 ** -1.9
    else:
        c1, c2 = 10 ** -8.35, 10 ** -5.9
    th_fc = c1 * (1.0 + np.sqrt(c2 * bl)) ** 2
    return float(th_fc * FC_TO_LUX)


# Model ambang kontras Crumey (2014, MNRAS 442, 2600–2619) — DIVERIFIKASI dari PDF makalah (4 Nov 2026).
# Satuan: B (luminansi latar) dalam cd m⁻²; A (luas sudut target) dalam steradian; R = C·A dalam sr;
# iluminansi ambang sumber titik ΔI = B·R(B) dalam lux (eq. 32–34, 53). Konstanta Blackwell 1946 (eq. 26–28, 37–40).
CRUMEY = dict(
    r1=6.505e-4, r2=-8.461e-4,                     # eq. 23/26 skotopik:  √R = r1 B^−1/4 + r2
    r3=1.772e-4, r4=7.167e-5,                      # eq. 24/27 fotopik:   √R = r3 B^−1/4 + r4 ; batas 7.08e-2 cd m⁻²
    a1=5.949e-8, a2=-2.389e-7, a3=2.459e-7, a4=4.120e-4, a5=-4.225e-4,   # eq. 25/28 bentuk hiperbola (seluruh rentang)
    k1=7.633e-3, k2=-7.174e-3,                     # eq. 35/37 C∞ skotopik = k1 B^−1/4 + k2
    k3=0.0, k4=2.720e-3,                           # eq. 36/38 C∞ fotopik (Weber)      ; batas 3.54e-1 cd m⁻²
    b1=9.606e-6, b2=-4.112e-5, b3=5.019e-5, b4=4.837e-3, b5=-4.884e-3,  # eq. 39/40 C∞ bentuk hiperbola
)


def crumey_R(B: float) -> float:
    """Konstanta Ricco R(B) [sr], eq. 25 dengan konstanta (28) — berlaku di seluruh rentang luminansi."""
    c = CRUMEY; b = max(B, 1e-7); q = b ** -0.25
    inner = max(c["a1"] * b ** -0.5 + c["a2"] * q + c["a3"], 0.0)
    return float((np.sqrt(inner) + c["a4"] * q + c["a5"]) ** 2)


def crumey_Cinf(B: float) -> float:
    """Ambang kontras target luas C∞(B), eq. 39 dengan konstanta (40)."""
    c = CRUMEY; b = max(B, 1e-7); q = b ** -0.25
    inner = max(c["b1"] * b ** -0.5 + c["b2"] * q + c["b3"], 0.0)
    return float(np.sqrt(inner) + c["b4"] * q + c["b5"])


def crumey_q(B: float) -> float:
    """Eksponen penggabung q(B), eq. 42–44."""
    if B >= 3.40:   return 1.146 - 0.0885 * np.log10(B)
    if B >= 0.193:  return 0.8861 + 0.4 * np.log10(B)
    return 0.6


def crumey_threshold_contrast(B: float, A_sr: float) -> float:
    """Ambang kontras C_th untuk target seluas A_sr (sr) di latar B (cd m⁻²): eq. 41, C = ((R/A)^q + C∞^q)^(1/q)."""
    q = crumey_q(B)
    return float((((crumey_R(B) / max(A_sr, 1e-12)) ** q) + crumey_Cinf(B) ** q) ** (1.0 / q))


def crumey_ricco_diameter_arcmin(B: float) -> float:
    """Diameter luas Ricco A_R = R/C∞ (eq. 22), dinyatakan sebagai diameter lingkaran setara (menit busur)."""
    A_R = crumey_R(B) / max(crumey_Cinf(B), 1e-9)
    return float(2.0 * np.sqrt(A_R / np.pi) / ARCMIN)


def threshold_point_lux_crumey(B_sky_cdm2: float) -> float:
    """Iluminansi ambang sumber titik Crumey: ΔI = B·R(B) [lux] (eq. 34)."""
    return float(B_sky_cdm2 * crumey_R(B_sky_cdm2))


def eye_resolution_arcmin(B_sky_cdm2: float) -> float:
    """Ukuran elemen resolusi mata (menit busur) sebagai fungsi luminansi adaptasi — pendekatan sederhana untuk
    jalur Hecht/Schaefer: 1' pada fotopik (>3 cd m⁻²), membesar ke ~3' pada mesopik rendah."""
    logb = np.log10(max(B_sky_cdm2, 1e-6))
    return float(np.interp(logb, [-3.0, -1.0, 0.5], [3.0, 1.6, 1.0]))


# ---------------------------------------------------------------- 6. Indeks visibilitas fisis
@dataclass
class ContrastResult:
    k_total: float
    X_moon: float
    B_sky_cdm2: float
    L_cres0_cdm2: float
    L_cres_obs_cdm2: float
    contrast: float            # C = L_obs / B_sky
    width_arcmin: float
    theta_res_arcmin: float
    E_elem_lux: float
    TH_schaefer_lux: float
    TH_crumey_lux: float
    field_factor: float
    V_phys_schaefer: float     # log10(E_elem / (F TH)) — jalur Hecht/Schaefer (sumber titik per elemen resolusi)
    V_phys_crumey: float       # log10(C / (F C_th)) — jalur Crumey 2014 (luas sabit vs luas Ricco), jalur UTAMA
    ricco_diam_arcmin: float   # diameter luas Ricco pada B_sky
    crescent_len_eff_arcmin: float  # panjang sabit yang dijumlahkan = min(panjang sabit, diameter Ricco)
    C_th_crumey: float         # ambang kontras Crumey untuk luas sabit efektif
    in_model_range: bool       # h_sun dalam rentang validitas rumus senja [V3]


def evaluate_contrast(h_sun_deg: float, moon_alt_deg: float, rho_deg: float, phase_angle_deg: float,
                      illum_frac: float, sd_arcmin: float, width_arcmin: float, dist_km: float,
                      k_total: float, X_moon: float, year: float = 2015.0,
                      field_factor: float = 1.0, crescent_sb_factor: float = 1.0,
                      summation_length_arcmin: float | None = None) -> ContrastResult:
    """summation_length_arcmin: panjang busur sabit yang dijumlahkan mata (penjumlahan spasial sepanjang
    sumber-garis). None → sama dengan elemen resolusi θ_res (asumsi paling konservatif, per Schaefer 1988);
    nilai 3–10' menggambarkan penjumlahan sepanjang busur dan menaikkan E_elem secara proporsional."""
    z_moon = 90.0 - moon_alt_deg
    B = sky_brightness_twilight_cdm2(h_sun_deg, z_moon, rho_deg, k_total, X_moon, year)
    L0, Lobs = crescent_luminance_cdm2(phase_angle_deg, illum_frac, sd_arcmin, dist_km, k_total, X_moon,
                                       crescent_sb_factor)
    C = Lobs / B if B > 0 else np.nan
    theta = eye_resolution_arcmin(B)
    w_eff = min(width_arcmin, theta)                      # sabit lebih tebal dari resolusi → luas penuh elemen
    length = theta if summation_length_arcmin is None else max(theta, summation_length_arcmin)
    E_elem = Lobs * (w_eff * ARCMIN) * (length * ARCMIN)  # lux
    th_s = threshold_point_lux_schaefer(B)
    th_c = threshold_point_lux_crumey(B)
    V_s = np.log10(E_elem / (field_factor * th_s)) if E_elem > 0 else -np.inf
    # --- Jalur Crumey (utama): sabit = sumber garis selebar W' dan sepanjang ℓ. Sistem visual menjumlahkan
    # cahaya di dalam luas Ricco; panjang yang efektif dijumlahkan = min(ℓ_sabit, diameter Ricco).
    # ℓ_sabit = π·SD' (busur 180°, batas atas geometris; efek Danjon memendekkannya — lihat [V4]).
    d_R = crumey_ricco_diameter_arcmin(B)
    len_cres = np.pi * sd_arcmin
    len_eff = min(len_cres, d_R)
    A_cres = (width_arcmin * ARCMIN) * (len_eff * ARCMIN)          # sr
    C_th = crumey_threshold_contrast(B, A_cres)
    V_c = np.log10(C / (field_factor * C_th)) if (C > 0 and np.isfinite(C_th)) else -np.inf
    return ContrastResult(k_total, X_moon, B, L0, Lobs, float(C), width_arcmin, theta, float(E_elem),
                          th_s, th_c, field_factor, float(V_s), float(V_c), d_R, float(len_eff), C_th,
                          in_model_range=bool(-18.0 <= h_sun_deg <= (-2.0 if SKY_MODEL == "sultan07" else -4.0)))


# ---------------------------------------------------------------- 7. Kriteria pembanding
def mabims(alt_deg: float, elong_deg: float, alt_min: float = 3.0, elong_min: float = 6.4) -> bool:
    """MABIMS 2021 (Neo-MABIMS): tinggi ≥ 3° dan elongasi ≥ 6,4° saat Matahari terbenam.
    Definisi operasional (toposentrik/geosentrik, dengan/tanpa refraksi) ditentukan pemanggil."""
    return bool(alt_deg >= alt_min and elong_deg >= elong_min)


def yallop_q(arcv_geo_deg: float, width_topo_arcmin: float) -> tuple[float, str]:
    """Yallop (1997) NAO TN 69: q = [ARCV − (11.8371 − 6.3226 W' + 0.7319 W'² − 0.1018 W'³)] / 10,
    dihitung pada 'best time' = sunset + 4/9 lag. ARCV geosentrik, W' toposentrik (menit busur)."""
    W = width_topo_arcmin
    q = (arcv_geo_deg - (11.8371 - 6.3226 * W + 0.7319 * W ** 2 - 0.1018 * W ** 3)) / 10.0
    if q > 0.216:      zone = "A: mudah terlihat mata telanjang"
    elif q > -0.014:   zone = "B: terlihat dalam kondisi sempurna"
    elif q > -0.160:   zone = "C: mungkin perlu alat optik untuk menemukan"
    elif q > -0.232:   zone = "D: perlu alat optik"
    elif q > -0.293:   zone = "E: tidak terlihat dengan teleskop"
    else:              zone = "F: di bawah batas Danjon"
    return float(q), zone


def odeh_V(arcv_topo_deg: float, width_topo_arcmin: float) -> tuple[float, str]:
    """Odeh (2006) Exp. Astron. 18, 39: V = ARCV − (−0.1018 W³ + 0.7319 W² − 6.3226 W + 7.1651),
    ARCV & W toposentrik, pada best time. Zona: V ≥ 5.65 mata telanjang; 2 ≤ V < 5.65 mungkin mata
    telanjang setelah ditemukan dengan optik; −0.96 ≤ V < 2 hanya optik; V < −0.96 tidak terlihat."""
    W = width_topo_arcmin
    V = arcv_topo_deg - (-0.1018 * W ** 3 + 0.7319 * W ** 2 - 6.3226 * W + 7.1651)
    if V >= 5.65:     zone = "A: mata telanjang"
    elif V >= 2.0:    zone = "B: optik lalu mata telanjang"
    elif V >= -0.96:  zone = "C: hanya optik"
    else:             zone = "D: tidak terlihat"
    return float(V), zone


if __name__ == "__main__":
    # Contoh: hilal tinggi 5°, Matahari −4°, elongasi 9°, k = 0.35
    from atmosphere import airmass_rozenberg
    for alt, hsun, el in [(3.0, -2.0, 6.4), (5.0, -4.0, 9.0), (8.0, -6.0, 12.0)]:
        X = airmass_rozenberg(alt)
        r = evaluate_contrast(hsun, alt, el, 180 - el, 0.5 * (1 - np.cos(np.radians(el))), 15.5,
                              15.5 * (1 - np.cos(np.radians(el))), 384400, 0.35, X)
        print(f"alt={alt} h_sun={hsun} elong={el}: B_sky={r.B_sky_cdm2:.2f} cd/m2  L_obs={r.L_cres_obs_cdm2:.1f}  "
              f"C={r.contrast:.2f}  V_phys={r.V_phys_schaefer:+.2f}  range={r.in_model_range}")
