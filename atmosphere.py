"""
atmosphere.py — Koefisien ekstingsi atmosfer (pita V, 550 nm) dari data meteorologi harian.

Masukan: CSV harian dengan kolom minimal
    date        (YYYY-MM-DD, tanggal lokal)
    RH2M        kelembapan relatif 2 m (%)                 — NASA POWER: RH2M ; ERA5: dari T2m & Td2m
    T2M         suhu 2 m (°C)                               — NASA POWER: T2M  ; ERA5: t2m − 273.15
opsional
    AOD_55      kedalaman optik aerosol 550 nm (tak berdimensi) — NASA POWER: AOD_55 (MERRA-2) ; CAMS EAC4: aod550
    CLOUD_AMT   tutupan awan (%)                            — NASA POWER: CLOUD_AMT ; ERA5: tcc × 100
    PS          tekanan permukaan (kPa, NASA POWER) atau hPa — dideteksi otomatis
    TQV / PW    uap air kolom total (kg m⁻² ≡ mm)          — NASA POWER: TQV ; ERA5: tcwv

Model ekstingsi (mag per airmass), mengikuti Schaefer (1993, Vistas Astron. 36, 311) sebagaimana
diimplementasikan dalam VISLIMIT.BAS (Schaefer 1998, Sky & Telescope), dengan dua pengembangan:
  (a) suku aerosol dapat diganti langsung dengan AOD reanalisis: k_a = 1.086 × AOD_55  (2.5 log10 e = 1.086);
  (b) suku uap air memakai RH & T harian (bukan iklim rata-rata).
Rumus asli Schaefer dipertahankan sebagai fallback bila AOD tidak tersedia.
Kolom-kolom ini diverifikasi langsung dari berkas VISLIMIT.BAS (baris 1150–1200).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

LAMBDA_V_UM = 0.55
MAG_PER_TAU = 2.5 * np.log10(np.e)  # 1.0857


@dataclass
class Extinction:
    k_rayleigh: float
    k_aerosol: float
    k_ozone: float
    k_water: float
    source_aerosol: str  # 'AOD' atau 'Schaefer-RH'

    @property
    def k_total(self) -> float:
        return self.k_rayleigh + self.k_aerosol + self.k_ozone + self.k_water


def rayleigh_k(elev_m: float, lam_um: float = LAMBDA_V_UM) -> float:
    """VISLIMIT 1150: KR = 0.1066 exp(−h/8200) (λ/0.55)^−4."""
    return 0.1066 * np.exp(-elev_m / 8200.0) * (lam_um / 0.55) ** -4


def aerosol_k_schaefer(rh_pct: float, elev_m: float, lat_deg: float, day_of_year: int,
                       lam_um: float = LAMBDA_V_UM) -> float:
    """VISLIMIT 1160–1170: aerosol iklim + faktor kelembapan + variasi musiman.
    KA = 0.1 (λ/0.55)^−1.3 exp(−h/1500) × (1 − 0.32/ln(RH/100))^1.33 × (1 + 0.33 sgn(lat) sin(RA)),
    RA = fase tahun (radian) dengan RA=0 pada ~ 21 Maret (vernal equinox) dalam konvensi Schaefer."""
    rh = np.clip(rh_pct, 1.0, 99.9)
    ra = 2 * np.pi * (day_of_year - 80) / 365.25
    sl = np.sign(lat_deg) if lat_deg != 0 else 1.0
    k = 0.1 * (lam_um / 0.55) ** -1.3 * np.exp(-elev_m / 1500.0)
    k *= (1.0 - 0.32 / np.log(rh / 100.0)) ** 1.33
    k *= (1.0 + 0.33 * sl * np.sin(ra))
    return float(k)


def aerosol_k_from_aod(aod_550: float, lam_um: float = LAMBDA_V_UM, angstrom: float = 1.3) -> float:
    """Konversi AOD reanalisis → mag/airmass: k_a = 1.0857 × τ(λ), τ(λ) = τ550 (λ/0.55)^−α."""
    return float(MAG_PER_TAU * aod_550 * (lam_um / 0.55) ** -angstrom)


def ozone_k(lat_deg: float, day_of_year: int, oz_V: float = 0.016) -> float:
    """VISLIMIT 1180: KO = OZ × (3 + 0.4 (lat cos RA − cos 3 lat))/3 ; OZ(V)=0.016. lat dalam radian."""
    lt = np.radians(lat_deg)
    ra = 2 * np.pi * (day_of_year - 80) / 365.25
    return float(oz_V * (3.0 + 0.4 * (lt * np.cos(ra) - np.cos(3 * lt))) / 3.0)


def water_k(rh_pct: float, temp_C: float, elev_m: float, wt_V: float = 0.031) -> float:
    """VISLIMIT 1190: KW = WT × 0.94 (RH/100) exp(T/15) exp(−h/8200) ; WT(V)=0.031 (Schaefer 1993, Tabel 5)."""
    return float(wt_V * 0.94 * (rh_pct / 100.0) * np.exp(temp_C / 15.0) * np.exp(-elev_m / 8200.0))


def extinction_from_met(rh_pct: float, temp_C: float, elev_m: float, lat_deg: float, day_of_year: int,
                        aod_550: float | None = None) -> Extinction:
    kr = rayleigh_k(elev_m)
    ko = ozone_k(lat_deg, day_of_year)
    kw = water_k(rh_pct, temp_C, elev_m)
    if aod_550 is not None and np.isfinite(aod_550):
        ka, src = aerosol_k_from_aod(aod_550), "AOD"
    else:
        ka, src = aerosol_k_schaefer(rh_pct, elev_m, lat_deg, day_of_year), "Schaefer-RH"
    return Extinction(kr, ka, ko, kw, src)


def airmass_rozenberg(alt_deg: float) -> float:
    """Airmass Rozenberg (1966), sah sampai horizon: X = 1/(cos z + 0.025 e^{−11 cos z})."""
    cz = np.cos(np.radians(90.0 - alt_deg))
    cz = max(cz, 0.0)
    return float(1.0 / (cz + 0.025 * np.exp(-11.0 * cz)))


# ---------------------------------------------------------------- Pembaca CSV
_COLMAP = {
    # kanonik : kandidat nama kolom (huruf kecil)
    "RH2M": ["rh2m", "rh", "relative_humidity", "r2"],
    "T2M": ["t2m", "t", "temperature", "temp_c"],
    "AOD_55": ["aod_55", "aod550", "aod", "aod_550"],
    "CLOUD_AMT": ["cloud_amt", "tcc", "cloud", "cloud_cover"],
    "PS": ["ps", "sp", "pressure", "surface_pressure"],
    "TQV": ["tqv", "pw", "tcwv", "precipitable_water"],
    "PRECTOTCORR": ["prectotcorr", "precip", "tp", "rain_mm"],
}


def read_met_csv(path: str) -> pd.DataFrame:
    """Baca CSV NASA POWER (lewati header '-END HEADER-') atau CSV generik; kembalikan indeks tanggal lokal."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    start = 0
    for i, ln in enumerate(lines):
        if "-END HEADER-" in ln:
            start = i + 1
            break
    df = pd.read_csv(path, skiprows=start)
    cols = {c.lower(): c for c in df.columns}

    # tanggal: NASA POWER daily → YEAR, MO, DY ; generik → date
    if {"year", "mo", "dy"} <= set(cols):
        df["date"] = pd.to_datetime(dict(year=df[cols["year"]], month=df[cols["mo"]], day=df[cols["dy"]]))
    elif "year" in cols and "doy" in cols:
        df["date"] = pd.to_datetime(df[cols["year"]].astype(str), format="%Y") + pd.to_timedelta(df[cols["doy"]] - 1, unit="D")
    elif "date" in cols:
        df["date"] = pd.to_datetime(df[cols["date"]])
    else:
        raise ValueError("CSV tidak punya kolom tanggal (date, atau YEAR/MO/DY).")

    out = pd.DataFrame({"date": df["date"].dt.normalize()})
    for canon, cands in _COLMAP.items():
        for c in cands:
            if c in cols:
                out[canon] = pd.to_numeric(df[cols[c]], errors="coerce")
                break
    # ERA5: t2m dalam K, tcc 0–1, sp dalam Pa → normalisasi
    if "T2M" in out and out["T2M"].median() > 200:
        out["T2M"] -= 273.15
    if "CLOUD_AMT" in out and out["CLOUD_AMT"].max() <= 1.0:
        out["CLOUD_AMT"] *= 100.0
    if "PS" in out:
        med = out["PS"].median()
        if med > 5000:        # Pa
            out["PS"] /= 100.0
        elif med < 200:       # kPa (NASA POWER)
            out["PS"] *= 10.0
    # nilai sentinel NASA POWER
    out = out.replace(-999.0, np.nan)
    return out.set_index("date").sort_index()


_MET_KEYS = ("RH2M", "T2M", "AOD_55", "CLOUD_AMT", "PS", "TQV", "PRECTOTCORR")


def monthly_climatology(met: pd.DataFrame) -> pd.DataFrame:
    """Rata-rata bulanan tiap variabel dari seluruh CSV (untuk tanggal di luar cakupan data, mis. 2027–2030)."""
    cols = [c for c in _MET_KEYS if c in met.columns]
    return met.groupby(met.index.month)[cols].mean()


def met_for_date(met: pd.DataFrame | None, date, defaults: dict, clim: pd.DataFrame | None = None) -> dict:
    """Ambil baris meteorologi untuk satu tanggal. Prioritas: nilai harian CSV → klimatologi bulanan CSV → default.
    met_source: 'csv' (semua variabel inti harian), 'csv+clim' (sebagian diisi klimatologi), 'climatology', 'default'."""
    row = dict(defaults)
    row["met_source"] = "default"
    if met is None:
        return row
    d = pd.Timestamp(date).normalize()
    filled_daily, filled_clim = set(), set()
    if d in met.index:
        r = met.loc[d]
        for k in _MET_KEYS:
            if k in r and np.isfinite(r[k]):
                row[k] = float(r[k]); filled_daily.add(k)
    if clim is None:
        clim = monthly_climatology(met)
    for k in _MET_KEYS:
        if k not in filled_daily and k in clim.columns and np.isfinite(clim.loc[d.month, k]):
            row[k] = float(clim.loc[d.month, k]); filled_clim.add(k)
    core = {"RH2M", "T2M", "AOD_55"}
    if core <= filled_daily:
        row["met_source"] = "csv"
    elif filled_daily & core:
        row["met_source"] = "csv+clim"
    elif filled_clim:
        row["met_source"] = "climatology"
    return row


# Default klimatologis pesisir Sulawesi Barat (GANTI dengan rata-rata POWER 2000–2030 setelah data diunduh)
MET_DEFAULTS = dict(RH2M=80.0, T2M=27.5, AOD_55=np.nan, CLOUD_AMT=60.0, PS=1009.0)


if __name__ == "__main__":
    for rh in (60, 75, 90):
        e = extinction_from_met(rh, 28.0, 10.0, -3.54, 200)
        print(f"RH={rh}%  kR={e.k_rayleigh:.3f} kA={e.k_aerosol:.3f} ({e.source_aerosol}) kO={e.k_ozone:.3f} kW={e.k_water:.3f}  k={e.k_total:.3f}")
    e = extinction_from_met(80, 28.0, 10.0, -3.54, 200, aod_550=0.25)
    print(f"AOD=0.25 → kA={e.k_aerosol:.3f}  k={e.k_total:.3f}")
    print("X(alt=3°) =", round(airmass_rozenberg(3.0), 2), " X(alt=6°) =", round(airmass_rozenberg(6.0), 2))
