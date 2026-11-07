"""
ephemeris.py — Posisi Bulan–Matahari untuk studi visibilitas hilal di Majene.

Menggunakan Skyfield + ephemeris JPL. Urutan pencarian berkas ephemeris:
  1. data/de440s.bsp  (unduh sendiri dari https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440s.bsp, ~32 MB)
  2. de421.bsp dari paket `skyfield-data` (1900–2050; selisih posisi Bulan terhadap DE440 < 0,01", tidak berpengaruh pada studi ini)

Semua sudut dalam derajat. Waktu internal UTC; kolom lokal memakai WITA (UTC+8).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from datetime import timedelta, timezone

import numpy as np
from skyfield import almanac
from skyfield.api import Loader, Topos, wgs84
from skyfield.framelib import ecliptic_frame

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
WITA = timezone(timedelta(hours=8))

# ---------------------------------------------------------------- Situs
@dataclass
class Site:
    name: str = "Majene (Pantai Pasir Putih)"
    lat_deg: float = -3.54      # LS negatif
    lon_deg: float = 118.97     # BT positif
    elev_m: float = 10.0        # elevasi pengamat di atas muka laut
    # Kondisi meteorologi standar untuk refraksi (dapat ditimpa per tanggal dari CSV)
    temperature_C: float = 28.0
    pressure_mbar: float = 1010.0


MAJENE = Site()


# ---------------------------------------------------------------- Pemuat ephemeris
def load_ephemeris():
    """Kembalikan (ts, eph, nama_berkas). Prioritas DE440s lokal, lalu DE421 bawaan skyfield-data."""
    load = Loader(DATA_DIR, verbose=False)
    ts = load.timescale(builtin=True)
    de440 = os.path.join(DATA_DIR, "de440s.bsp")
    if os.path.exists(de440) and os.path.getsize(de440) > 1_000_000:
        return ts, load("de440s.bsp"), "DE440s"
    try:
        from skyfield_data import get_skyfield_data_path
        p = os.path.join(get_skyfield_data_path(), "de421.bsp")
        from skyfield.api import load_file
        return ts, load_file(p), "DE421 (skyfield-data)"
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Tidak ada ephemeris. Letakkan de440s.bsp di folder data/ atau `pip install skyfield-data`."
        ) from exc


# ---------------------------------------------------------------- Geometri pada satu waktu
@dataclass
class Geometry:
    """Besaran geometri hilal pada satu instan (semua derajat kecuali disebut lain)."""
    t_utc: str
    t_wita: str
    sun_alt_topo: float          # tinggi pusat Matahari, toposentrik, tanpa refraksi
    sun_az: float
    moon_alt_topo: float         # tinggi pusat Bulan, toposentrik, tanpa refraksi
    moon_alt_topo_refr: float    # tinggi tampak (mar'i) pusat Bulan, dengan refraksi
    moon_alt_geo: float          # tinggi geosentrik (dari pusat Bumi) — seperti dipakai Yallop utk ARCV
    moon_az: float
    daz: float                   # beda azimut Bulan − Matahari (DAZ)
    arcv_topo: float             # beda tinggi Bulan − Matahari toposentrik (ARCV)
    arcv_geo: float              # ARCV geosentrik
    elong_geo: float             # elongasi geosentrik (ARCL geosentrik)
    elong_topo: float            # elongasi toposentrik
    phase_angle_topo: float      # sudut fase Matahari–Bulan–pengamat (≈180° − elongasi)
    illum_frac: float            # fraksi piringan tersinari (toposentrik)
    moon_sd_arcmin: float        # semidiameter Bulan toposentrik (menit busur)
    width_arcmin: float          # lebar sabit toposentrik W' = SD'(1 − cos ARCL') (Yallop/Odeh)
    moon_dist_km: float
    moon_age_h: float            # umur Bulan sejak ijtima' (jam)
    lag_min: float               # selisih moonset − sunset (menit), diisi oleh pemanggil


def _refracted_alt(alt_deg: float, temperature_C: float, pressure_mbar: float) -> float:
    """Refraksi Bennett (seperti dipakai Skyfield) untuk tinggi sejati → tinggi tampak."""
    from skyfield.earthlib import refraction
    r = refraction(alt_deg, temperature_C=temperature_C, pressure_mbar=pressure_mbar)
    return alt_deg + r


def geometry_at(t, ts, eph, site: Site, t_conj=None, lag_min: float = np.nan) -> Geometry:
    earth, sun, moon = eph["earth"], eph["sun"], eph["moon"]
    topos = wgs84.latlon(site.lat_deg, site.lon_deg, elevation_m=site.elev_m)
    obs = earth + topos

    a_sun = obs.at(t).observe(sun).apparent()
    a_moon = obs.at(t).observe(moon).apparent()
    alt_s, az_s, _ = a_sun.altaz()
    alt_m, az_m, dist_m = a_moon.altaz()

    # geosentrik
    g_sun = earth.at(t).observe(sun).apparent()
    g_moon = earth.at(t).observe(moon).apparent()
    elong_geo = g_moon.separation_from(g_sun).degrees
    # tinggi geosentrik: proyeksikan vektor geosentrik ke horizon lokal pengamat
    # (cara standar Yallop: ARCV geosentrik = beda tinggi tanpa paralaks)
    alt_m_geo = _geocentric_alt(g_moon, topos, t)
    alt_s_geo = _geocentric_alt(g_sun, topos, t)

    elong_topo = a_moon.separation_from(a_sun).degrees
    # sudut fase toposentrik: sudut di Bulan antara arah ke Matahari dan ke pengamat
    moon_to_sun = (eph["moon"].at(t).observe(sun).apparent()).position.km
    moon_to_obs = -a_moon.position.km
    cosang = np.dot(moon_to_sun, moon_to_obs) / (np.linalg.norm(moon_to_sun) * np.linalg.norm(moon_to_obs))
    phase_angle = float(np.degrees(np.arccos(np.clip(cosang, -1, 1))))
    illum = 0.5 * (1.0 + np.cos(np.radians(phase_angle)))

    R_MOON_KM = 1737.4
    sd_arcmin = float(np.degrees(np.arcsin(R_MOON_KM / dist_m.km)) * 60.0)
    width = sd_arcmin * (1.0 - np.cos(np.radians(elong_topo)))

    daz = _wrap180(az_m.degrees - az_s.degrees)
    age_h = float((t.tt - t_conj.tt) * 24.0) if t_conj is not None else np.nan

    dt = t.utc_datetime()
    return Geometry(
        t_utc=dt.strftime("%Y-%m-%d %H:%M:%S"),
        t_wita=dt.astimezone(WITA).strftime("%Y-%m-%d %H:%M:%S"),
        sun_alt_topo=float(alt_s.degrees), sun_az=float(az_s.degrees),
        moon_alt_topo=float(alt_m.degrees),
        moon_alt_topo_refr=_refracted_alt(float(alt_m.degrees), site.temperature_C, site.pressure_mbar),
        moon_alt_geo=alt_m_geo, moon_az=float(az_m.degrees),
        daz=daz,
        arcv_topo=float(alt_m.degrees - alt_s.degrees),
        arcv_geo=alt_m_geo - alt_s_geo,
        elong_geo=float(elong_geo), elong_topo=float(elong_topo),
        phase_angle_topo=phase_angle, illum_frac=float(illum),
        moon_sd_arcmin=sd_arcmin, width_arcmin=float(width),
        moon_dist_km=float(dist_m.km), moon_age_h=age_h, lag_min=lag_min,
    )


def _geocentric_alt(g_apparent, topos, t) -> float:
    """Tinggi 'geosentrik' = arah dari pusat Bumi, dinyatakan dalam horizon lokal pengamat."""
    # vektor arah geosentrik (ICRF) diputar ke kerangka horizon (alt-az) pengamat
    R = topos.rotation_at(t)
    x, y, z = R @ g_apparent.position.km
    return float(np.degrees(np.arctan2(z, np.hypot(x, y))))


def _wrap180(x: float) -> float:
    return float((x + 180.0) % 360.0 - 180.0)


# ---------------------------------------------------------------- Peristiwa: ijtima', sunset, moonset
def new_moons(ts, eph, year_start: int, year_end: int):
    """Semua ijtima' (konjungsi geosentrik, bujur ekliptika) dari 1 Jan year_start s.d. 31 Des year_end."""
    t0 = ts.utc(year_start, 1, 1)
    t1 = ts.utc(year_end + 1, 1, 2)
    t, y = almanac.find_discrete(t0, t1, almanac.moon_phases(eph))
    return t[y == 0]


def sunset_after(t_from, ts, eph, site: Site, body="sun"):
    """Terbenam pertama (tepi atas piringan menyentuh horizon, refraksi standar) setelah t_from."""
    topos = wgs84.latlon(site.lat_deg, site.lon_deg, elevation_m=site.elev_m)
    t1 = ts.tt_jd(t_from.tt + 2.0)
    if body == "sun":
        f = almanac.sunrise_sunset(eph, topos)
        t, y = almanac.find_discrete(t_from, t1, f)
        sets = t[y == 0]
    else:
        f = almanac.risings_and_settings(eph, eph["moon"], topos)
        t, y = almanac.find_discrete(t_from, t1, f)
        sets = t[y == 0]
    return sets[0] if len(sets) else None


def first_evening_after_conjunction(t_conj, ts, eph, site: Site):
    """Sunset pertama setelah ijtima' (malam ke-29 menurut praktik rukyat), beserta moonset & lag."""
    t_sunset = sunset_after(t_conj, ts, eph, site, "sun")
    # Cari semua moonset dalam jendela [sunset − 4 jam, sunset + 3 jam]; ambil yang terdekat ke sunset
    # (lag negatif bila Bulan terbenam lebih dulu). Menghindari salah tangkap moonset hari berikutnya.
    topos = wgs84.latlon(site.lat_deg, site.lon_deg, elevation_m=site.elev_m)
    f = almanac.risings_and_settings(eph, eph["moon"], topos)
    t0 = ts.tt_jd(t_sunset.tt - 4.0 / 24.0)
    t1 = ts.tt_jd(t_sunset.tt + 3.0 / 24.0)
    t, y = almanac.find_discrete(t0, t1, f)
    sets = t[y == 0]
    if len(sets) == 0:
        t_moonset, lag = None, np.nan
    else:
        i = int(np.argmin(np.abs(sets.tt - t_sunset.tt)))
        t_moonset = sets[i]
        lag = (t_moonset.tt - t_sunset.tt) * 1440.0
    return t_sunset, t_moonset, lag


def best_time_yallop(t_sunset, lag_min: float, ts):
    """Waktu terbaik Yallop (1997): T_b = sunset + (4/9) × lag."""
    return ts.tt_jd(t_sunset.tt + (4.0 / 9.0) * lag_min / 1440.0)


def scan_evening(t_sunset, t_moonset, ts, step_min: float = 2.0):
    """Deret waktu dari sunset sampai moonset (atau 60 menit) dengan langkah step_min."""
    end = t_moonset.tt if t_moonset is not None else t_sunset.tt + 60 / 1440.0
    n = max(2, int((end - t_sunset.tt) * 1440.0 / step_min) + 1)
    return ts.tt_jd(np.linspace(t_sunset.tt, end, n))


if __name__ == "__main__":
    ts, eph, name = load_ephemeris()
    print("Ephemeris:", name)
    nm = new_moons(ts, eph, 2026, 2026)
    for t in nm:
        ss, ms, lag = first_evening_after_conjunction(t, ts, eph, MAJENE)
        g = geometry_at(ss, ts, eph, MAJENE, t_conj=t, lag_min=lag)
        print(f"ijtima' {t.utc_datetime().astimezone(WITA):%Y-%m-%d %H:%M} WITA | sunset {g.t_wita[11:16]} | "
              f"h_topo={g.moon_alt_topo:5.2f} h_geo={g.moon_alt_geo:5.2f} elong_geo={g.elong_geo:5.2f} "
              f"umur={g.moon_age_h:5.1f} j lag={lag:5.1f} mnt W'={g.width_arcmin:.3f}'")
