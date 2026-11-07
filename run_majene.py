"""
run_majene.py — Pipeline: ijtima' → sunset Majene → geometri → ekstingsi dari CSV meteorologi → kontras →
                 indeks visibilitas fisis vs MABIMS / Yallop / Odeh, untuk 2000–2030.

Pakai:
    python run_majene.py                                  # default 2000–2030, tanpa CSV (klimatologi default)
    python run_majene.py --met data/power_majene_daily.csv --start 2000 --end 2030
    python run_majene.py --met data/power.csv --field-factor 1.4 --sb-factor 0.5 --evenings 2

Keluaran:
    output/hilal_majene_<start>_<end>.csv   satu baris per malam evaluasi
    output/ringkasan_<start>_<end>.txt      tabel kebingungan MABIMS vs fisis, per musim
"""
from __future__ import annotations

import argparse
import os
from dataclasses import asdict

import numpy as np
import pandas as pd

from ephemeris import (MAJENE, WITA, Site, best_time_yallop, first_evening_after_conjunction,
                       geometry_at, load_ephemeris, new_moons, scan_evening, sunset_after)
from atmosphere import (MET_DEFAULTS, airmass_rozenberg, extinction_from_met, met_for_date, monthly_climatology, read_met_csv)
from contrast import evaluate_contrast, mabims, odeh_V, yallop_q

VSEL = lambda r: r.V_phys_crumey
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")


def season_of(month: int) -> str:
    """Musim hujan Sulawesi Barat ≈ Nov–Apr, kemarau ≈ Mei–Okt (sesuaikan dengan klimatologi BMKG Majene)."""
    return "hujan" if month in (11, 12, 1, 2, 3, 4) else "kemarau"


def evaluate_evening(t_conj, t_sunset, t_moonset, lag, ts, eph, site: Site, met, args, evening_idx: int, clim=None):
    g_set = geometry_at(t_sunset, ts, eph, site, t_conj=t_conj, lag_min=lag)
    date_local = pd.Timestamp(g_set.t_wita[:10])
    m = met_for_date(met, date_local, MET_DEFAULTS, clim)
    ext = extinction_from_met(m["RH2M"], m["T2M"], site.elev_m, site.lat_deg, date_local.dayofyear,
                              aod_550=m.get("AOD_55", np.nan))
    k = ext.k_total
    year = date_local.year + date_local.dayofyear / 365.25

    row = dict(evening=evening_idx, conj_utc=t_conj.utc_datetime().strftime("%Y-%m-%d %H:%M"),
               conj_wita=t_conj.utc_datetime().astimezone(WITA).strftime("%Y-%m-%d %H:%M"),
               date_local=date_local.strftime("%Y-%m-%d"), month=date_local.month, season=season_of(date_local.month),
               sunset_wita=g_set.t_wita[11:19], lag_min=lag)
    row.update({f"set_{kk}": v for kk, v in asdict(g_set).items()
                if kk in ("moon_alt_topo", "moon_alt_topo_refr", "moon_alt_geo", "elong_geo", "elong_topo",
                          "arcv_topo", "arcv_geo", "daz", "width_arcmin", "moon_age_h", "illum_frac")})
    row.update(dict(RH2M=m["RH2M"], T2M=m["T2M"], AOD_55=m.get("AOD_55", np.nan), CLOUD_AMT=m.get("CLOUD_AMT", np.nan),
                    met_source=m["met_source"], k_rayleigh=ext.k_rayleigh, k_aerosol=ext.k_aerosol,
                    k_aerosol_src=ext.source_aerosol, k_water=ext.k_water, k_total=k))

    # --- MABIMS pada sunset (tiga varian definisi tinggi; elongasi geosentrik seperti praktik Kemenag)
    row["mabims_topo"] = mabims(g_set.moon_alt_topo, g_set.elong_geo)
    row["mabims_topo_refr"] = mabims(g_set.moon_alt_topo_refr, g_set.elong_geo)
    row["mabims_geo"] = mabims(g_set.moon_alt_geo, g_set.elong_geo)

    # --- Yallop & Odeh pada best time
    if np.isfinite(lag) and lag > 0:
        t_best = best_time_yallop(t_sunset, lag, ts)
        g_best = geometry_at(t_best, ts, eph, site, t_conj=t_conj, lag_min=lag)
        q, qzone = yallop_q(g_best.arcv_geo, g_best.width_arcmin)
        V, vzone = odeh_V(g_best.arcv_topo, g_best.width_arcmin)
        row.update(best_wita=g_best.t_wita[11:19], best_sun_alt=g_best.sun_alt_topo, best_moon_alt=g_best.moon_alt_topo,
                   yallop_q=q, yallop_zone=qzone, odeh_V=V, odeh_zone=vzone)
    else:
        row.update(best_wita=None, best_sun_alt=np.nan, best_moon_alt=np.nan, yallop_q=np.nan, yallop_zone="Bulan terbenam lebih dulu",
                   odeh_V=np.nan, odeh_zone="Bulan terbenam lebih dulu")

    # --- Model fisis: pindai sunset → moonset, ambil V_phys maksimum
    best = None       # maksimum V di dalam rentang validitas rumus senja (h_sun ≤ −4°)
    best_any = None   # maksimum V tanpa memandang rentang (fallback, ditandai phys_in_range=False)
    if np.isfinite(lag) and lag > 0:
        for t in scan_evening(t_sunset, t_moonset, ts, step_min=args.step):
            g = geometry_at(t, ts, eph, site, t_conj=t_conj, lag_min=lag)
            if g.moon_alt_topo < 0.5:   # pusat Bulan < 0,5°: anggap sudah tidak teramati
                continue
            X = airmass_rozenberg(g.moon_alt_topo_refr)
            r = evaluate_contrast(g.sun_alt_topo, g.moon_alt_topo_refr, g.elong_topo, g.phase_angle_topo,
                                  g.illum_frac, g.moon_sd_arcmin, g.width_arcmin, g.moon_dist_km, k, X, year,
                                  field_factor=args.field_factor, crescent_sb_factor=args.sb_factor,
                                  summation_length_arcmin=args.sum_length)
            if best_any is None or VSEL(r) > VSEL(best_any[1]):
                best_any = (g, r)
            if r.in_model_range and (best is None or VSEL(r) > VSEL(best[1])):
                best = (g, r)
    # Jika tidak ada sampel di dalam rentang rumus senja (Bulan terbenam sebelum h_sun ≤ −4°, yaitu < ~16 menit
    # setelah sunset), malam itu dinyatakan 'tidak dapat dievaluasi' dan secara fisis dianggap tak teramati —
    # bukan dinilai dengan rumus di luar rentangnya. (Patat dkk. 2006 pun hanya mengukur sampai h_sun ≈ −4°.)
    row["phys_evaluable"] = best is not None
    if best is None and best_any is not None:
        g, r = best_any
        row["phys_note"] = f"Bulan terbenam sebelum h_sun=-4 (V luar-rentang {VSEL(r):+.2f} diabaikan)"
    else:
        row["phys_note"] = ""
    if best is not None:
        g, r = best
        row.update(phys_time_wita=g.t_wita[11:19], phys_sun_alt=g.sun_alt_topo, phys_moon_alt=g.moon_alt_topo_refr,
                   phys_X=r.X_moon, phys_B_sky_cdm2=r.B_sky_cdm2, phys_L_cres_obs_cdm2=r.L_cres_obs_cdm2,
                   phys_contrast=r.contrast, phys_E_elem_lux=r.E_elem_lux, phys_TH_lux=r.TH_schaefer_lux,
                   V_phys=VSEL(r), V_phys_schaefer=r.V_phys_schaefer, V_phys_crumey=r.V_phys_crumey,
                   phys_C_th_crumey=r.C_th_crumey, phys_ricco_diam_arcmin=r.ricco_diam_arcmin,
                   phys_len_eff_arcmin=r.crescent_len_eff_arcmin, phys_in_range=r.in_model_range,
                   phys_visible=bool(VSEL(r) >= 0.0))
    else:
        row.update(phys_time_wita=None, phys_sun_alt=np.nan, phys_moon_alt=np.nan, phys_X=np.nan, phys_B_sky_cdm2=np.nan,
                   phys_L_cres_obs_cdm2=np.nan, phys_contrast=np.nan, phys_E_elem_lux=np.nan, phys_TH_lux=np.nan,
                   V_phys=-np.inf, V_phys_schaefer=-np.inf, V_phys_crumey=-np.inf, phys_C_th_crumey=np.nan,
                   phys_ricco_diam_arcmin=np.nan, phys_len_eff_arcmin=np.nan, phys_in_range=False, phys_visible=False)
    # Awan: visibilitas efektif (fisis ∧ langit cukup cerah) — ambang awan parameter
    cl = row["CLOUD_AMT"]
    row["cloud_ok"] = bool(np.isfinite(cl) and cl <= args.cloud_max) if np.isfinite(cl) else None
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2000)
    ap.add_argument("--end", type=int, default=2030)
    ap.add_argument("--met", type=str, default=None, help="CSV meteorologi harian (NASA POWER / ERA5)")
    ap.add_argument("--field-factor", type=float, default=1.0, help="faktor lapangan F pada ambang (Blackwell & Blackwell: 2.4)")
    ap.add_argument("--sb-factor", type=float, default=1.0, help="faktor kalibrasi luminansi sabit [V1]")
    ap.add_argument("--step", type=float, default=2.0, help="langkah pindai waktu (menit)")
    ap.add_argument("--threshold", choices=["crumey", "schaefer"], default="crumey", help="hukum ambang yang dipakai untuk V_phys (utama: crumey)")
    ap.add_argument("--sky-model", choices=["sultan07", "schaefer98"], default="sultan07", help="model langit senja")
    ap.add_argument("--sky-scale", type=float, default=1.0, help="faktor skala kecerahan langit senja Schaefer (uji sensitivitas)")
    ap.add_argument("--sum-length", type=float, default=None, help="panjang penjumlahan spasial sepanjang sabit (menit busur); default = θ_res")
    ap.add_argument("--evenings", type=int, default=1, help="jumlah malam dievaluasi setelah ijtima' (1 atau 2)")
    ap.add_argument("--cloud-max", type=float, default=70.0, help="tutupan awan maks (%%) agar dianggap 'langit terbuka'")
    ap.add_argument("--lat", type=float, default=MAJENE.lat_deg)
    ap.add_argument("--lon", type=float, default=MAJENE.lon_deg)
    ap.add_argument("--elev", type=float, default=MAJENE.elev_m)
    args = ap.parse_args()

    import contrast as _c; _c.SKY_SCALE = args.sky_scale; _c.SKY_MODEL = args.sky_model
    global VSEL
    VSEL = (lambda r: r.V_phys_crumey) if args.threshold == 'crumey' else (lambda r: r.V_phys_schaefer)
    site = Site(lat_deg=args.lat, lon_deg=args.lon, elev_m=args.elev)
    ts, eph, ephname = load_ephemeris()
    met = read_met_csv(args.met) if args.met else None
    clim = monthly_climatology(met) if met is not None else None
    print(f"Ephemeris: {ephname} | situs: {site.lat_deg}, {site.lon_deg}, {site.elev_m} m | met: {args.met or 'default klimatologi'}")

    rows = []
    conj = new_moons(ts, eph, args.start, args.end)
    for i, t_conj in enumerate(conj):
        t_sunset, t_moonset, lag = first_evening_after_conjunction(t_conj, ts, eph, site)
        rows.append(evaluate_evening(t_conj, t_sunset, t_moonset, lag, ts, eph, site, met, args, 1, clim))
        if args.evenings >= 2:
            t_sunset2 = sunset_after(ts.tt_jd(t_sunset.tt + 0.5), ts, eph, site, "sun")
            t_moonset2 = sunset_after(t_sunset2, ts, eph, site, "moon")
            lag2 = (t_moonset2.tt - t_sunset2.tt) * 1440.0
            rows.append(evaluate_evening(t_conj, t_sunset2, t_moonset2, lag2, ts, eph, site, met, args, 2, clim))
        if (i + 1) % 24 == 0:
            print(f"  ... {i + 1}/{len(conj)} ijtima' selesai")

    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    csv_path = os.path.join(OUT, f"hilal_majene_{args.start}_{args.end}.csv")
    df.to_csv(csv_path, index=False)
    print("Tersimpan:", csv_path)

    # ---------------------------------------------------------------- ringkasan
    d1 = df[df.evening == 1]
    lines = [f"Ringkasan malam pertama setelah ijtima', Majene, {args.start}–{args.end}  (N = {len(d1)})",
             f"Ephemeris {ephname}; met = {args.met or 'default'}; ambang = {args.threshold}; langit = {args.sky_model}; F = {args.field_factor}; sb_factor = {args.sb_factor}", ""]

    def confusion(sub, label):
        tab = pd.crosstab(sub["mabims_topo"], sub["phys_visible"], rownames=["MABIMS (topo, elong geo)"],
                          colnames=["Fisis terlihat (V_phys ≥ 0)"], dropna=False)
        return [f"--- {label} (N = {len(sub)})", tab.to_string(), ""]

    lines += confusion(d1, "Semua")
    for s in ("hujan", "kemarau"):
        lines += confusion(d1[d1.season == s], f"Musim {s}")
    lines += ["Catatan: hasil dengan phys_in_range = False (h_sun di atas batas model langit saat V maksimum) berada di luar rentang validitas rumus senja Schaefer [V3].",
              f"  Proporsi malam dengan V maksimum di luar rentang: {(~d1.phys_in_range.astype(bool)).mean():.0%}"]
    summ = "\n".join(lines)
    with open(os.path.join(OUT, f"ringkasan_{args.start}_{args.end}.txt"), "w") as f:
        f.write(summ)
    print(summ)


if __name__ == "__main__":
    main()
