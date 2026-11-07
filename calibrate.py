"""
calibrate.py — Kalibrasi ofset indeks visibilitas fisis dan analisis sensitivitas.

Tiga parameter psikofisis (F, sb_factor, panjang penjumlahan) hampir degenerat: pengaruhnya pada V_phys adalah ofset
aditif δ = log10(sb_factor · L_sum/θ / F). Maka kalibrasi dilakukan sebagai SATU ofset δ yang dijangkarkan pada:
    V_cal(q = q0, k = k_ref) = 0,   q0 = −0,014 (batas zona A/B Yallop), k_ref = ekstingsi "acuan" data Yallop.
k_ref tidak diketahui pasti → dilaporkan sebagai pita sensitivitas (0,25; 0,35; rata-rata Majene).
Keluaran per malam: V_cal dan k_crit = k_malam + V_cal/|∂V/∂k| (ekstingsi maksimum yang masih ditoleransi).

Pakai: python calibrate.py output/run_sum24.csv
"""
import sys, numpy as np, pandas as pd

path = sys.argv[1] if len(sys.argv) > 1 else "output/hilal_majene_2000_2030.csv"
Q0 = -0.014
d = pd.read_csv(path)
d = d[d.evening == 1].copy()
ok = np.isfinite(d.V_phys) & (d.lag_min > 0)
inr = d[ok & d.phys_in_range.astype(bool)]

a, b = np.polyfit(inr.yallop_q, inr.V_phys, 1)
resid = inr.V_phys - (a * inr.yallop_q + b)
s, i0 = np.polyfit(inr.k_total, resid, 1)          # ∂V/∂k (dex per mag/airmass)
kbar = inr.k_total.mean()
print(f"Fit (N={len(inr)}): V = {a:.2f} q {b:+.2f};  ∂V/∂k = {s:.2f} dex per mag/airmass;  k̄ = {kbar:.3f}")

rows = []
for kref in (0.25, 0.35, round(kbar, 2)):
    V_at_anchor = a * Q0 + b + s * (kref - kbar)
    delta = -V_at_anchor
    V_cal = d.V_phys + delta
    vis = (V_cal >= 0) & ok
    d[f"V_cal_k{kref}"] = V_cal
    d[f"k_crit_k{kref}"] = d.k_total + V_cal / abs(s)
    mab = d.mabims_topo.astype(bool)
    for season in ("semua", "hujan", "kemarau"):
        sel = np.ones(len(d), bool) if season == "semua" else (d.season == season).values
        n_mab = (mab & sel).sum()
        n_mab_vis = (mab & vis & sel).sum()
        n_vis_nomab = (~mab & vis & sel).sum()
        rows.append(dict(k_ref=kref, delta=round(delta, 2), musim=season, N=int(sel.sum()), MABIMS_lolos=int(n_mab),
                         lolos_dan_fisis=int(n_mab_vis), lolos_tapi_tak_fisis=int(n_mab - n_mab_vis),
                         pct_lolos_tak_fisis=round(100 * (n_mab - n_mab_vis) / max(n_mab, 1), 1),
                         fisis_tanpa_MABIMS=int(n_vis_nomab),
                         k_crit_median_lolos=round(float(d.loc[mab & sel & ok, f"k_crit_k{kref}"].median()), 2)))
tab = pd.DataFrame(rows)
print(tab.to_string(index=False))
out = path.replace(".csv", "_calibrated.csv")
d.to_csv(out, index=False); print("Tersimpan:", out)
