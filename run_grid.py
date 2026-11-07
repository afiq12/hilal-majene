"""
run_grid.py — Jalankan pipeline untuk tiap titik grid Sulbar (data/grid/titik_grid_sulbar.csv) dengan CSV POWER
masing-masing (data/grid/power_<nama>.csv), lalu gabungkan hasil per titik dan per musim untuk peta zona.

Pakai:  python run_grid.py            # semua titik yang CSV-nya sudah ada
Keluaran: output/grid/<nama>.csv (per malam) dan output/grid/ringkasan_grid.csv (per titik x musim x k_ref)
"""
import os, subprocess, sys, shutil
import numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); G=os.path.join(HERE,'data','grid'); OUTG=os.path.join(HERE,'output','grid'); os.makedirs(OUTG,exist_ok=True)
pts=pd.read_csv(os.path.join(G,'titik_grid_sulbar.csv'))
rows=[]
for _,p in pts.iterrows():
    met=os.path.join(G,f'power_{p.nama}.csv')
    if not os.path.exists(met):
        print(f'[lewati] {p.nama}: belum ada {met}'); continue
    out=os.path.join(OUTG,f'{p.nama}.csv')
    if not os.path.exists(out):
        print(f'[jalan] {p.nama} ...')
        subprocess.run([sys.executable, os.path.join(HERE,'run_majene.py'),'--start','2000','--end','2030','--met',met,
                        '--lat',str(p.lat),'--lon',str(p.lon),'--elev',str(p.elev_m)],check=True,cwd=HERE,stdout=subprocess.DEVNULL)
        shutil.copy(os.path.join(HERE,"output","hilal_majene_2000_2030.csv"),out)
    d=pd.read_csv(out); ok=np.isfinite(d.V_phys)&(d.lag_min>0); mab=d.mabims_topo.astype(bool)
    inr=d[ok&d.phys_in_range.astype(bool)]; a,b=np.polyfit(inr.yallop_q,inr.V_phys,1); r=inr.V_phys-(a*inr.yallop_q+b); s,_=np.polyfit(inr.k_total,r,1); kbar=inr.k_total.mean()
    for kref in (0.25,0.35,0.54):
        delta=-(a*-0.014+b+s*(kref-kbar)); vis=(d.V_phys+delta>=0)&ok
        for season in ('hujan','kemarau'):
            sel=(d.season==season).values; n=(mab&sel).sum()
            rows.append(dict(nama=p.nama,lat=p.lat,lon=p.lon,k_ref=kref,musim=season,k_V_median=d.loc[sel,'k_total'].median(),
                             AOD_median=d.loc[sel,'AOD_55'].median(),awan_median=d.loc[sel,'CLOUD_AMT'].median(),N_mabims=int(n),
                             pct_fisis=100*(mab&sel&vis).sum()/n, pct_awan_ok=100*(mab&sel&(d.cloud_ok==True)).sum()/n,
                             pct_fisis_dan_awan=100*(mab&sel&vis&(d.cloud_ok==True)).sum()/n))
if rows:
    out=pd.DataFrame(rows).round(2); out.to_csv(os.path.join(OUTG,'ringkasan_grid.csv'),index=False); print(out.to_string(index=False))
else:
    print('Belum ada CSV POWER di data/grid/. Lihat data/grid/URL_UNDUHAN_POWER_SULBAR.txt')
