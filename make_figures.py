"""make_figures.py — Gambar 2–5 dari output/hilal_majene_2000_2030.csv dan data/power_majene_daily.csv."""
import os, sys
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from atmosphere import read_met_csv
HERE=os.path.dirname(os.path.abspath(__file__)); EN=os.environ.get('LANG_FIG','id')=='en'
FIG=os.path.join(HERE,'output','fig','en' if EN else ''); os.makedirs(FIG,exist_ok=True)
T=lambda id_,en: en if EN else id_

def _finalize(fig):
    """Mode EN (naskah): hapus judul internal (informasi ada di caption) dan rapikan."""
    if EN:
        for ax in fig.axes:
            for loc in ('left','center','right'):
                if ax.get_title(loc=loc): ax.set_title('',loc=loc)
        try: fig.suptitle('')
        except Exception: pass

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,
 'axes.edgecolor':'#c3c2b7','axes.labelcolor':'#52514e','xtick.color':'#898781','ytick.color':'#898781','grid.color':'#e1e0d9','axes.grid':True,'axes.axisbelow':True,'figure.facecolor':'#fcfcfb','axes.facecolor':'#fcfcfb'})
BLUE,ORANGE,AQUA='#2a78d6','#eb6834','#1baf7a'
met=sys.argv[1] if len(sys.argv)>1 else os.path.join(HERE,'data','power_majene_daily.csv')
res=sys.argv[2] if len(sys.argv)>2 else os.path.join(HERE,'output','hilal_majene_2000_2030.csv')
m=read_met_csv(met); m=m[m.AOD_55.notna()]
d=pd.read_csv(res); d=d[d.evening==1]
bulan=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'] if EN else ['Jan','Feb','Mar','Apr','Mei','Jun','Jul','Agu','Sep','Okt','Nov','Des']

fig,ax=plt.subplots(1,3,figsize=(11,3.6),constrained_layout=True); g=m.groupby(m.index.month)
for a,(col,lab,c) in zip(ax,[('RH2M',T('Kelembapan relatif 2 m (%)','Relative humidity at 2 m (%)'),BLUE),('AOD_55',T('AOD 550 nm (CERES/MERRA-2)','Aerosol optical depth, 550 nm (CERES)'),ORANGE),('CLOUD_AMT',T('Tutupan awan (%)','Cloud amount (%)'),AQUA)]):
    q25,q50,q75=g[col].quantile(.25),g[col].median(),g[col].quantile(.75)
    a.fill_between(range(1,13),q25,q75,color=c,alpha=.18,lw=0); a.plot(range(1,13),q50,color=c,lw=2,marker='o',ms=4)
    a.set_xticks(range(1,13)); a.set_xticklabels([b if i%2==0 else '' for i,b in enumerate(bulan)],fontsize=9); a.axvspan(4.5,10.5,color='#898781',alpha=.06,lw=0)
    if EN: a.set_ylabel(lab)
    else: a.set_title(lab,loc='left',fontsize=10,color='#0b0b0b')
ax[0].text(7.5,ax[0].get_ylim()[1],T('kemarau','dry season'),ha='center',va='top',fontsize=8,color='#898781')
fig.suptitle(T(f'Klimatologi harian Majene {m.index.year.min()}–{m.index.year.max()} (median dan rentang kuartil), NASA POWER',f'Daily climatology at Majene, {m.index.year.min()}–{m.index.year.max()} (median and interquartile range), NASA POWER'),x=0.01,ha='left',fontsize=11,color='#0b0b0b')
_finalize(fig)
fig.savefig(os.path.join(FIG,'fig2_klimatologi_majene.png'),dpi=300); fig.savefig(os.path.join(FIG,'fig2_klimatologi_majene.pdf'))

fig,a=plt.subplots(figsize=(7,3.6),constrained_layout=True)
a.boxplot([d[d.month==i].k_total.values for i in range(1,13)],widths=.55,patch_artist=True,flierprops=dict(marker='o',ms=3,mfc=BLUE,mec='none',alpha=.5),
          medianprops=dict(color='#0b0b0b',lw=1.5),boxprops=dict(facecolor=BLUE,alpha=.25,edgecolor=BLUE),whiskerprops=dict(color=BLUE),capprops=dict(color=BLUE))
a.set_xticks(range(1,13)); a.set_xticklabels(bulan); a.set_ylabel(T('k_V (mag per airmass)','$k_V$ (mag per air mass)'))
a.axhline(0.30,color=ORANGE,lw=1.5,ls='--'); a.text(12.4,0.305,T('k = 0,30 (OASA Surabaya, Hasby dkk. 2023)','k = 0.30 (Surabaya, Hasby et al. 2023)'),ha='right',va='bottom',fontsize=8,color=ORANGE)
a.set_title(T(f'Koefisien ekstingsi V Majene pada {len(d)} malam ke-29 (aerosol dari AOD, uap air dari RH & T)',f'Visual extinction coefficient at Majene on {len(d)} evenings of the 29th lunar day (aerosol from AOD, water vapour from RH and T)'),loc='left',fontsize=10,color='#0b0b0b')
_finalize(fig)
fig.savefig(os.path.join(FIG,'fig3_kV_bulanan.png'),dpi=300); fig.savefig(os.path.join(FIG,'fig3_kV_bulanan.pdf'))

ok=d[np.isfinite(d.V_phys)&(d.lag_min>0)].copy()
fig,a=plt.subplots(figsize=(7.5,5),constrained_layout=True)
sc=a.scatter(ok.set_width_arcmin,ok.set_arcv_geo,c=ok.V_phys,cmap='Blues',vmin=-5,vmax=2,s=22,edgecolor='#fcfcfb',lw=.5)
W=np.linspace(0,0.8,200); yq=11.8371-6.3226*W+0.7319*W**2-0.1018*W**3
a.plot(W,yq,color=ORANGE,lw=2); a.text(0.62,8.9,'Yallop q = 0',color=ORANGE,fontsize=9)
a.plot(W,yq-1.6,color=ORANGE,lw=1,ls=':'); a.text(0.62,7.1,T('q = −0,16','q = −0.16'),color=ORANGE,fontsize=8)
a.axhline(3.83,color=AQUA,lw=1.5,ls='--'); a.text(0.79,4.0,T('MABIMS h = 3° (ARCV ≈ 3,8°)','MABIMS h = 3° (ARCV ≈ 3.8°)'),color=AQUA,fontsize=9,ha='right',va='bottom')
a.axvline(15.5*(1-np.cos(np.radians(6.4))),color=AQUA,lw=1.5,ls='--'); a.text(0.10,15.3,T('MABIMS\nelong 6,4°','MABIMS\nelong. 6.4°'),color=AQUA,fontsize=8)
cb=fig.colorbar(sc,ax=a,pad=.02); cb.set_label(T('V_phys (belum dikalibrasi; log10 E/TH)','$V_{\\rm phys}$ (untuned)')); cb.outline.set_visible(False)
a.set_xlabel(T("Lebar sabit toposentrik W' saat sunset (menit busur)","Topocentric crescent width W' at sunset (arcmin)")); a.set_ylabel(T('ARCV geosentrik saat sunset (°)','Geocentric ARCV at sunset (°)')); a.set_xlim(0,0.85); a.set_ylim(-2,17)
a.set_title(T('Malam ke-29 di Majene: indeks fisis vs kriteria geometris\n(malam dengan Bulan terbenam sebelum Matahari mencapai −2° tidak digambar)','Evenings of the 29th lunar day at Majene: physical index against the geometric criteria\n(evenings on which the Moon set before the Sun reached −2° are omitted)'),loc='left',fontsize=10,color='#0b0b0b')
_finalize(fig)
fig.savefig(os.path.join(FIG,'fig4_arcv_w_vphys.png'),dpi=300); fig.savefig(os.path.join(FIG,'fig4_arcv_w_vphys.pdf'))

inr=ok[ok.phys_in_range]; a_,b_=np.polyfit(inr.yallop_q,inr.V_phys,1); r=inr.V_phys-(a_*inr.yallop_q+b_); s_,i_=np.polyfit(inr.k_total,r,1)
fig,ax=plt.subplots(1,2,figsize=(10,3.8),constrained_layout=True)
ax[0].scatter(inr.yallop_q,inr.V_phys,s=16,color=BLUE,alpha=.7,edgecolor='none'); xq=np.linspace(-0.9,0.7,2); ax[0].plot(xq,a_*xq+b_,color=ORANGE,lw=2)
ax[0].axvline(0,color='#898781',lw=1,ls=':'); ax[0].axhline(0,color='#898781',lw=1,ls=':'); ax[0].set_xlabel('Yallop q (best time)'); ax[0].set_ylabel(T('V_phys (belum dikalibrasi)','$V_{\\rm phys}$ (untuned)'))
ax[0].set_title(f'V = {a_:.2f} q {b_:+.2f};  Spearman ρ = {inr.V_phys.corr(inr.yallop_q,method="spearman"):.2f}  (N = {len(inr)})',loc='left',fontsize=10,color='#0b0b0b')
ax[1].scatter(inr.k_total,r,s=16,color=BLUE,alpha=.7,edgecolor='none'); xk=np.linspace(inr.k_total.min(),inr.k_total.max(),2); ax[1].plot(xk,s_*xk+i_,color=ORANGE,lw=2)
ax[1].set_xlabel(T('k_V (mag per airmass)','$k_V$ (mag per air mass)')); ax[1].set_ylabel(T('residual V_phys (dex)','residual $V_{\\rm phys}$ (dex)')); ax[1].set_title(T(f'Sisa setelah geometri dikeluarkan: {s_*0.1:+.2f} dex per +0,1 k_V',f'Residual after removing the geometric term: {s_*0.1:+.2f} dex per +0.1 $k_V$'),loc='left',fontsize=10,color='#0b0b0b')
_finalize(fig)
fig.savefig(os.path.join(FIG,'fig5_vphys_vs_yallop_k.png'),dpi=300); fig.savefig(os.path.join(FIG,'fig5_vphys_vs_yallop_k.pdf'))
print(f"fit V={a_:.2f}q{b_:+.2f}; residual slope {s_*0.1:+.2f} dex/0.1k; N_inrange={len(inr)}")
