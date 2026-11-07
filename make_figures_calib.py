"""make_figures_calib.py — Gambar 6 (sensitivitas k_ref) dan 7 (k_crit vs k aktual) dari CSV *_calibrated.csv."""
import os, sys
import pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#c3c2b7','axes.labelcolor':'#52514e','xtick.color':'#898781','ytick.color':'#898781','grid.color':'#e1e0d9','axes.grid':True,'axes.axisbelow':True,'figure.facecolor':'#fcfcfb','axes.facecolor':'#fcfcfb'})
BLUE,ORANGE='#2a78d6','#eb6834'
HERE=os.path.dirname(os.path.abspath(__file__)); EN=os.environ.get('LANG_FIG','id')=='en'
FIG=os.path.join(HERE,'output','fig','en' if EN else ''); os.makedirs(FIG,exist_ok=True)
T=lambda id_,en: en if EN else id_
path=sys.argv[1] if len(sys.argv)>1 else os.path.join(HERE,'output','run_crumey_sultan_calibrated.csv')
tag=sys.argv[2] if len(sys.argv)>2 else ''
d=pd.read_csv(path); ok=np.isfinite(d.V_phys)&(d.lag_min>0); inr=d[ok&d.phys_in_range.astype(bool)]
a,b=np.polyfit(inr.yallop_q,inr.V_phys,1); r=inr.V_phys-(a*inr.yallop_q+b); s,_=np.polyfit(inr.k_total,r,1); kbar=inr.k_total.mean(); mab=d.mabims_topo.astype(bool)
krefs=np.linspace(0.2,0.6,41); fig,ax=plt.subplots(1,2,figsize=(10,4.2),constrained_layout=True)
for season,c in (("hujan",BLUE),("kemarau",ORANGE)):
    sel=(d.season==season).values; pct=[];pj=[]
    for k in krefs:
        delta=-(a*-0.014+b+s*(k-kbar)); vis=(d.V_phys+delta>=0)&ok; n=(mab&sel).sum()
        pct.append(100*(mab&sel&~vis).sum()/n); pj.append(100*(mab&sel&vis&(d.cloud_ok==True)).sum()/n)
    ax[0].plot(krefs,pct,color=c,lw=2,label=T(f'musim {season}','wet season' if season=='hujan' else 'dry season')); ax[1].plot(krefs,pj,color=c,lw=2,label=T(f'musim {season}','wet season' if season=='hujan' else 'dry season'))
for a_ in ax:
    a_.axvspan(0.25,0.35,color='#898781',alpha=.08,lw=0); a_.set_xlabel(T('k_ref (mag/airmass)','$k_{\\rm ref}$ (mag per air mass)')); a_.legend(frameon=False,fontsize=9)
ax[0].set_ylabel(T('% lolos MABIMS tetapi V_cal < 0','% of MABIMS-passing evenings with $V_{\\rm cal}$ < 0')); ax[0].set_title(T('Lolos MABIMS tetapi di bawah ambang mata','MABIMS-passing but below the naked-eye threshold'),loc='left',fontsize=10,color='#0b0b0b')
ax[1].set_ylabel(T('% lolos MABIMS, fisis, dan awan ≤ 70 %','% above threshold and cloud ≤ 70 %')); ax[1].set_title(T('Peluang efektif terlihat','Effective chance of a sighting'),loc='left',fontsize=10,color='#0b0b0b')
fig.suptitle(T(f'Sensitivitas terhadap jangkar kalibrasi k_ref; 215 malam lolos MABIMS, Majene 2000–2030 {tag}',f'Sensitivity to the calibration anchor $k_{{\\rm ref}}$; 215 MABIMS-passing evenings, Majene 2000–2030'),x=0.01,ha='left',fontsize=10,color='#0b0b0b')
fig.savefig(os.path.join(FIG,'fig6_sensitivitas_kref.png'),dpi=200)
fig,a_=plt.subplots(figsize=(7,3.8),constrained_layout=True); bins=np.linspace(0,1.1,23)
for season,c in (("hujan",BLUE),("kemarau",ORANGE)):
    sel=mab&(d.season==season).values&ok
    a_.hist(d.loc[sel,'k_crit_k0.35'],bins=bins,histtype='step',lw=2,color=c,label=T(f'k_crit, {season} (N={sel.sum()})',f"$k_{{\\rm crit}}$, {'wet' if season=='hujan' else 'dry'} season (N={sel.sum()})"))
    a_.hist(d.loc[sel,'k_total'],bins=bins,histtype='stepfilled',alpha=.15,color=c,lw=0,label=T(f'k aktual, {season}',f"actual $k_V$, {'wet' if season=='hujan' else 'dry'} season"))
a_.set_xlabel(T('k_V (mag/airmass)','$k_V$ (mag per air mass)')); a_.set_ylabel(T('jumlah malam','number of evenings')); a_.legend(frameon=False,fontsize=8)
a_.set_title(T('Malam lolos MABIMS: ekstingsi kritis (garis) vs ekstingsi aktual Majene (isi); jangkar k_ref = 0,35','MABIMS-passing evenings: critical extinction (lines) against actual extinction at Majene (filled); anchor $k_{\\rm ref}$ = 0.35'),loc='left',fontsize=9,color='#0b0b0b')
fig.savefig(os.path.join(FIG,'fig7_kcrit_vs_kaktual.png'),dpi=200)
print(f"V={a:.2f}q{b:+.2f}; dV/dk={s:.2f}; N={len(inr)}")
