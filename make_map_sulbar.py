"""make_map_sulbar.py — Gambar 8: peta zona visibilitas fisis Sulawesi Barat per musim dari output/grid/ringkasan_grid.csv.
Sel POWER 0,5° x 0,625° digambar sebagai kotak berwarna (peluang efektif: fisis dan langit terbuka, jangkar k_ref = 0,35)."""
import os, sys, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap, Normalize
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#c3c2b7','axes.labelcolor':'#52514e','xtick.color':'#898781','ytick.color':'#898781','grid.color':'#e1e0d9','axes.grid':True,'axes.axisbelow':True,'figure.facecolor':'#fcfcfb','axes.facecolor':'#fcfcfb'})
HERE=os.path.dirname(os.path.abspath(__file__)); EN=os.environ.get('LANG_FIG','id')=='en'
FIG=os.path.join(HERE,'output','fig','en' if EN else ''); os.makedirs(FIG,exist_ok=True)
T=lambda id_,en: en if EN else id_
kref=float(sys.argv[1]) if len(sys.argv)>1 else 0.35
g=pd.read_csv(os.path.join(HERE,'output','grid','ringkasan_grid.csv')); g=g[g.k_ref==kref]
def cell(lat,lon):  # sel POWER: lintang kelipatan 0,5 (pusat ±0,25), bujur kelipatan 0,625
    la=np.floor(lat/0.5)*0.5; lo=np.floor(lon/0.625)*0.625; return la,lo
cmap=LinearSegmentedColormap.from_list('blues',['#f3f7fd','#cde2fb','#86b6ef','#3987e5','#1c5cab','#0d366b'])
label={'majene':'Majene','polewali':'Polewali','mamuju':'Mamuju','mamuju_tengah':'Mamuju Tengah','pasangkayu':'Pasangkayu','mamasa':'Mamasa','laut_barat_mamuju':T('laut (Mamuju)','sea (Mamuju)'),'laut_barat_majene':T('laut (Majene)','sea (Majene)')}
fig,axes=plt.subplots(1,2,figsize=(11,5.6),constrained_layout=True)
vmax=max(12,g.pct_fisis_dan_awan.max()); norm=Normalize(0,vmax)
for ax,season in zip(axes,('hujan','kemarau')):
    s=g[g.musim==season]
    off={'majene':(8,-14),'polewali':(8,4),'mamuju':(8,4),'mamuju_tengah':(8,4),'pasangkayu':(8,4),'mamasa':(8,-14),'laut_barat_mamuju':(-8,4),'laut_barat_majene':(-8,4)}
    for _,r in s.iterrows():
        ax.scatter(r.lon,r.lat,s=420,color=cmap(norm(r.pct_fisis_dan_awan)),edgecolor='#52514e',lw=0.8,zorder=3)
        ha='right' if r.nama.startswith('laut') else 'left'
        ax.annotate(f"{label[r.nama]}: {r.pct_fisis_dan_awan:.0f} %, k = {r.k_V_median:.2f}",(r.lon,r.lat),xytext=off[r.nama],textcoords='offset points',fontsize=8,ha=ha,color='#0b0b0b',zorder=4)
    ax.set_xlim(117.9,120.3); ax.set_ylim(-4.0,-0.8); ax.set_aspect('equal'); ax.set_xlabel(T('Bujur Timur (°)','Longitude (°E)')); ax.set_ylabel(T('Lintang (°)','Latitude (°)'))
    ax.set_title(T(f'Musim {season}','Wet season (Nov–Apr)' if season=='hujan' else 'Dry season (May–Oct)'),loc='left',fontsize=11,color='#0b0b0b')
sm=plt.cm.ScalarMappable(cmap=cmap,norm=norm); cb=fig.colorbar(sm,ax=axes,shrink=0.8,pad=0.02); cb.set_label(T('% malam lolos MABIMS yang fisis terlihat dan langit terbuka (awan ≤ 70 %)','% of MABIMS-passing evenings above threshold with cloud ≤ 70 %')); cb.outline.set_visible(False)
fig.suptitle(T(f'Zona visibilitas fisis hilal Sulawesi Barat, 2000–2030, pada 8 titik reanalisis POWER (MERRA-2 0,5° × 0,625°; awan dan AOD CERES 1°); jangkar k_ref = {kref}; k = median k_V musiman',f'Effective crescent-sighting chance in West Sulawesi, 2000–2030, at 8 reanalysis points (MERRA-2 0.5° × 0.625°; CERES cloud and AOD 1°); anchor $k_{{\\rm ref}}$ = {kref}; k = seasonal median $k_V$'),x=0.01,ha='left',fontsize=10,color='#0b0b0b')
fig.savefig(os.path.join(FIG,'fig8_peta_zona_sulbar.png'),dpi=200); print('ok')
