#!/usr/bin/env python3
"""Metro case studies (county subsets of the statewide NC layer) + manuscript
figures. Consistent store set (the statewide 349-store OSM extract) across all
metros. No network calls. Writes PNGs to manuscript/figures/ and a JSON table.
"""
import json, warnings
import numpy as np, pandas as pd, geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from libpysal.weights import Queen
from esda.moran import Moran

warnings.filterwarnings("ignore")
REPO = "/home/user/wilbertfletcher/arc-gis-mapping-hair-glue-safety"
NC_ENR = f"{REPO}/outputs/streamlit_queries/nc_usa/beauty_access_enriched.geojson"
NC_STORES = f"{REPO}/outputs/streamlit_queries/nc_usa/beauty_supply_stores.geojson"
FIG = f"{REPO}/manuscript/figures"; import os; os.makedirs(FIG, exist_ok=True)

INK="#1b1b1b"; MUTED="#6b6b6b"; ACCENT="#7a1f3d"  # NCCU maroon for highlight
plt.rcParams.update({"font.size":10,"axes.edgecolor":"#999","axes.linewidth":.6,
                     "axes.spines.top":False,"axes.spines.right":False,
                     "figure.dpi":200,"savefig.dpi":200,"font.family":"DejaVu Sans"})

METROS = {"119":"Charlotte","183":"Raleigh","081":"Greensboro","063":"Durham","051":"Fayetteville"}

g = gpd.read_file(NC_ENR)
g["COUNTYFP"] = g["COUNTYFP"].astype(str).str.zfill(3)
for c in ["total_pop","black_pop","median_income","poverty_rate","pct_black",
          "nearest_store_km","store_count_5km","underserved_index"]:
    g[c] = pd.to_numeric(g[c], errors="coerce")
g = g[g["total_pop"] > 0].copy()
stores = gpd.read_file(NC_STORES)

def fnum(x,nd=2):
    return None if x is None or (isinstance(x,float) and np.isnan(x)) else round(float(x),nd)

def moran(sub, col):
    s = sub.dropna(subset=[col]).reset_index(drop=True)
    if len(s) < 15: return (None,None)
    w = Queen.from_dataframe(s, use_index=False); w.transform="r"
    m = Moran(s[col].values, w, permutations=999)
    return (fnum(m.I,3), fnum(m.p_sim,3))

# ---- metro table ----
rows=[]
for fp,name in METROS.items():
    s=g[g.COUNTYFP==fp]
    maj=s[s.pct_black>=50]; oth=s[s.pct_black<50]
    mi,mp=moran(s,"nearest_store_km")
    rows.append({"metro":name,"tracts":int(len(s)),
        "pop":int(s.total_pop.sum()),
        "med_pct_black":fnum(s.pct_black.median(),1),
        "med_nearest_km":fnum(s.nearest_store_km.median(),1),
        "pct_no_store_5km":fnum(100*(s.store_count_5km==0).mean(),1),
        "mean_store_5km":fnum(s.store_count_5km.mean(),2),
        "nearest_majBlack":fnum(maj.nearest_store_km.median(),1) if len(maj) else None,
        "nearest_other":fnum(oth.nearest_store_km.median(),1) if len(oth) else None,
        "moran_dist_I":mi,"moran_dist_p":mp})
tbl=pd.DataFrame(rows)
tbl.to_json(f"{FIG}/../analysis/metro_table.json", orient="records", indent=2)
print(tbl.to_string(index=False))

# ---- Figure 1: statewide choropleth (nearest-store distance) + store overlay ----
# Store extract is noisy: 161/349 OSM points fall outside NC. Clip to the state
# (+8 km) for display; distances in the data were computed against the full set.
nc_poly = g.to_crs(4326).union_all()
inside = stores[stores.to_crs(4326).within(nc_poly.buffer(0.08))]
fig,ax=plt.subplots(figsize=(7.2,5.2))
gp=g.to_crs(3857); sp=inside.to_crs(3857)
cap=np.nanpercentile(g["nearest_store_km"],95)
gp["d"]=gp["nearest_store_km"].clip(upper=cap)
gp.plot(column="d",cmap="viridis_r",linewidth=0.05,edgecolor="#ffffff",
        legend=True,ax=ax,
        legend_kwds={"label":"Distance to nearest store (km)","shrink":0.6})
sp.plot(ax=ax,color=ACCENT,markersize=6,marker="o",alpha=0.9,
        edgecolor="white",linewidth=0.2)
ax.set_title("Beauty-supply retail access across North Carolina",
             fontsize=12,color=INK,loc="left",weight="bold")
ax.text(0,-0.04,f"Dark = far from a store (worse access).  Maroon dots = stores (n={len(inside)} within NC).",
        transform=ax.transAxes,fontsize=8,color=MUTED,va="top")
ax.text(0,-0.095,"67.5% of NC tracts have no store within 5 km; access deserts are predominantly rural.",
        transform=ax.transAxes,fontsize=8,color=MUTED,va="top")
ax.axis("off")
plt.tight_layout(); plt.savefig(f"{FIG}/fig1_nc_choropleth.png",bbox_inches="tight"); plt.close()

# ---- Figure 2: access gradient by percent-Black quartile ----
g["q"]=pd.qcut(g["pct_black"],4,labels=["Q1\nlowest\n%Black","Q2","Q3","Q4\nhighest\n%Black"])
agg=g.groupby("q").agg(med_near=("nearest_store_km","median"),
                       no_store=("store_count_5km",lambda s:100*(s==0).mean()))
fig,(a1,a2)=plt.subplots(1,2,figsize=(8,3.6))
bars=a1.bar(range(4),agg["med_near"],color="#2a6f97",width=0.68,zorder=3)
bars[3].set_color(ACCENT)
a1.set_xticks(range(4)); a1.set_xticklabels(agg.index,fontsize=8)
a1.set_ylabel("Median distance to\nnearest store (km)",fontsize=9)
a1.set_title("Closer where more residents are Black",fontsize=10,loc="left",color=INK)
for i,v in enumerate(agg["med_near"]): a1.text(i,v+0.3,f"{v:.1f}",ha="center",fontsize=8,color=INK)
a1.grid(axis="y",color="#eee",zorder=0)
bars2=a2.bar(range(4),agg["no_store"],color="#2a6f97",width=0.68,zorder=3)
bars2[3].set_color(ACCENT)
a2.set_xticks(range(4)); a2.set_xticklabels(agg.index,fontsize=8)
a2.set_ylabel("% of tracts with no\nstore within 5 km",fontsize=9)
a2.set_title("Fewer access deserts, too",fontsize=10,loc="left",color=INK)
for i,v in enumerate(agg["no_store"]): a2.text(i,v+1.2,f"{v:.0f}%",ha="center",fontsize=8,color=INK)
a2.grid(axis="y",color="#eee",zorder=0)
fig.suptitle("Beauty-supply access improves with tract percent-Black (2,649 NC tracts)",
             fontsize=11,x=0.01,ha="left",weight="bold",color=INK)
plt.tight_layout(rect=[0,0,1,0.95]); plt.savefig(f"{FIG}/fig2_quartile_gradient.png",bbox_inches="tight"); plt.close()

# ---- Figure 3: metro comparison ----
fig,ax=plt.subplots(figsize=(7.2,3.6))
m=tbl.sort_values("med_nearest_km")
x=range(len(m))
bars=ax.bar(x,m["pct_no_store_5km"],color="#2a6f97",width=0.6,zorder=3)
ax.set_xticks(list(x)); ax.set_xticklabels(m["metro"],fontsize=9)
ax.set_ylabel("% of tracts with no\nstore within 5 km",fontsize=9)
ax.set_title("Retail access in five NC metros (county subsets of statewide layer)",
             fontsize=11,loc="left",weight="bold",color=INK)
for i,(v,med) in enumerate(zip(m["pct_no_store_5km"],m["med_nearest_km"])):
    ax.text(i,v+1,f"{v:.0f}%",ha="center",fontsize=8,color=INK)
ax.grid(axis="y",color="#eee",zorder=0)
ax.annotate("Labels: % of tracts with no store within 5 km. All five metros have far better access than rural NC (67.5% statewide have none).",
            xy=(0,-0.18),xycoords="axes fraction",fontsize=8,color=MUTED)
plt.tight_layout(); plt.savefig(f"{FIG}/fig3_metros.png",bbox_inches="tight"); plt.close()
print("figures written to", FIG)
