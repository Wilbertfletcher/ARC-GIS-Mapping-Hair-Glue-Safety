#!/usr/bin/env python3
"""Real statistics for the NC beauty-supply access manuscript.
Reads the repo's own enriched GeoJSON outputs; computes nothing fabricated.
Writes a JSON of results to scratchpad/nc_results.json.
"""
import json, warnings
import numpy as np, pandas as pd, geopandas as gpd
from libpysal.weights import Queen
from esda.moran import Moran
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
BASE = "/home/user/wilbertfletcher/arc-gis-mapping-hair-glue-safety/outputs"
STORES = {
    "nc": f"{BASE}/streamlit_queries/nc_usa/beauty_supply_stores.geojson",
    "greensboro": f"{BASE}/streamlit_queries/greensboro_nc_usa/beauty_supply_stores.geojson",
    "durham": f"{BASE}/durham_beauty_supply_only/beauty_supply_stores.geojson",
}
ENRICHED = {
    "nc": f"{BASE}/streamlit_queries/nc_usa/beauty_access_enriched.geojson",
    "greensboro": f"{BASE}/streamlit_queries/greensboro_nc_usa/beauty_access_enriched.geojson",
    "durham": f"{BASE}/durham_beauty_supply_only/beauty_access_enriched.geojson",
}

def fnum(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)

def analyze(name):
    g = gpd.read_file(ENRICHED[name])
    n_stores = len(gpd.read_file(STORES[name]))
    # clean numerics
    for c in ["total_pop","black_pop","median_income","poverty_rate","pct_black",
              "nearest_store_km","store_count_5km","underserved_index","beauty_access_score"]:
        g[c] = pd.to_numeric(g[c], errors="coerce")
    g = g[g["total_pop"] > 0].copy()
    res = {"name": name, "n_tracts": int(len(g)), "n_stores": int(n_stores)}
    res["total_pop"] = int(g["total_pop"].sum())
    res["pct_black_median"] = fnum(g["pct_black"].median())
    res["tracts_no_store_5km"] = int((g["store_count_5km"] == 0).sum())
    res["pct_tracts_no_store_5km"] = fnum(100*(g["store_count_5km"]==0).mean())
    res["nearest_km_median"] = fnum(g["nearest_store_km"].median())
    res["nearest_km_iqr"] = [fnum(g["nearest_store_km"].quantile(.25)), fnum(g["nearest_store_km"].quantile(.75))]
    res["hotspot_tracts"] = int((g.get("hotspot_flag","")==True).sum()) if "hotspot_flag" in g else None

    # Disparity: majority-Black (>=50%) vs others — access comparison
    maj = g[g["pct_black"]>=50]; oth = g[g["pct_black"]<50]
    res["majBlack_n"] = int(len(maj)); res["othBlack_n"] = int(len(oth))
    res["nearest_km_majBlack"] = fnum(maj["nearest_store_km"].median())
    res["nearest_km_other"]    = fnum(oth["nearest_store_km"].median())
    res["storecount_majBlack"] = fnum(maj["store_count_5km"].mean())
    res["storecount_other"]    = fnum(oth["store_count_5km"].mean())

    # Correlations (Spearman, robust to skew)
    def spear(a,b):
        s = g[[a,b]].dropna()
        if len(s)<10: return None
        return fnum(s[a].corr(s[b], method="spearman"))
    res["rho_pctblack_nearest"]   = spear("pct_black","nearest_store_km")
    res["rho_pctblack_storecnt"]  = spear("pct_black","store_count_5km")
    res["rho_poverty_nearest"]    = spear("poverty_rate","nearest_store_km")
    res["rho_income_storecnt"]    = spear("median_income","store_count_5km")

    # Negative-binomial: store_count_5km ~ pct_black + poverty + log(pop) offset-ish
    try:
        m = g.dropna(subset=["store_count_5km","pct_black","poverty_rate","median_income","total_pop"]).copy()
        m["inc10k"] = m["median_income"]/10000
        model = smf.glm("store_count_5km ~ pct_black + poverty_rate + inc10k",
                        data=m, family=sm.families.NegativeBinomial()).fit()
        res["nb"] = {k: {"coef": fnum(model.params[k]), "p": fnum(model.pvalues[k]),
                         "irr": fnum(np.exp(model.params[k]))}
                     for k in model.params.index}
        res["nb_n"] = int(model.nobs)
    except Exception as e:
        res["nb_error"] = str(e)[:200]

    # Moran's I on underserved_index (spatial clustering), queen contiguity
    try:
        gg = g.dropna(subset=["underserved_index"]).reset_index(drop=True)
        w = Queen.from_dataframe(gg, use_index=False); w.transform="r"
        mi = Moran(gg["underserved_index"].values, w, permutations=999)
        res["moran_I"] = fnum(mi.I); res["moran_p"] = fnum(mi.p_sim)
        res["moran_n"] = int(len(gg))
    except Exception as e:
        res["moran_error"] = str(e)[:200]
    return res

out = {k: analyze(k) for k in ["nc","greensboro","durham"]}
with open("/tmp/claude-0/-home-user/9d31be27-e6e7-5d4e-8b89-647421178f48/scratchpad/nc_results.json","w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
