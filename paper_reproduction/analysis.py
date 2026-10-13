"""Pure computation for the Fletcher / Mulrooney manuscript.

Every statistic, table value, and in-text number in the paper is produced by
`compute_all()`. Nothing here draws figures or writes files; see `figures.py`
and `reproduce.py` for that.

Inputs are the two files the pipeline wrote for the statewide North Carolina run:
  beauty_access_enriched.geojson   one row per census tract, ACS 2022 + access metrics
  beauty_supply_stores.geojson     the OpenStreetMap store extract (points)
"""
from __future__ import annotations

import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from esda.moran import Moran
from libpysal.weights import Queen

warnings.filterwarnings("ignore")

# County FIPS (state 37) -> metro label used in the paper. These are county
# subsets of the statewide layer, so every metro shares one store set.
METROS = {
    "119": "Charlotte",      # Mecklenburg
    "183": "Raleigh",        # Wake
    "081": "Greensboro",     # Guilford
    "063": "Durham",         # Durham
    "051": "Fayetteville",   # Cumberland
}

NUMERIC_COLS = [
    "total_pop", "black_pop", "median_income", "poverty_rate", "pct_black",
    "nearest_store_km", "store_count_5km", "underserved_index",
]

# Stores are mapped if they fall within this many degrees (about 8 km) of the
# state, so that stores just over the border that serve edge tracts stay visible.
STATE_BUFFER_DEG = 0.08


def load_inputs(input_dir):
    """Read tracts and stores; return (tracts with population > 0, all tracts, stores)."""
    tracts_all = gpd.read_file(f"{input_dir}/beauty_access_enriched.geojson")
    stores = gpd.read_file(f"{input_dir}/beauty_supply_stores.geojson")
    tracts_all["COUNTYFP"] = tracts_all["COUNTYFP"].astype(str).str.zfill(3)
    for col in NUMERIC_COLS:
        tracts_all[col] = pd.to_numeric(tracts_all[col], errors="coerce")
    tracts = tracts_all[tracts_all["total_pop"] > 0].copy().reset_index(drop=True)
    return tracts, tracts_all, stores


def state_outline(tracts):
    geom = tracts.to_crs(4326).geometry
    return geom.union_all() if hasattr(geom, "union_all") else geom.unary_union


def stores_in_state(stores, tracts):
    """Split stores into (inside the state outline, mapped incl. 8 km buffer)."""
    outline = state_outline(tracts)
    pts = stores.to_crs(4326)
    strict = stores[pts.within(outline)]
    mapped = stores[pts.within(outline.buffer(STATE_BUFFER_DEG))]
    return strict, mapped


def stores_outside_in_state_bbox(stores, tracts, mapped):
    """How many of the out-of-state points still fall inside NC's bounding box.

    A high share means the store query used a bounding box, so the extra points
    are real stores in neighbouring states (e.g. Greenville SC), not bad coordinates.
    """
    outside = stores.drop(mapped.index).to_crs(4326).geometry
    west, south, east, north = tracts.to_crs(4326).total_bounds
    return int(((outside.x >= west) & (outside.x <= east) &
                (outside.y >= south) & (outside.y <= north)).sum())


def _moran(frame, col, permutations, seed):
    sub = frame.dropna(subset=[col]).reset_index(drop=True)
    if len(sub) < 15:
        return {"I": None, "p": None, "n": int(len(sub))}
    w = Queen.from_dataframe(sub, use_index=False)
    w.transform = "r"
    np.random.seed(seed)
    m = Moran(sub[col].values, w, permutations=permutations)
    return {"I": float(m.I), "p": float(m.p_sim), "n": int(len(sub))}


def _round(x, nd):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


def quartile_table(tracts):
    q = pd.qcut(tracts["pct_black"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    rows = []
    for label, sub in tracts.groupby(q, observed=True):
        rows.append({
            "quartile": str(label),
            "tracts": int(len(sub)),
            "pct_black_min": _round(sub["pct_black"].min(), 1),
            "pct_black_max": _round(sub["pct_black"].max(), 1),
            "median_nearest_km": _round(sub["nearest_store_km"].median(), 1),
            "mean_stores_5km": _round(sub["store_count_5km"].mean(), 2),
            "pct_no_store_5km": _round(100 * (sub["store_count_5km"] == 0).mean(), 1),
        })
    return rows


def density_quartiles(tracts):
    """Access by population-density quartile (persons per km2), the urban/rural check."""
    dens = tracts["total_pop"] / (pd.to_numeric(tracts["ALAND"], errors="coerce") / 1e6)
    q = pd.qcut(dens.rank(method="first"), 4,
                labels=["lowest density", "Q2", "Q3", "highest density"])
    rows = []
    for label, sub in tracts.groupby(q, observed=True):
        d = dens.loc[sub.index]
        rows.append({
            "quartile": str(label), "tracts": int(len(sub)),
            "median_density_per_km2": _round(d.median(), 0),
            "median_nearest_km": _round(sub["nearest_store_km"].median(), 1),
            "pct_no_store_5km": _round(100 * (sub["store_count_5km"] == 0).mean(), 1),
        })
    return rows


def negative_binomial(tracts, cluster_by_county=False):
    """Store count within 5 km on percent-Black, poverty, income (per $10k).

    The paper's model is a negative-binomial GLM with the dispersion parameter
    fixed at statsmodels' default (alpha = 1). `cluster_by_county` and the
    estimated-dispersion fit below are sensitivity checks, not paper values.
    """
    m = tracts.dropna(subset=["store_count_5km", "pct_black", "poverty_rate",
                              "median_income", "total_pop"]).copy()
    m["inc10k"] = m["median_income"] / 10000.0
    formula = "store_count_5km ~ pct_black + poverty_rate + inc10k"

    fit = smf.glm(formula, data=m, family=sm.families.NegativeBinomial()).fit()
    out = {"n": int(fit.nobs), "dispersion_alpha": 1.0, "terms": {}}
    for k in fit.params.index:
        ci = fit.conf_int().loc[k]
        out["terms"][k] = {
            "coef": float(fit.params[k]), "p": float(fit.pvalues[k]),
            "irr": float(np.exp(fit.params[k])),
            "irr_lo": float(np.exp(ci[0])), "irr_hi": float(np.exp(ci[1])),
        }

    # Sensitivity 1: dispersion estimated by maximum likelihood (NB2).
    try:
        nb2 = smf.negativebinomial(formula, data=m).fit(disp=0, maxiter=200)
        out["sensitivity_estimated_alpha"] = {
            "alpha": float(nb2.params["alpha"]),
            "terms": {k: {"irr": float(np.exp(nb2.params[k])), "p": float(nb2.pvalues[k])}
                      for k in ["pct_black", "poverty_rate", "inc10k"]},
        }
    except Exception as exc:  # pragma: no cover
        out["sensitivity_estimated_alpha"] = {"error": str(exc)[:160]}

    # Sensitivity 2: county-clustered standard errors (spatial dependence).
    if cluster_by_county:
        try:
            cl = smf.glm(formula, data=m, family=sm.families.NegativeBinomial()).fit(
                cov_type="cluster", cov_kwds={"groups": pd.factorize(m["COUNTYFP"])[0]})
            out["sensitivity_county_clustered_se"] = {
                "terms": {k: {"irr": float(np.exp(cl.params[k])), "p": float(cl.pvalues[k])}
                          for k in ["pct_black", "poverty_rate", "inc10k"]}}
        except Exception as exc:  # pragma: no cover
            out["sensitivity_county_clustered_se"] = {"error": str(exc)[:160]}
    return out


def metro_table(tracts, permutations, seed):
    rows = []
    for fips, name in METROS.items():
        sub = tracts[tracts["COUNTYFP"] == fips].reset_index(drop=True)
        maj, oth = sub[sub["pct_black"] >= 50], sub[sub["pct_black"] < 50]
        mor = _moran(sub, "nearest_store_km", permutations, seed)
        rows.append({
            "metro": name, "county_fips": fips,
            "tracts": int(len(sub)), "population": int(sub["total_pop"].sum()),
            "median_pct_black": _round(sub["pct_black"].median(), 1),
            "median_nearest_km": _round(sub["nearest_store_km"].median(), 1),
            "pct_no_store_5km": _round(100 * (sub["store_count_5km"] == 0).mean(), 1),
            "mean_stores_5km": _round(sub["store_count_5km"].mean(), 2),
            "majority_black_tracts": int(len(maj)),
            "nearest_km_majority_black": _round(maj["nearest_store_km"].median(), 1) if len(maj) else None,
            "nearest_km_other": _round(oth["nearest_store_km"].median(), 1) if len(oth) else None,
            "moran_I_distance": _round(mor["I"], 2), "moran_p_distance": _round(mor["p"], 3),
        })
    return rows


def compute_all(input_dir, permutations=999, seed=42, sensitivity=True):
    tracts, tracts_all, stores = load_inputs(input_dir)
    strict, mapped = stores_in_state(stores, tracts)

    nearest = tracts["nearest_store_km"]
    state = {
        "tracts_in_file": int(len(tracts_all)),
        "zero_population_tracts_excluded": int(len(tracts_all) - len(tracts)),
        "tracts": int(len(tracts)),
        "population": int(tracts["total_pop"].sum()),
        "median_pct_black": _round(tracts["pct_black"].median(), 1),
        "stores_in_extract": int(len(stores)),
        "stores_inside_state": int(len(strict)),
        "stores_mapped_within_8km_of_state": int(len(mapped)),
        "stores_outside_state": int(len(stores) - len(mapped)),
        "stores_outside_but_inside_state_bbox": stores_outside_in_state_bbox(stores, tracts, mapped),
        "tracts_no_store_5km": int((tracts["store_count_5km"] == 0).sum()),
        "pct_no_store_5km": _round(100 * (tracts["store_count_5km"] == 0).mean(), 1),
        "mean_stores_5km": _round(tracts["store_count_5km"].mean(), 2),
        "nearest_km_median": _round(nearest.median(), 1),
        "nearest_km_q1": _round(nearest.quantile(0.25), 1),
        "nearest_km_q3": _round(nearest.quantile(0.75), 1),
    }

    moran = {col: _moran(tracts, col, permutations, seed)
             for col in ["nearest_store_km", "store_count_5km", "pct_black"]}

    def spearman(a, b):
        s = tracts[[a, b]].dropna()
        return {"rho": float(s[a].corr(s[b], method="spearman")), "n": int(len(s))}

    spear = {
        "pct_black_vs_nearest_km": spearman("pct_black", "nearest_store_km"),
        "pct_black_vs_store_count_5km": spearman("pct_black", "store_count_5km"),
    }

    metros = metro_table(tracts, permutations, seed)
    moran_vals = [m["moran_I_distance"] for m in metros if m["moran_I_distance"] is not None]

    return {
        "statewide": state,
        "moran": moran,
        "spearman": spear,
        "quartiles": quartile_table(tracts),
        "density_quartiles": density_quartiles(tracts),
        "negative_binomial": negative_binomial(tracts, cluster_by_county=sensitivity),
        "metros": metros,
        "metro_moran_range": [min(moran_vals), max(moran_vals)],
        "settings": {"permutations": permutations, "seed": seed,
                     "weights": "queen contiguity, row standardised",
                     "state_buffer_deg": STATE_BUFFER_DEG},
    }, tracts, strict, mapped
