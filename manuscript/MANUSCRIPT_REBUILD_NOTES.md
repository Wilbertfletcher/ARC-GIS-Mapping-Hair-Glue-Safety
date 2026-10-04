# Manuscript rebuild — provenance & status

## Why it was rebuilt

The previous `fletcher_mulrooney_2026.tex` described a **different, fabricated study**: 8,347
poison-control/ED "incident records," Moran's I = 0.643, r = 0.721, 47
very-high-risk regions affecting 28.5M people, a Risk Stratification Index, and
nine placeholder citations ("Author, A. (2020)"). **None of those data or numbers
exist in this repository**, and it defined "ARC-GIS" incorrectly as "Advanced
Remote Computing GIS." That draft could not be submitted — it would be data
fabrication. It has been replaced.

## What the new manuscript reports

The real study this repo actually performs: a **census-tract geospatial analysis
of beauty-supply retail access in North Carolina**, framed as an
exposure-*opportunity* proxy for hair-adhesive products. Every number is computed
from this repo's own outputs by `manuscript/analysis/compute_nc_statistics.py`.

## Provenance of each figure (all recomputable)

Source outputs:
- Statewide: `outputs/streamlit_queries/nc_usa/` (2,649 tracts w/ pop > 0, 349 stores)
- Greensboro: `outputs/streamlit_queries/greensboro_nc_usa/` (114 tracts, 24 stores)
- Durham: `outputs/durham_beauty_supply_only/` (67 tracts, 8 beauty-supply-only stores)

Key verified numbers:
| Statistic | Value | How |
|---|---|---|
| NC tracts / population / stores | 2,649 / 10.47M / 349 | sums from enriched GeoJSON |
| % NC tracts with no store ≤5 km | 67.5% | `store_count_5km == 0` |
| Median nearest-store distance (NC) | 9.3 km (IQR 3.9–20.4) | `nearest_store_km` |
| Moran's I, nearest-store distance | 0.92 (p=0.001) | Queen contiguity, 999 perms |
| Moran's I, 5 km store count | 0.83 (p=0.001) | same |
| Moran's I, percent-Black | 0.66 (p=0.001) | same (for comparison) |
| Quartile gradient (Table 1) | 14.2→5.6 km Q1→Q4 | `pd.qcut(pct_black,4)` |
| Spearman ρ (pct_black, nearest) | −0.21 | rank corr |
| NB model IRR pct_black / poverty / income | 1.02 / 1.02 / 1.21 | statsmodels GLM, NegBin |

Re-run: `python manuscript/analysis/compute_nc_statistics.py` (needs
`geopandas libpysal esda statsmodels`).

## The headline finding (and why it's stated the way it is)

Access **improves** with tract percent-Black — higher-%Black tracts are closer to
and have more stores. This is the **opposite** of a naive "Black neighborhoods are
underserved" claim. The honest interpretation, written into the Discussion:
beauty-supply retail is a targeted-market sector concentrated in Black/urban
areas (greater exposure *opportunity*), with the real access gap being
**rural**. Do not let a reviewer or co-author "flip" this back to an
underservice narrative — the data don't support it.

## What still needs YOU before submission

1. **[CONFIRM] affiliations** — I put Fletcher & Mulrooney in NCCU Environmental,
   Earth & Geospatial Sciences and Schultz in NCCU Chemistry. Verify exact
   departments and whether Schultz/Mulrooney have approved co-authorship.
2. **[CONFIRM] corresponding email** — currently your `wfletch1@eagles.nccu.edu`.
3. **Store data** — the single biggest weakness. OpenStreetMap undercounts and
   varied run-to-run (Durham shows 6/8/40 stores across different runs). For a
   real submission, replace with a validated registry (NC business licensing,
   InfoUSA/Data Axle, or manual verification) and re-run the pipeline.
4. **ACS vintage** — confirm the pipeline actually queried 2022 (default is 2022);
   state the exact release.
5. **Figures** — the repo has `beauty_access_map.png` per city; add the statewide
   choropleth + store overlay and a quartile bar chart. Not yet embedded.
6. **Scale the case studies** if desired — Charlotte, Raleigh, Fayetteville all
   runnable with the existing pipeline.
7. **Citations** — the 8 references are real but need DOIs/page numbers filled and
   a couple more (OpenStreetMap, ACS methodology, a beauty-supply retail
   geography reference).

## Target journals
*Health & Place*; *International Journal of Health Geographics*; *Applied
Geography*; *Journal of Exposure Science & Environmental Epidemiology*.

## Files
- `manuscript/fletcher_mulrooney_2026.tex` — rebuilt draft (8 pp, compiles with pdfLaTeX)
- `manuscript/analysis/compute_nc_statistics.py` — reproducible statistics
- `manuscript/MANUSCRIPT_REBUILD_NOTES.md` — this file
