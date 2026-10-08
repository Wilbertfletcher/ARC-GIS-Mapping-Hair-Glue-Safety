# Reproducing the Fletcher / Mulrooney paper

This folder holds the complete code that regenerates every number, table and
figure in `manuscript/fletcher_mulrooney_2026.tex` from the data already in this
repository, plus a checker that confirms the paper's text matches the results.

## Quick start

```bash
cd paper_reproduction
pip install -r requirements.txt
python reproduce.py --sync-manuscript     # about 40 seconds
python verify_against_paper.py            # expect: 120 of 120 checks passed
```

`--sync-manuscript` copies the three figures into `../manuscript/figures/` so the
paper compiles with them. Without it, figures are written only to `results/figures/`.

Compile the paper from the `manuscript/` folder with `pdflatex` (two passes).

## What gets produced

| Paper item | File under `results/` |
|---|---|
| Figure 1, statewide access map | `figures/fig1_nc_choropleth.png` |
| Figure 2, access by percent-Black quartile | `figures/fig2_quartile_gradient.png` |
| Figure 3, five metros | `figures/fig3_metros.png` |
| Table 1, quartile table | `tables/table1_quartiles.csv`, `tables/table1_quartiles_rows.tex` |
| Table 2, metro table | `tables/table2_metros.csv`, `tables/table2_metros_rows.tex` |
| Population density quartiles (Results 3.3) | `tables/density_quartiles.csv` |
| Every in-text number (abstract, results, limitations) | `results.json` |
| The same numbers, labelled by paper section | `paper_numbers.md` |
| Package versions used | `environment.txt` |

The `*_rows.tex` files are the exact table rows printed in the paper. The checker
confirms the manuscript contains them unchanged.

## Inputs

`reproduce.py` reads two files that are already committed:

```
outputs/streamlit_queries/nc_usa/beauty_access_enriched.geojson   2,672 census tracts, ACS 2022 plus access metrics
outputs/streamlit_queries/nc_usa/beauty_supply_stores.geojson     349 OpenStreetMap store points
```

Use `--input-dir` to point at a different run of the pipeline.

These two files are the output of the project's pipeline
(`scripts/beauty_supply_access_pipeline.py`; the folder name indicates they were
written by the Streamlit app's North Carolina query). **They are a snapshot.** Re-running the pipeline queries live
OpenStreetMap and Census services, so a new run will return different store
counts and will not reproduce the paper's numbers exactly. The committed snapshot
is the reproducible input; the pipeline is how it was made.

## How each number is computed

All code is in `analysis.py`.

- **Tracts analysed.** Tracts with nonzero population: 2,649 of 2,672.
- **Access measures.** Distance from the tract centroid to the nearest store, and
  the number of stores within 5 km of the centroid. Both come from the pipeline.
- **Moran's I.** Queen contiguity weights, row standardised, 999 permutations,
  seed 42. With 999 permutations the smallest possible pseudo-p is 0.001, which is
  what the paper reports.
- **Quartiles.** `pandas.qcut` on tract percent Black (Table 1) and on population
  density (persons per km2, ranked, so ties cannot break the quartiles).
- **Negative binomial model.** Store count within 5 km on percent Black, poverty
  rate and median income per $10,000, using a statsmodels GLM with the dispersion
  fixed at 1 (the statsmodels default). Two sensitivity checks are also computed:
  dispersion estimated by maximum likelihood, and county-clustered standard errors.
  Both leave the direction and significance of all three terms unchanged.
- **Metros.** County subsets of the statewide layer: Charlotte (Mecklenburg, 119),
  Raleigh (Wake, 183), Greensboro (Guilford, 081), Durham (063), Fayetteville
  (Cumberland, 051). Moran's I is recomputed within each county.
- **Stores inside the state.** 183 of the 349 points are inside the state outline.
  The map also draws stores within about 8 km of the border (188 in total). The
  other 161 are farther out, and 156 of them fall inside North Carolina's bounding
  box, which indicates the store query used a bounding box and returned real stores
  in neighbouring states. Distances were computed against all 349.

## Files

```
paper_reproduction/
  reproduce.py                 entry point: compute, write results/, draw figures
  analysis.py                  all statistics (no plotting, no file writing)
  figures.py                   the three figures
  verify_against_paper.py      checks the manuscript text against results/
  requirements.txt
  results/                     generated outputs, committed so they can be inspected
```

## What the checker verifies

`verify_against_paper.py` runs 120 checks. For each number printed in the paper it
compares the value typed from the manuscript with the value computed here, and
confirms the number appears in the manuscript text as written. It also confirms
that Tables 1 and 2 in the manuscript equal the generated rows, and that the
manuscript contains no en or em dashes. It exits with status 1 if anything fails,
so it can be used in CI or as a pre-submission check.

## Known limitations of the analysis

- The store list is a volunteer-contributed OpenStreetMap extract and is incomplete.
  A validated business registry would be better. See the paper's Limitations.
- Distances are straight-line from tract centroids, not road or travel distances.
- The negative binomial p-values do not account for spatial autocorrelation, which
  is strong (Moran's I of 0.83 for store counts). The county-clustered sensitivity
  check is a partial answer; a spatial regression would be a fuller one.
- The composite underserved index built by the pipeline weights percent Black at
  0.40 by construction, so it is not used for any disparity result in the paper.
