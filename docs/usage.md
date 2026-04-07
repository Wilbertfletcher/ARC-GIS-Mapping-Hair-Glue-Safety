# Usage Notes

## Workflow A — ArcGIS Pro / `arcpy`

### Inputs expected

1. **Stores**: either a CSV of store addresses or a point feature class.
2. **Low-income areas**: polygon feature class.
3. **Optional addresses**: point feature class for nearest-store analysis.
4. **Projected spatial reference**: local PCS or WKID suitable for distance work.

### Outputs created

- Annotated stores feature class with:
  - `IN_LOW_INCOME`
  - `NEAR_TO_LOW_INCOME`
  - `DIST_TO_LOWINC_MI`
- Qualifying stores feature class
- Low-income buffer feature class
- Optional address results with `NEAR_STORE_MI`
- Optional near table for address-to-store relationships
- CSV exports for reporting

## Workflow B — GeoPandas + Census API

### Required data inputs

- **Census tracts**: GeoJSON or shapefile with `GEOID`
- **Beauty supply store CSV** with columns:
  - `name,address,city,state,zip,lat,lon,category`
- **Optional Census API key** via `--census_api_key`, `CENSUS_API_KEY`, or a local `.env` file

### ACS variables used

- `B02001_001E` → total population
- `B02001_003E` → African American population
- `B19013_001E` → median household income
- `B17001_001E` and `B17001_002E` → poverty rate

### Outputs created

- `beauty_access_enriched.geojson`
- `beauty_access_summary.csv`
- `beauty_access_map.png`
- optional `beauty_access_map.html`

## Workflow C — Streamlit app

Run:

```powershell
streamlit run .\streamlit_app.py --server.port 8765
```

The app lets users enter a **city** and **state**, then queries live Census and OpenStreetMap data to build an interactive map and downloadable outputs.

## Recommended workflow

- Put source tables in `data/raw/` and tract layers in `data/reference/`.
- Save your Census key in `.env` as `CENSUS_API_KEY=...`.
- Write geodatabases to `outputs/gdb/`.
- Write GeoPandas outputs to a folder such as `outputs/beauty_access/` or `outputs/beauty_access_live/`.
- Use `data/raw/beauty_stores_template.csv` and `data/reference/sample_tracts.geojson` as starter examples.

