# ARC GIS Mapping Hair Glue Safety

Geospatial analysis workspace for the hair glue safety study, including both an **ArcGIS Pro / `arcpy` workflow** and a **GeoPandas + Census API workflow** for beauty supply access analysis.

## Project structure

```text
ARC GIS Mapping Hair Glue Safety/
├── data/
│   ├── raw/                        # store CSVs and other input tables
│   │   └── beauty_stores_template.csv
│   └── reference/                  # tract layers, locators, polygons
│       └── sample_tracts.geojson
├── docs/
│   └── usage.md
├── outputs/
│   ├── csv/
│   └── gdb/
├── scripts/
│   ├── beauty_supply_access_pipeline.py
│   └── stores_low_income_distance.py
├── requirements.txt
├── LICENSE
└── README.md
```

## Available workflows

### 1) `scripts/stores_low_income_distance.py`
ArcGIS Pro / `arcpy` automation for:
- identifying stores **inside** low-income polygons,
- flagging stores **near** low-income areas,
- computing **planar** distances, and
- optionally linking address points to the **nearest qualifying store**.

### 2) `scripts/beauty_supply_access_pipeline.py`
GeoPandas workflow for:
- loading census tracts from GeoJSON or shapefiles,
- pulling ACS demographics from the **U.S. Census API**,
- calculating `% Black`, poverty, and low-income flags,
- measuring nearest-store distance and stores within a **5 km** buffer,
- creating a **beauty access score** and **underserved index**,
- clustering hotspots with **DBSCAN**, and
- exporting **GeoJSON**, **CSV**, and map outputs.

### 3) `streamlit_app.py`
Interactive Streamlit dashboard for entering a **city** and **state** and generating a live map of:
- Census tract demographics,
- filtered **beauty supply only** stores,
- underserved tracts, and
- downloadable CSV / GeoJSON outputs.

## Example runs

### ArcGIS Pro script
Run this inside the **ArcGIS Pro Python environment**:

```powershell
python .\scripts\stores_low_income_distance.py ^
  --workspace ".\outputs\gdb\hair_glue_analysis.gdb" ^
  --target_sr "NAD 1983 (2011) StatePlane Texas South Central FIPS 4204 (US Feet)" ^
  --stores "C:/GIS/data/stores.csv" ^
  --locator "C:/GIS/locators/EnterpriseLocator.loc" ^
  --address_fields "Address;City;State;ZIP" ^
  --low_income_fc "C:/GIS/data/low_income.gdb/LowIncomeBlockGroups" ^
  --near_threshold "1 Miles" ^
  --output_prefix "Humble" ^
  --addresses_fc "C:/GIS/data/addresses.gdb/ServiceAddresses"
```

### GeoPandas + Census pipeline
This can run in the local project `.venv`.
Store your Census key in `.env` as `CENSUS_API_KEY=...`, then run:

```powershell
python .\scripts\beauty_supply_access_pipeline.py ^
  --tracts ".\data\reference\sample_tracts.geojson" ^
  --stores ".\data\raw\beauty_stores_template.csv" ^
  --state "NC" ^
  --output_dir ".\outputs\beauty_access_live" ^
  --buffer_km 5 ^
  --low_income_threshold 40000 ^
  --create_interactive_map
```

## Streamlit app

Run the interactive app with:

```powershell
streamlit run .\streamlit_app.py --server.port 8765
```

Then open `http://localhost:8765` in your browser.

## Notes

- The Durham store CSVs have been narrowed to **beauty supply retail only** by excluding salons, spas, nails, waxing, and similar service businesses.
- If `--stores` is already a point feature class, omit `--locator` and `--address_fields` in the ArcGIS script.
- For accurate **planar** distances, use a projected CRS appropriate to your study area.
- The requested packages were installed successfully in the project environment: `geopandas`, `pandas`, `shapely`, `geopy`, `scikit-learn`, `matplotlib`, `contextily`, `census`, `us`, `folium`, `numpy`, `requests`, and `streamlit`.
- The separate `arcgis` pip package did **not** install in this Python 3.14 environment; use **ArcGIS Pro / `arcpy`** for the ArcGIS-specific workflow.

