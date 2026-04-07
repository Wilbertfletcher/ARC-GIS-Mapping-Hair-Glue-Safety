# -*- coding: utf-8 -*-
"""
Beauty supply access analysis pipeline for the Hair Glue project.

This script:
1. loads tract boundaries (GeoJSON/shapefile),
2. optionally pulls ACS demographics from the U.S. Census API,
3. loads beauty supply store points from CSV,
4. computes planar access metrics,
5. identifies underserved hotspots with DBSCAN, and
6. exports GeoJSON, CSV, and map outputs.

Example:
    python scripts/beauty_supply_access_pipeline.py ^
      --tracts data/reference/sample_tracts.geojson ^
      --stores data/raw/beauty_stores_template.csv ^
      --state NC ^
      --output_dir outputs/beauty_access ^
      --buffer_km 5 ^
      --low_income_threshold 40000 ^
      --skip_census_api ^
      --create_interactive_map
"""

from __future__ import annotations

import argparse
import os
import re
import zipfile
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import us
from shapely.geometry import box
from sklearn.cluster import DBSCAN
from sklearn.neighbors import BallTree

try:
    import contextily as ctx
except Exception:  # pragma: no cover
    ctx = None

try:
    import folium
except Exception:  # pragma: no cover
    folium = None

ACS_FIELDS = {
    "B02001_001E": "total_pop",
    "B02001_003E": "black_pop",
    "B19013_001E": "median_income",
    "B17001_001E": "poverty_base",
    "B17001_002E": "poverty_below",
}

REQUEST_HEADERS = {
    "User-Agent": "GitHub Copilot (Preview) thesis geospatial analysis"
}

SUPPLY_NAME_PATTERN = re.compile(
    r"beauty world|beauty supply|beauty outlet|sally beauty|cosmoprof|"
    r"ulta|merle norman|bath\s*&\s*body|hair supply|cosmetic|perfumery|wig",
    re.IGNORECASE,
)
SERVICE_EXCLUDE_PATTERN = re.compile(
    r"nail|spa|salon|lash|wax|barber|brow|massage|facial|academy|medical",
    re.IGNORECASE,
)


def load_local_env() -> str:
    """Load `CENSUS_API_KEY` from a project `.env` file if present."""
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.exists():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(
                key.strip(),
                value.strip().strip('"').strip("'"),
            )
    return os.getenv("CENSUS_API_KEY", "")


DEFAULT_CENSUS_API_KEY = load_local_env()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze beauty supply access and underserved areas."
    )
    parser.add_argument(
        "--tracts",
        required=True,
        help="Path to tract boundaries (GeoJSON, shapefile, or geopackage).",
    )
    parser.add_argument(
        "--stores",
        required=True,
        help="CSV with store name/address/lat/lon columns.",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/beauty_access",
        help="Folder for GeoJSON, CSV, and map outputs.",
    )
    parser.add_argument(
        "--state",
        default="",
        help="State abbreviation, name, or FIPS for Census API pulls.",
    )
    parser.add_argument(
        "--census_api_key",
        default=DEFAULT_CENSUS_API_KEY,
        help="Optional Census API key. Can also be set in CENSUS_API_KEY or `.env`.",
    )
    parser.add_argument(
        "--acs_year",
        type=int,
        default=2022,
        help="ACS 5-year dataset year to query from the Census API.",
    )
    parser.add_argument(
        "--buffer_km",
        type=float,
        default=5.0,
        help="Radius in kilometers for counting nearby stores.",
    )
    parser.add_argument(
        "--low_income_threshold",
        type=float,
        default=40000.0,
        help="Median income threshold used to flag low-income tracts.",
    )
    parser.add_argument(
        "--metric_crs",
        default="auto",
        help="Projected CRS for planar distance work (for example EPSG:26917).",
    )
    parser.add_argument(
        "--category_filter",
        default="",
        help="Optional text filter for the store category column.",
    )
    parser.add_argument(
        "--include_wig_shops",
        action="store_true",
        help="Include wig shops in addition to beauty supply retailers.",
    )
    parser.add_argument(
        "--skip_census_api",
        action="store_true",
        help="Use demographic fields already present in the tract file.",
    )
    parser.add_argument(
        "--create_interactive_map",
        action="store_true",
        help="Also create an interactive Folium HTML map.",
    )
    return parser.parse_args()


def log(message: str) -> None:
    print(message)


def ensure_output_dir(path: str | Path) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_tracts(path: str) -> gpd.GeoDataFrame:
    tracts = gpd.read_file(path)
    if tracts.empty:
        raise ValueError("The tract layer is empty.")
    if tracts.crs is None:
        log("No CRS found on the tract layer. Defaulting to EPSG:4326.")
        tracts = tracts.set_crs(epsg=4326)
    return standardize_geoid(tracts)


def standardize_geoid(tracts: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    tracts = tracts.copy()
    for col in ["GEOID", "geoid", "GEOID10", "geoid10"]:
        if col in tracts.columns:
            tracts["GEOID"] = (
                tracts[col]
                .astype(str)
                .str.replace("1400000US", "", regex=False)
                .str.replace(r"\.0$", "", regex=True)
                .str.zfill(11)
            )
            return tracts

    required = {"STATEFP", "COUNTYFP", "TRACTCE"}
    if required.issubset(tracts.columns):
        tracts["GEOID"] = (
            tracts["STATEFP"].astype(str).str.zfill(2)
            + tracts["COUNTYFP"].astype(str).str.zfill(3)
            + tracts["TRACTCE"].astype(str).str.replace(".", "", regex=False)
            .str.zfill(6)
        )
        return tracts

    raise ValueError(
        "Could not build GEOID. Supply a tract file with GEOID or STATEFP/COUNTYFP/TRACTCE."
    )


def resolve_state_fips(state_value: str) -> str:
    state_value = (state_value or "").strip()
    if not state_value:
        raise ValueError("A state value is required when pulling from the Census API.")
    if state_value.isdigit():
        return state_value.zfill(2)

    state_obj = us.states.lookup(state_value)
    if state_obj is None:
        raise ValueError(f"Could not resolve state from: {state_value}")
    return str(state_obj.fips).zfill(2)


def slugify(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip().lower())
    return value.strip("_") or "query"


def fetch_place_record(place_query: str) -> dict:
    response = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": place_query, "format": "jsonv2", "limit": 1},
        headers=REQUEST_HEADERS,
        timeout=60,
    )
    response.raise_for_status()
    results = response.json()
    if not results:
        raise ValueError(f"Could not resolve a place for query: {place_query}")
    return results[0]


def fetch_state_tracts(
    state_value: str,
    year: int = 2022,
    cache_dir: str | Path = "outputs/cache",
) -> gpd.GeoDataFrame:
    state_fips = resolve_state_fips(state_value)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    zip_name = f"tl_{year}_{state_fips}_tract.zip"
    extract_dir = cache_dir / zip_name.replace(".zip", "")
    zip_path = cache_dir / zip_name
    shp_pattern = f"tl_{year}_{state_fips}_tract.shp"
    shp_path = extract_dir / shp_pattern

    if not shp_path.exists():
        url = f"https://www2.census.gov/geo/tiger/TIGER{year}/TRACT/{zip_name}"
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=120)
        response.raise_for_status()
        zip_path.write_bytes(response.content)
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(extract_dir)

    tracts = gpd.read_file(shp_path)
    if tracts.crs is None:
        tracts = tracts.set_crs(epsg=4269)
    return standardize_geoid(tracts)


def subset_tracts_to_bbox(
    tracts: gpd.GeoDataFrame,
    west: float,
    south: float,
    east: float,
    north: float,
) -> gpd.GeoDataFrame:
    tracts_ll = tracts.to_crs(epsg=4326)
    bbox_geom = box(float(west), float(south), float(east), float(north))
    subset = tracts_ll[tracts_ll.geometry.intersects(bbox_geom)].copy()
    if subset.empty:
        return tracts_ll.copy()
    return subset


def filter_beauty_supply_locations(
    stores: pd.DataFrame,
    include_wig_shops: bool = False,
) -> pd.DataFrame:
    stores = stores.copy()
    stores["name"] = stores.get("name", "").fillna("")
    if "category" not in stores.columns:
        stores["category"] = ""

    name_series = stores["name"].astype(str)
    category_series = stores["category"].astype(str)

    include_mask = category_series.str.contains(
        r"cosmetics|perfumery|hairdresser_supply",
        case=False,
        na=False,
    )
    if include_wig_shops:
        include_mask = include_mask | category_series.str.contains(
            "wig",
            case=False,
            na=False,
        )

    include_mask = include_mask | name_series.str.contains(
        SUPPLY_NAME_PATTERN,
        na=False,
    )
    exclude_mask = name_series.str.contains(SERVICE_EXCLUDE_PATTERN, na=False)

    filtered = stores[include_mask & ~exclude_mask].copy()
    if filtered.empty:
        filtered = stores.copy()

    return filtered.drop_duplicates(subset=["name", "lat", "lon"]).reset_index(
        drop=True
    )


def fetch_osm_beauty_supply_stores(
    city: str = "",
    state_value: str = "",
    include_wig_shops: bool = False,
) -> pd.DataFrame:
    place_parts = [
        part.strip()
        for part in [city, state_value, "USA"]
        if part and part.strip()
    ]
    place_query = ", ".join(place_parts)
    place = fetch_place_record(place_query)
    south, north, west, east = place["boundingbox"]

    base_shop_tags = ["cosmetics", "perfumery", "hairdresser_supply"]
    if include_wig_shops:
        base_shop_tags.append("wig")
    retail_pattern = "|".join(base_shop_tags)
    beauty_name_pattern = "Beauty|Supply|Cosmo|Sally|Ulta|Merle|Outlet|Wig"

    overpass_query = f"""
    [out:json][timeout:90];
    (
      node[\"shop\"~\"{retail_pattern}\"]({south},{west},{north},{east});
      way[\"shop\"~\"{retail_pattern}\"]({south},{west},{north},{east});
      relation[\"shop\"~\"{retail_pattern}\"]({south},{west},{north},{east});
      node[\"shop\"=\"beauty\"][\"name\"~\"{beauty_name_pattern}\",i]({south},{west},{north},{east});
      way[\"shop\"=\"beauty\"][\"name\"~\"{beauty_name_pattern}\",i]({south},{west},{north},{east});
      relation[\"shop\"=\"beauty\"][\"name\"~\"{beauty_name_pattern}\",i]({south},{west},{north},{east});
    );
    out center tags;
    """

    elements = []
    errors = []
    overpass_endpoints = [
        "https://overpass-api.de/api/interpreter",
        "https://lz4.overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
    ]

    for endpoint in overpass_endpoints:
        try:
            response = requests.get(
                endpoint,
                params={"data": overpass_query},
                headers=REQUEST_HEADERS,
                timeout=120,
            )
            response.raise_for_status()
            elements = response.json().get("elements", [])
            if elements:
                break
        except requests.RequestException as exc:
            errors.append(f"{endpoint}: {exc}")

    rows = []
    for element in elements:
        tags = element.get("tags", {})
        lat = element.get("lat", element.get("center", {}).get("lat"))
        lon = element.get("lon", element.get("center", {}).get("lon"))
        if lat is None or lon is None:
            continue
        address = " ".join(
            piece
            for piece in [
                tags.get("addr:housenumber"),
                tags.get("addr:street"),
            ]
            if piece
        )
        rows.append(
            {
                "name": (
                    tags.get("name")
                    or tags.get("brand")
                    or "Unnamed beauty store"
                ),
                "address": address,
                "city": tags.get("addr:city", city or ""),
                "state": tags.get("addr:state", state_value or ""),
                "zip": tags.get("addr:postcode", ""),
                "lat": float(lat),
                "lon": float(lon),
                "category": (
                    tags.get("shop")
                    or tags.get("beauty")
                    or "beauty_supply"
                ),
            }
        )

    stores = pd.DataFrame(rows)
    if stores.empty:
        error_text = "; ".join(errors) if errors else "No rows returned"
        raise ValueError(
            f"No beauty supply store records were returned for {place_query}. "
            f"Details: {error_text}"
        )
    return filter_beauty_supply_locations(
        stores,
        include_wig_shops=include_wig_shops,
    )


def fetch_census_demographics(
    state_value: str,
    acs_year: int,
    census_api_key: str = "",
) -> pd.DataFrame:
    state_fips = resolve_state_fips(state_value)
    fields = ",".join(["NAME", *ACS_FIELDS.keys()])
    url = f"https://api.census.gov/data/{acs_year}/acs/acs5"
    params = {
        "get": fields,
        "for": "tract:*",
        "in": f"state:{state_fips} county:*",
    }
    if census_api_key:
        params["key"] = census_api_key

    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    data = response.json()

    frame = pd.DataFrame(data[1:], columns=data[0])
    frame["GEOID"] = (
        frame["state"].astype(str).str.zfill(2)
        + frame["county"].astype(str).str.zfill(3)
        + frame["tract"].astype(str).str.zfill(6)
    )

    for src_col, dst_col in ACS_FIELDS.items():
        frame[dst_col] = pd.to_numeric(frame[src_col], errors="coerce")

    frame["median_income"] = frame["median_income"].where(
        frame["median_income"] >= 0,
        np.nan,
    )
    return frame[["GEOID", "NAME", *ACS_FIELDS.values()]].copy()


def attach_demographics(
    tracts: gpd.GeoDataFrame,
    state_value: str,
    acs_year: int,
    census_api_key: str,
    skip_census_api: bool,
) -> gpd.GeoDataFrame:
    tracts = tracts.copy()

    rename_map = {
        "B02001_001E": "total_pop",
        "B02001_003E": "black_pop",
        "B19013_001E": "median_income",
        "B17001_001E": "poverty_base",
        "B17001_002E": "poverty_below",
    }
    for old, new in rename_map.items():
        if old in tracts.columns and new not in tracts.columns:
            tracts[new] = pd.to_numeric(tracts[old], errors="coerce")

    if not skip_census_api:
        census_frame = fetch_census_demographics(
            state_value=state_value,
            acs_year=acs_year,
            census_api_key=census_api_key,
        )
        tracts = tracts.drop(
            columns=[
                col
                for col in [
                    "NAME",
                    "total_pop",
                    "black_pop",
                    "median_income",
                    "poverty_base",
                    "poverty_below",
                ]
                if col in tracts.columns
            ],
            errors="ignore",
        ).merge(census_frame, on="GEOID", how="left")

    needed = {"total_pop", "black_pop", "median_income"}
    if not needed.issubset(tracts.columns):
        raise ValueError(
            "Missing demographic fields. Provide --state for Census API access or "
            "include total_pop, black_pop, and median_income in the tract file."
        )

    tracts["total_pop"] = pd.to_numeric(tracts["total_pop"], errors="coerce")
    tracts["black_pop"] = pd.to_numeric(tracts["black_pop"], errors="coerce")
    tracts["median_income"] = pd.to_numeric(
        tracts["median_income"],
        errors="coerce",
    )

    if {"poverty_base", "poverty_below"}.issubset(tracts.columns):
        tracts["poverty_base"] = pd.to_numeric(
            tracts["poverty_base"],
            errors="coerce",
        )
        tracts["poverty_below"] = pd.to_numeric(
            tracts["poverty_below"],
            errors="coerce",
        )
        tracts["poverty_rate"] = np.where(
            tracts["poverty_base"] > 0,
            (tracts["poverty_below"] / tracts["poverty_base"]) * 100,
            np.nan,
        )
    elif "poverty_rate" not in tracts.columns:
        tracts["poverty_rate"] = np.nan

    tracts["pct_black"] = np.where(
        tracts["total_pop"] > 0,
        (tracts["black_pop"] / tracts["total_pop"]) * 100,
        0,
    )
    return tracts


def load_stores(
    csv_path: str,
    category_filter: str = "",
    include_wig_shops: bool = False,
) -> gpd.GeoDataFrame:
    stores = pd.read_csv(csv_path)
    required = {"name", "lat", "lon"}
    missing = required.difference(stores.columns)
    if missing:
        raise ValueError(f"Store CSV is missing required columns: {sorted(missing)}")

    stores = stores.dropna(subset=["lat", "lon"]).copy()
    stores = filter_beauty_supply_locations(
        stores,
        include_wig_shops=include_wig_shops,
    )

    if category_filter and "category" in stores.columns:
        stores = stores[
            stores["category"].astype(str).str.contains(
                category_filter,
                case=False,
                na=False,
            )
        ].copy()

    if stores.empty:
        raise ValueError("No store rows remain after filtering.")

    stores["store_id"] = np.arange(1, len(stores) + 1)
    stores_gdf = gpd.GeoDataFrame(
        stores,
        geometry=gpd.points_from_xy(stores["lon"], stores["lat"]),
        crs="EPSG:4326",
    )
    return stores_gdf


def choose_metric_crs(
    tracts: gpd.GeoDataFrame,
    metric_crs: str,
):
    if str(metric_crs).lower() != "auto":
        return metric_crs
    try:
        return tracts.estimate_utm_crs()
    except Exception:
        return "EPSG:3857"


def minmax_scale(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").replace(
        [np.inf, -np.inf],
        np.nan,
    )
    values = values.fillna(values.median() if not values.dropna().empty else 0)
    min_value = float(values.min())
    max_value = float(values.max())
    if np.isclose(min_value, max_value):
        return pd.Series(0.0, index=series.index)
    return (values - min_value) / (max_value - min_value)


def merge_geometries(geometries):
    if hasattr(geometries, "union_all"):
        return geometries.union_all()
    return geometries.unary_union


def compute_access_metrics(
    tracts: gpd.GeoDataFrame,
    stores: gpd.GeoDataFrame,
    buffer_km: float,
    low_income_threshold: float,
) -> gpd.GeoDataFrame:
    tracts = tracts.copy()
    radius_m = float(buffer_km) * 1000.0

    centroids = tracts.geometry.centroid
    tract_coords = np.column_stack([centroids.x.to_numpy(), centroids.y.to_numpy()])
    store_coords = np.column_stack(
        [stores.geometry.x.to_numpy(), stores.geometry.y.to_numpy()]
    )

    tree = BallTree(store_coords, metric="euclidean")
    dist, _ = tree.query(tract_coords, k=1)
    tracts["nearest_store_m"] = dist[:, 0]
    tracts["nearest_store_km"] = dist[:, 0] / 1000.0

    centroid_buffers = gpd.GeoDataFrame(
        {"GEOID": tracts["GEOID"]},
        geometry=centroids.buffer(radius_m),
        crs=tracts.crs,
    )
    joined = gpd.sjoin(
        stores[["store_id", "geometry"]],
        centroid_buffers,
        how="left",
        predicate="within",
    )
    counts = joined.groupby("GEOID").size()
    tracts["store_count_5km"] = (
        tracts["GEOID"].map(counts).fillna(0).astype(int)
    )

    service_area = merge_geometries(stores.buffer(radius_m))
    covered_area = tracts.geometry.intersection(service_area).area
    tract_area = tracts.geometry.area.replace(0, np.nan)
    tracts["service_area_pct"] = (
        (covered_area / tract_area) * 100
    ).fillna(0).clip(0, 100)
    tracts["service_area_covered"] = tracts["service_area_pct"] > 0

    tracts["low_income"] = tracts["median_income"] < low_income_threshold
    tracts["beauty_access_score"] = np.where(
        tracts["total_pop"] > 0,
        (tracts["store_count_5km"] / tracts["total_pop"]) * 1000.0,
        0,
    )

    pct_black_norm = minmax_scale(tracts["pct_black"])
    poverty_norm = minmax_scale(tracts["poverty_rate"].fillna(0))
    access_norm = minmax_scale(tracts["beauty_access_score"])
    distance_norm = minmax_scale(tracts["nearest_store_km"])
    low_access_norm = (1 - access_norm + distance_norm) / 2

    tracts["underserved_index"] = (
        0.40 * pct_black_norm
        + 0.25 * poverty_norm
        + 0.20 * low_access_norm
        + 0.15 * tracts["low_income"].astype(int)
    ).round(4)

    return tracts


def cluster_hotspots(
    tracts: gpd.GeoDataFrame,
    buffer_km: float,
    quantile: float = 0.75,
    min_samples: int = 3,
) -> gpd.GeoDataFrame:
    tracts = tracts.copy()
    tracts["hotspot_cluster"] = -1
    threshold = tracts["underserved_index"].quantile(quantile)
    high_need = tracts[tracts["underserved_index"] >= threshold].copy()

    if len(high_need) < min_samples:
        tracts["hotspot_flag"] = False
        return tracts

    coords = np.column_stack(
        [high_need.geometry.centroid.x.to_numpy(), high_need.geometry.centroid.y.to_numpy()]
    )
    labels = DBSCAN(eps=buffer_km * 1000.0, min_samples=min_samples).fit(coords).labels_
    tracts.loc[high_need.index, "hotspot_cluster"] = labels
    tracts["hotspot_flag"] = tracts["hotspot_cluster"] >= 0
    return tracts


def create_static_map(
    tracts: gpd.GeoDataFrame,
    stores: gpd.GeoDataFrame,
    png_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 9))
    tracts.plot(
        column="pct_black",
        cmap="OrRd",
        linewidth=0.4,
        edgecolor="white",
        legend=True,
        ax=ax,
        alpha=0.75,
    )

    high_need = tracts[tracts["underserved_index"] >= tracts["underserved_index"].quantile(0.75)]
    if not high_need.empty:
        high_need.boundary.plot(ax=ax, color="navy", linewidth=1.2, label="High underserved")

    stores.plot(ax=ax, color="black", markersize=18, alpha=0.85, label="Beauty supply stores")

    if ctx is not None:
        try:
            ctx.add_basemap(ax, crs=tracts.crs, source=ctx.providers.CartoDB.Positron)
        except Exception:
            pass

    ax.set_title("Beauty Supply Access and Underserved Areas")
    ax.set_axis_off()
    ax.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def create_interactive_map(
    tracts: gpd.GeoDataFrame,
    stores: gpd.GeoDataFrame,
    html_path: Path,
) -> None:
    if folium is None:
        return

    tracts_ll = tracts.to_crs(epsg=4326)
    stores_ll = stores.to_crs(epsg=4326)
    center = merge_geometries(tracts_ll.geometry).centroid

    fmap = folium.Map(
        location=[center.y, center.x],
        zoom_start=10,
        tiles="CartoDB positron",
    )

    folium.Choropleth(
        geo_data=tracts_ll.to_json(),
        data=tracts_ll[["GEOID", "pct_black"]],
        columns=["GEOID", "pct_black"],
        key_on="feature.properties.GEOID",
        fill_color="YlOrRd",
        fill_opacity=0.65,
        line_opacity=0.25,
        legend_name="% African American",
    ).add_to(fmap)

    high_need = tracts_ll[tracts_ll["underserved_index"] >= tracts_ll["underserved_index"].quantile(0.75)]
    if not high_need.empty:
        folium.GeoJson(
            high_need.to_json(),
            name="High underserved tracts",
            style_function=lambda _: {
                "color": "#08306b",
                "weight": 2,
                "fillOpacity": 0.0,
            },
        ).add_to(fmap)

    for _, row in stores_ll.iterrows():
        label = str(row.get("name", "Store"))
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=4,
            color="black",
            fill=True,
            fill_opacity=0.9,
            popup=label,
        ).add_to(fmap)

    folium.LayerControl().add_to(fmap)
    fmap.save(str(html_path))


def save_outputs(
    tracts: gpd.GeoDataFrame,
    stores: gpd.GeoDataFrame,
    output_dir: Path,
    create_html: bool,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    geojson_path = output_dir / "beauty_access_enriched.geojson"
    csv_path = output_dir / "beauty_access_summary.csv"
    png_path = output_dir / "beauty_access_map.png"
    stores_geojson_path = output_dir / "beauty_supply_stores.geojson"
    html_path = output_dir / "beauty_access_map.html"

    tracts_out = tracts.to_crs(epsg=4326)
    stores_out = stores.to_crs(epsg=4326)

    tracts_out.to_file(geojson_path, driver="GeoJSON")
    stores_out.to_file(stores_geojson_path, driver="GeoJSON")

    summary_cols = [
        col
        for col in [
            "GEOID",
            "NAME",
            "total_pop",
            "black_pop",
            "pct_black",
            "median_income",
            "poverty_rate",
            "low_income",
            "nearest_store_km",
            "store_count_5km",
            "service_area_pct",
            "beauty_access_score",
            "underserved_index",
            "hotspot_flag",
            "hotspot_cluster",
        ]
        if col in tracts_out.columns
    ]
    tracts_out[summary_cols].to_csv(csv_path, index=False)

    create_static_map(tracts, stores, png_path)
    if create_html:
        create_interactive_map(tracts, stores, html_path)

    return {
        "geojson": geojson_path,
        "csv": csv_path,
        "png": png_path,
        "stores_geojson": stores_geojson_path,
        "html": html_path if create_html else None,
    }


def analyze_place(
    city: str,
    state_value: str,
    output_dir: str | Path,
    census_api_key: str = "",
    acs_year: int = 2022,
    buffer_km: float = 5.0,
    low_income_threshold: float = 40000.0,
    include_wig_shops: bool = False,
) -> dict:
    place_query = ", ".join(
        [part.strip() for part in [city, state_value, "USA"] if part and part.strip()]
    )
    if not place_query:
        raise ValueError("Please provide at least a city or state to query.")

    place = fetch_place_record(place_query)
    south, north, west, east = [float(value) for value in place["boundingbox"]]

    tracts = fetch_state_tracts(state_value=state_value, year=acs_year)
    tracts = subset_tracts_to_bbox(
        tracts,
        west=west,
        south=south,
        east=east,
        north=north,
    )
    tracts = attach_demographics(
        tracts,
        state_value=state_value,
        acs_year=acs_year,
        census_api_key=census_api_key,
        skip_census_api=False,
    )

    stores_df = fetch_osm_beauty_supply_stores(
        city=city,
        state_value=state_value,
        include_wig_shops=include_wig_shops,
    )
    stores = gpd.GeoDataFrame(
        stores_df,
        geometry=gpd.points_from_xy(stores_df["lon"], stores_df["lat"]),
        crs="EPSG:4326",
    )
    stores["store_id"] = np.arange(1, len(stores) + 1)

    metric_crs = choose_metric_crs(tracts, "auto")
    tracts_proj = tracts.to_crs(metric_crs)
    stores_proj = stores.to_crs(metric_crs)

    tracts_proj = compute_access_metrics(
        tracts_proj,
        stores_proj,
        buffer_km=buffer_km,
        low_income_threshold=low_income_threshold,
    )
    tracts_proj = cluster_hotspots(tracts_proj, buffer_km=buffer_km)

    result_dir = ensure_output_dir(Path(output_dir) / slugify(place_query))
    stores_csv_path = result_dir / "live_beauty_supply_stores.csv"
    stores_df.to_csv(stores_csv_path, index=False)

    outputs = save_outputs(
        tracts_proj,
        stores_proj,
        output_dir=result_dir,
        create_html=True,
    )
    outputs["stores_csv"] = stores_csv_path

    return {
        "place": place,
        "tracts": tracts_proj,
        "stores": stores_proj,
        "stores_table": stores_df,
        "outputs": outputs,
    }


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)

    log("Loading tract boundaries...")
    tracts = load_tracts(args.tracts)

    log("Loading beauty supply stores...")
    stores = load_stores(
        args.stores,
        category_filter=args.category_filter,
        include_wig_shops=args.include_wig_shops,
    )

    log("Attaching demographic data...")
    tracts = attach_demographics(
        tracts,
        state_value=args.state,
        acs_year=args.acs_year,
        census_api_key=args.census_api_key,
        skip_census_api=args.skip_census_api,
    )

    metric_crs = choose_metric_crs(tracts, args.metric_crs)
    log(f"Using projected CRS: {metric_crs}")
    tracts_proj = tracts.to_crs(metric_crs)
    stores_proj = stores.to_crs(metric_crs)

    log("Computing distance and access metrics...")
    tracts_proj = compute_access_metrics(
        tracts_proj,
        stores_proj,
        buffer_km=args.buffer_km,
        low_income_threshold=args.low_income_threshold,
    )

    log("Clustering underserved hotspots...")
    tracts_proj = cluster_hotspots(tracts_proj, buffer_km=args.buffer_km)

    outputs = save_outputs(
        tracts_proj,
        stores_proj,
        output_dir=output_dir,
        create_html=args.create_interactive_map,
    )

    print("\n=== Outputs ===")
    for key, value in outputs.items():
        if value:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
