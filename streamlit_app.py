from pathlib import Path
import html
import re

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

try:
    import folium
except Exception:  # pragma: no cover
    folium = None

from scripts.beauty_supply_access_pipeline import (
    DEFAULT_CENSUS_API_KEY,
    analyze_place,
    merge_geometries,
)

PRODUCT_BRAND_HINTS = {
    "beauty world": ["hair glue", "bonding glue", "weave", "wig"],
    "cosmoprof": ["hair color", "bonding glue", "styling", "salon pro"],
    "sally beauty": ["hair glue", "wig glue", "got2b", "mielle"],
    "ulta beauty": ["got2b", "edge control", "mielle", "beauty"],
    "merle norman": ["cosmetics", "makeup", "beauty"],
    "nc beauty outlet": ["beauty supply", "cosmetics", "hair products"],
    "bath & body works": ["body care", "fragrance", "lotion"],
}

PRODUCT_CATEGORY_OPTIONS = {
    "All categories": "",
    "Hair glue": "hair glue bonding glue salon pro",
    "Wig glue": "wig glue wig care lace glue",
    "Extensions": "extensions weave bundles hair products",
    "Edge control": "edge control got2b gel styling",
}

GENERIC_CATEGORY_HINTS = {
    "beauty": ["beauty supply", "hair care", "cosmetics"],
    "cosmetics": ["makeup", "cosmetics", "beauty"],
    "hairdresser_supply": [
        "hair glue",
        "bonding glue",
        "extensions",
        "wig care",
    ],
}


def format_text_value(value: object, fallback: str = "No data") -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none"}:
        return fallback
    return text


def format_category_label(value: object) -> str:
    text = format_text_value(value, fallback="No data")
    if text == "No data":
        return text

    normalized = text.replace("_", " ").strip()
    lowered = normalized.lower()
    if "hairdresser" in lowered or "hair dresser" in lowered:
        return "Hair Dresser"
    return normalized.title()


def build_search_tags(row: pd.Series) -> str:
    name = str(row.get("name", "") or "")
    category = str(row.get("category", "") or "")
    lower_name = name.lower()
    lower_category = category.lower()

    tags = set()
    if lower_name:
        tags.add(lower_name)
    if lower_category:
        tags.add(lower_category.replace("_", " "))

    for key, values in PRODUCT_BRAND_HINTS.items():
        if key in lower_name:
            tags.add(key)
            tags.update(values)

    for key, values in GENERIC_CATEGORY_HINTS.items():
        if key in lower_category:
            tags.update(values)

    return ", ".join(sorted(tags))


def filter_stores_for_search(
    stores_table: pd.DataFrame,
    product_query: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stores = stores_table.copy()
    stores["search_tags"] = stores.apply(build_search_tags, axis=1)

    if not product_query.strip():
        stores["matches_search"] = True
        stores["match_reason"] = "Showing all filtered beauty supply stores"
        return stores, stores.copy()

    tokens = [
        token.strip().lower()
        for token in re.split(r"[,;/]+|\s+", product_query)
        if token.strip()
    ]

    searchable = (
        stores["name"].fillna("").astype(str)
        + " "
        + stores["category"].fillna("").astype(str)
        + " "
        + stores["search_tags"].fillna("").astype(str)
    ).str.lower()

    match_mask = searchable.apply(
        lambda text: any(token in text for token in tokens)
    )
    stores["matches_search"] = match_mask
    stores["match_reason"] = stores["search_tags"].apply(
        lambda text: ", ".join(
            sorted({token for token in tokens if token in str(text).lower()})
        )
        or "name/category match"
    )

    return stores, stores[stores["matches_search"]].copy()


def render_store_cards(stores_table: pd.DataFrame) -> None:
    if stores_table.empty:
        st.warning("No stores matched the current search.")
        return

    st.markdown("### Highlighted results")
    card_columns = st.columns(min(3, len(stores_table)))

    for idx, (_, row) in enumerate(stores_table.head(6).iterrows()):
        with card_columns[idx % len(card_columns)]:
            with st.container(border=True):
                st.markdown(
                    f"**{format_text_value(row.get('name', 'Store'), fallback='Store')}**"
                )
                st.write(
                    f"**Category:** {format_category_label(row.get('category', ''))}"
                )
                st.write(
                    format_text_value(
                        row.get('address', ''),
                        fallback="No address listed",
                    )
                )
                st.caption(
                    f"Matches: {format_text_value(row.get('match_reason', 'general match'), fallback='general match')}"
                )


def render_map_legend_and_explanation() -> None:
    st.markdown("### Map legend and output explanation")
    st.markdown(
        """
        <div style="border:1px solid #e2e8f0;border-radius:12px;padding:1rem;
        background:#ffffff;margin-bottom:1rem;">
            <div style="font-weight:700;margin-bottom:0.6rem;color:#0f172a;">
                Color guide
            </div>
            <div style="display:flex;gap:0.6rem;align-items:center;margin-bottom:0.35rem;">
                <span style="display:inline-block;width:18px;height:18px;background:#fff7bc;
                border:1px solid #cbd5e1;"></span>
                <span><b>Light yellow / pale orange</b>: lower % African American</span>
            </div>
            <div style="display:flex;gap:0.6rem;align-items:center;margin-bottom:0.35rem;">
                <span style="display:inline-block;width:18px;height:18px;background:#fd8d3c;
                border:1px solid #cbd5e1;"></span>
                <span><b>Darker orange</b>: higher % African American</span>
            </div>
            <div style="display:flex;gap:0.6rem;align-items:center;margin-bottom:0.35rem;">
                <span style="display:inline-block;width:18px;height:18px;border:3px solid #08306b;
                background:#ffffff;"></span>
                <span><b>Dark blue outline</b>: higher-need / underserved tract</span>
            </div>
            <div style="display:flex;gap:0.6rem;align-items:center;">
                <span style="display:inline-block;width:18px;height:18px;border-radius:50%;
                background:#1d4ed8;"></span>
                <span><b>Blue point</b>: beauty supply location matching the filters</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("What the outputs mean", expanded=True):
        st.markdown(
            """
            - **Census tracts**: neighborhood-sized Census areas included in the query.
            - **Stores**: filtered beauty supply locations found in the selected area.
            - **Avg nearest store**: average distance from each tract center to the closest store.
            - **High-need tracts**: tracts with the highest underserved scores.
            - **Top underserved tracts**: places where lower access and higher need overlap.
            - **Product / brand results**: stores most relevant to the chosen product category or typed brand.
            """
        )


def build_map_html(
    tracts,
    stores_table: pd.DataFrame,
    product_query: str,
) -> str | None:
    if folium is None:
        return None

    tracts_ll = tracts.to_crs(epsg=4326)
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

    high_need = tracts_ll[
        tracts_ll["underserved_index"]
        >= tracts_ll["underserved_index"].quantile(0.75)
    ]
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

    rows_to_plot = stores_table.copy()
    if product_query.strip() and rows_to_plot["matches_search"].any():
        rows_to_plot = rows_to_plot[rows_to_plot["matches_search"]].copy()

    for _, row in rows_to_plot.iterrows():
        popup_parts = [
            f"<b>{html.escape(format_text_value(row.get('name', 'Store'), fallback='Store'))}</b>",
            f"Category: {html.escape(format_category_label(row.get('category', '')))}",
            f"Address: {html.escape(format_text_value(row.get('address', ''), fallback='No data'))}",
            f"Search tags: {html.escape(format_text_value(row.get('search_tags', ''), fallback='No data'))}",
        ]
        folium.CircleMarker(
            location=[float(row["lat"]), float(row["lon"])],
            radius=6,
            color="#1d4ed8",
            fill=True,
            fill_color="#1d4ed8",
            fill_opacity=0.9,
            popup="<br>".join(popup_parts),
        ).add_to(fmap)

    folium.LayerControl().add_to(fmap)
    return fmap.get_root().render()


st.set_page_config(
    page_title="Beauty Supply Access Explorer",
    layout="wide",
)

if "analysis_result" not in st.session_state:
    st.session_state["analysis_result"] = None

st.title("Beauty Supply Access Explorer")
st.caption(
    "Search any U.S. city, any U.S. state, or both to map beauty supply access, "
    "Census demographics, and underserved tracts."
)

with st.sidebar:
    st.header("Query settings")
    city = st.text_input(
        "City",
        value="Durham",
        placeholder="e.g. Chicago or leave blank for state-wide search",
    )
    state = st.text_input(
        "State",
        value="NC",
        placeholder="e.g. NC, Texas, or California",
    )
    product_category = st.selectbox(
        "Product category",
        options=list(PRODUCT_CATEGORY_OPTIONS.keys()),
        index=0,
    )
    product_query = st.text_input(
        "Product or brand search",
        placeholder="e.g. hair glue, Got2b, Sally Beauty",
    )
    st.caption(
        "Matches store names and likely-carried brand/product keywords."
    )
    buffer_km = st.slider(
        "Buffer distance (km)",
        min_value=1,
        max_value=25,
        value=5,
    )
    low_income_threshold = st.number_input(
        "Low-income threshold ($)",
        min_value=0,
        value=40000,
        step=5000,
    )
    include_wig_shops = st.checkbox("Include wig shops", value=False)
    st.caption("Census API access is configured securely from the local `.env` file.")
    run_query = st.button("Run analysis", type="primary")

st.info(
    "Tip: you can enter a U.S. city, a U.S. state, or both. "
    "State-only queries may take longer because they cover a larger area."
)

if run_query:
    if not city.strip() and not state.strip():
        st.error(
            "Please enter a U.S. city, a U.S. state, or both."
        )
    else:
        try:
            with st.spinner(
                "Fetching tracts, Census data, and beauty supply stores..."
            ):
                st.session_state["analysis_result"] = analyze_place(
                    city=city.strip(),
                    state_value=state.strip(),
                    output_dir=Path("outputs") / "streamlit_queries",
                    census_api_key=DEFAULT_CENSUS_API_KEY,
                    buffer_km=float(buffer_km),
                    low_income_threshold=float(low_income_threshold),
                    include_wig_shops=include_wig_shops,
                )
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            st.exception(exc)

result = st.session_state.get("analysis_result")

if result:
    tracts = result["tracts"].to_crs(epsg=4326)
    stores = result["stores"].to_crs(epsg=4326)
    outputs = result["outputs"]
    place_name = result["place"].get(
        "display_name",
        f"{city}, {state}",
    )

    stores_table = pd.DataFrame(result["stores_table"])
    category_query = PRODUCT_CATEGORY_OPTIONS.get(product_category, "")
    combined_query = " ".join(
        part for part in [category_query, product_query] if part.strip()
    )
    searchable_stores, matched_stores = filter_stores_for_search(
        stores_table,
        combined_query,
    )

    st.success(
        f"Mapped {len(stores):,} beauty supply stores across "
        f"{len(tracts):,} tracts for {place_name}."
    )

    if combined_query.strip():
        search_label = product_query.strip() or product_category
        st.info(
            f"Search for `{search_label}` matched {len(matched_stores):,} "
            "stores in the current results."
        )
        if matched_stores.empty:
            st.warning(
                "No current stores matched that brand/product term. "
                "Try a broader keyword such as `hair glue`, `wig`, or `beauty`."
            )

    metric_cols = st.columns(4)
    metric_cols[0].metric("Census tracts", f"{len(tracts):,}")
    metric_cols[1].metric("Stores", f"{len(stores):,}")
    metric_cols[2].metric(
        "Avg nearest store",
        f"{tracts['nearest_store_km'].mean():.2f} km",
    )
    high_need_count = int(
        (
            tracts["underserved_index"]
            >= tracts["underserved_index"].quantile(0.75)
        ).sum()
    )
    metric_cols[3].metric("High-need tracts", f"{high_need_count:,}")

    render_store_cards(
        matched_stores if combined_query.strip() else searchable_stores
    )

    st.subheader("Interactive map")
    map_col, legend_col = st.columns([3, 1.35])
    with map_col:
        map_html = build_map_html(tracts, searchable_stores, combined_query)
        if map_html:
            components.html(map_html, height=720, scrolling=True)
        else:
            html_path = outputs.get("html")
            if html_path and Path(html_path).exists():
                components.html(
                    Path(html_path).read_text(encoding="utf-8"),
                    height=720,
                    scrolling=True,
                )
            else:
                st.warning("Interactive HTML map was not created.")

    with legend_col:
        render_map_legend_and_explanation()

    st.subheader("Top underserved tracts")
    top_tracts = tracts[
        [
            col
            for col in [
                "NAME",
                "GEOID",
                "pct_black",
                "median_income",
                "poverty_rate",
                "nearest_store_km",
                "store_count_5km",
                "service_area_pct",
                "underserved_index",
            ]
            if col in tracts.columns
        ]
    ].sort_values("underserved_index", ascending=False)
    st.dataframe(top_tracts.head(25), use_container_width=True)

    st.subheader("Product / brand search results")
    table_to_show = matched_stores if combined_query.strip() else searchable_stores
    table_to_show = table_to_show.copy()
    if "category" in table_to_show.columns:
        table_to_show["category"] = table_to_show["category"].apply(
            format_category_label
        )
    if "address" in table_to_show.columns:
        table_to_show["address"] = table_to_show["address"].apply(
            lambda value: format_text_value(value, fallback="No data")
        )
    st.dataframe(table_to_show, use_container_width=True)

    st.subheader("Download outputs")
    download_cols = st.columns(4)

    csv_path = Path(outputs["csv"])
    geojson_path = Path(outputs["geojson"])
    stores_csv_path = Path(outputs["stores_csv"])
    matches_csv = table_to_show.to_csv(index=False).encode("utf-8")

    download_cols[0].download_button(
        "Download summary CSV",
        data=csv_path.read_bytes(),
        file_name=csv_path.name,
        mime="text/csv",
    )
    download_cols[1].download_button(
        "Download tract GeoJSON",
        data=geojson_path.read_bytes(),
        file_name=geojson_path.name,
        mime="application/geo+json",
    )
    download_cols[2].download_button(
        "Download stores CSV",
        data=stores_csv_path.read_bytes(),
        file_name=stores_csv_path.name,
        mime="text/csv",
    )
    download_cols[3].download_button(
        "Download matched stores",
        data=matches_csv,
        file_name="matched_stores.csv",
        mime="text/csv",
    )

    st.caption(f"Saved outputs to: {Path(csv_path).parent}")
else:
    st.write(
        "Use the sidebar to enter a city and state, click **Run analysis**, "
        "and optionally search for a product or brand."
    )
