# -*- coding: utf-8 -*-
"""
Automate distance calculations between supply stores and low-income areas,
and optionally from address points to the nearest qualifying store.

Author: Wilbert Fletcher
Requires: ArcGIS Pro (arcpy), permissions to the inputs and output GDB.

USAGE (example):
    python scripts/stores_low_income_distance.py ^
        --workspace "C:/GIS/projects/analysis.gdb" ^
        --target_sr "NAD 1983 (2011) StatePlane Texas South Central FIPS 4204 (US Feet)" ^
        --stores "C:/GIS/data/stores.csv" ^
        --locator "C:/GIS/locators/EnterpriseLocator.loc" ^
        --address_fields "Address;City;State;ZIP" ^
        --low_income_fc "C:/GIS/data/low_income.gdb/LowIncomeBlockGroups" ^
        --near_threshold "1 Miles" ^
        --output_prefix "Humble" ^
        --addresses_fc "C:/GIS/data/addresses.gdb/ServiceAddresses"

Notes:
- If your stores are already a point feature class, pass it to --stores and omit
  --locator and --address_fields.
- Use a projected coordinate system appropriate for your study area so PLANAR
  distance measurements are accurate.
"""

import argparse
import os
import sys
import traceback

import arcpy

arcpy.env.overwriteOutput = True


def log(message):
    """Print to the console and ArcGIS messages when available."""
    print(message)
    try:
        arcpy.AddMessage(message)
    except Exception:
        pass


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Identify supply stores inside/near low-income areas and compute distances."
    )
    parser.add_argument(
        "--workspace",
        required=True,
        help="Output file geodatabase (.gdb). It will be created if it does not exist.",
    )
    parser.add_argument(
        "--target_sr",
        required=True,
        help="Projected spatial reference name or WKID appropriate for local distance analysis.",
    )
    parser.add_argument(
        "--stores",
        required=True,
        help="Stores input: CSV (addresses) or point feature class.",
    )
    parser.add_argument(
        "--locator",
        required=False,
        help="Locator path for geocoding (required if --stores is a CSV).",
    )
    parser.add_argument(
        "--address_fields",
        required=False,
        help="CSV address fields in order: Address;City;State;ZIP or a single full-address field.",
    )
    parser.add_argument(
        "--low_income_fc",
        required=True,
        help="Polygon feature class representing low-income areas.",
    )
    parser.add_argument(
        "--near_threshold",
        default="1 Miles",
        help="Buffer distance for the 'near' classification (for example: '1 Miles').",
    )
    parser.add_argument(
        "--output_prefix",
        default="hair_glue",
        help="Prefix for generated datasets and CSV exports.",
    )
    parser.add_argument(
        "--addresses_fc",
        required=False,
        help="Optional address points for nearest-qualifying-store analysis.",
    )
    return parser.parse_args(argv)


def get_spatial_reference(sr_param):
    """Accept a WKID or a spatial reference name."""
    try:
        sr = arcpy.SpatialReference(int(sr_param))
    except Exception:
        sr = arcpy.SpatialReference(str(sr_param))

    if not sr or sr.name in (None, "", "Unknown"):
        raise ValueError(f"Could not resolve target spatial reference from: {sr_param}")
    return sr


def ensure_gdb(path):
    if not path.lower().endswith(".gdb"):
        raise ValueError("--workspace must point to a file geodatabase ending in .gdb")

    folder, name = os.path.split(path)
    if not arcpy.Exists(path):
        if folder and not os.path.isdir(folder):
            os.makedirs(folder, exist_ok=True)
        log(f"Creating output geodatabase: {path}")
        arcpy.management.CreateFileGDB(folder if folder else os.getcwd(), name.replace(".gdb", ""))
    return path


def make_output_path(out_gdb, base_name):
    return os.path.join(out_gdb, arcpy.ValidateTableName(base_name, out_gdb))


def unit_to_miles_divisor(sr):
    """Return a divisor that converts the spatial reference linear unit to miles."""
    try:
        meters_per_unit = float(getattr(sr, "metersPerUnit", 0) or 0)
        if meters_per_unit > 0:
            return 1609.344 / meters_per_unit
    except Exception:
        pass

    name = (getattr(sr, "linearUnitName", "") or "").lower()
    if "mile" in name:
        return 1.0
    if "foot" in name:
        return 5280.0
    if "kilometer" in name:
        return 1.609344
    return 1609.344  # default assume meters


def project_if_needed(in_fc, out_gdb, target_sr, name_suffix="_proj"):
    desc = arcpy.Describe(in_fc)
    in_sr = desc.spatialReference

    if not in_sr or in_sr.name == "Unknown":
        raise ValueError(f"Layer has an unknown spatial reference: {in_fc}")

    same_factory = in_sr.factoryCode not in (None, 0) and in_sr.factoryCode == target_sr.factoryCode
    same_name = in_sr.name == target_sr.name
    if same_factory or same_name:
        return in_fc

    out_name = f"{desc.baseName}{name_suffix}"
    out_fc = make_output_path(out_gdb, out_name)
    if arcpy.Exists(out_fc):
        arcpy.management.Delete(out_fc)

    log(f"Projecting {in_fc} -> {out_fc}")
    arcpy.management.Project(in_fc, out_fc, target_sr)
    return out_fc


def build_geocode_field_map(address_fields):
    """
    Build a common geocode field mapping string.
    If your enterprise locator uses different role names, adjust this function.
    """
    parts = [part.strip() for part in address_fields.split(";") if part.strip()]
    if not parts:
        raise ValueError("--address_fields must include at least one field name.")

    if len(parts) == 1:
        return f"'Single Line Input' {parts[0]} VISIBLE NONE"

    mappings = [("Address or Place", parts[0])]
    if len(parts) > 1:
        mappings.append(("City", parts[1]))
    if len(parts) > 2:
        mappings.append(("Region", parts[2]))
    if len(parts) > 3:
        mappings.append(("Postal", parts[3]))

    return ";".join(f"'{role}' {field} VISIBLE NONE" for role, field in mappings if field)


def geocode_csv(csv_path, locator, address_fields, out_gdb, target_sr, out_name="StoresGeocoded"):
    if not locator or not address_fields:
        raise ValueError("Geocoding requires both --locator and --address_fields.")

    out_fc = make_output_path(out_gdb, out_name)
    field_map = build_geocode_field_map(address_fields)

    log("Geocoding store addresses from CSV...")
    with arcpy.EnvManager(outputCoordinateSystem=target_sr):
        arcpy.geocoding.GeocodeAddresses(csv_path, locator, field_map, out_fc)

    return project_if_needed(out_fc, out_gdb, target_sr)


def add_or_calc_field(fc, field_name, field_type="SHORT", expr=None):
    existing = {field.name.upper() for field in arcpy.ListFields(fc)}
    if field_name.upper() not in existing:
        arcpy.management.AddField(fc, field_name, field_type)
    if expr is not None:
        arcpy.management.CalculateField(fc, field_name, expr, expression_type="PYTHON3")


def flag_stores_in_and_near(stores_fc, low_income_fc, near_threshold, out_gdb, target_sr, output_prefix):
    stores_fc = project_if_needed(stores_fc, out_gdb, target_sr)
    low_income_fc = project_if_needed(low_income_fc, out_gdb, target_sr)

    stores_lyr = "stores_lyr"
    low_lyr = "low_lyr"
    low_buf_lyr = "low_buf_lyr"

    arcpy.management.MakeFeatureLayer(stores_fc, stores_lyr)
    arcpy.management.MakeFeatureLayer(low_income_fc, low_lyr)

    add_or_calc_field(stores_fc, "IN_LOW_INCOME", "SHORT", "0")
    arcpy.management.SelectLayerByLocation(stores_lyr, "INTERSECT", low_lyr, selection_type="NEW_SELECTION")
    arcpy.management.CalculateField(stores_lyr, "IN_LOW_INCOME", "1", expression_type="PYTHON3")
    arcpy.management.SelectLayerByAttribute(stores_lyr, "CLEAR_SELECTION")

    low_buf = make_output_path(out_gdb, f"{output_prefix}_LowIncome_Buffer")
    if arcpy.Exists(low_buf):
        arcpy.management.Delete(low_buf)
    arcpy.analysis.Buffer(low_income_fc, low_buf, near_threshold, dissolve_option="ALL")

    add_or_calc_field(stores_fc, "NEAR_TO_LOW_INCOME", "SHORT", "0")
    arcpy.management.MakeFeatureLayer(low_buf, low_buf_lyr)
    arcpy.management.SelectLayerByLocation(stores_lyr, "INTERSECT", low_buf_lyr, selection_type="NEW_SELECTION")
    arcpy.management.CalculateField(stores_lyr, "NEAR_TO_LOW_INCOME", "1", expression_type="PYTHON3")
    arcpy.management.SelectLayerByAttribute(stores_lyr, "CLEAR_SELECTION")

    arcpy.analysis.Near(stores_fc, low_income_fc, method="PLANAR")
    miles_divisor = unit_to_miles_divisor(arcpy.Describe(stores_fc).spatialReference)
    add_or_calc_field(
        stores_fc,
        "DIST_TO_LOWINC_MI",
        "DOUBLE",
        f"0 if !NEAR_DIST! < 0 else !NEAR_DIST! / {miles_divisor}",
    )

    qualifying = make_output_path(out_gdb, f"{output_prefix}_Stores_Qualifying")
    arcpy.management.MakeFeatureLayer(
        stores_fc,
        "stores_qualifying_lyr",
        "IN_LOW_INCOME = 1 OR NEAR_TO_LOW_INCOME = 1",
    )
    if arcpy.Exists(qualifying):
        arcpy.management.Delete(qualifying)
    arcpy.management.CopyFeatures("stores_qualifying_lyr", qualifying)

    return stores_fc, low_income_fc, low_buf, qualifying


def nearest_store_for_addresses(addresses_fc, qualifying_stores_fc, out_gdb, target_sr, out_prefix):
    addresses_fc = project_if_needed(addresses_fc, out_gdb, target_sr)
    qualifying_stores_fc = project_if_needed(qualifying_stores_fc, out_gdb, target_sr)

    arcpy.analysis.Near(addresses_fc, qualifying_stores_fc, method="PLANAR")
    miles_divisor = unit_to_miles_divisor(arcpy.Describe(addresses_fc).spatialReference)
    add_or_calc_field(
        addresses_fc,
        "NEAR_STORE_MI",
        "DOUBLE",
        f"0 if !NEAR_DIST! < 0 else !NEAR_DIST! / {miles_divisor}",
    )

    near_tab = make_output_path(out_gdb, f"{out_prefix}_AddrToStore_Near")
    if arcpy.Exists(near_tab):
        arcpy.management.Delete(near_tab)

    arcpy.analysis.GenerateNearTable(
        addresses_fc,
        qualifying_stores_fc,
        near_tab,
        "",
        "NO_LOCATION",
        "NO_ANGLE",
        "CLOSEST",
        1,
        "PLANAR",
    )

    return addresses_fc, near_tab


def export_csvs(items, out_folder, prefix):
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder, exist_ok=True)

    for item in items:
        if not item:
            continue
        csv_name = f"{prefix}_{os.path.basename(str(item))}.csv"
        out_csv = os.path.join(out_folder, csv_name)
        if arcpy.Exists(out_csv):
            arcpy.management.Delete(out_csv)
        arcpy.conversion.TableToTable(item, out_folder, csv_name)
        log(f"CSV exported: {out_csv}")


def main(argv):
    args = parse_args(argv)
    gdb = ensure_gdb(args.workspace)
    target_sr = get_spatial_reference(args.target_sr)

    if args.stores.lower().endswith(".csv"):
        if not args.locator:
            raise ValueError("When --stores is a CSV, --locator is required.")
        if not args.address_fields:
            raise ValueError("When --stores is a CSV, --address_fields is required.")
        stores_fc = geocode_csv(
            args.stores,
            args.locator,
            args.address_fields,
            gdb,
            target_sr,
            out_name=f"{args.output_prefix}_Stores",
        )
    else:
        stores_fc = project_if_needed(args.stores, gdb, target_sr)

    low_income_fc = project_if_needed(args.low_income_fc, gdb, target_sr)

    stores_fc, low_income_fc, low_buf, qualifying_stores_fc = flag_stores_in_and_near(
        stores_fc,
        low_income_fc,
        args.near_threshold,
        gdb,
        target_sr,
        args.output_prefix,
    )

    addresses_fc = None
    near_table = None
    if args.addresses_fc:
        addresses_fc, near_table = nearest_store_for_addresses(
            args.addresses_fc,
            qualifying_stores_fc,
            gdb,
            target_sr,
            args.output_prefix,
        )

    out_folder = os.path.dirname(gdb) if os.path.dirname(gdb) else os.getcwd()
    export_list = [stores_fc, qualifying_stores_fc]
    if addresses_fc:
        export_list.append(addresses_fc)
    if near_table:
        export_list.append(near_table)
    export_csvs(export_list, out_folder, args.output_prefix)

    print("\n=== Outputs ===")
    print(f"All stores annotated:   {stores_fc}")
    print(f"Qualifying stores:      {qualifying_stores_fc}")
    print(f"Low-income buffer:      {low_buf}")
    if addresses_fc:
        print(f"Addresses annotated:    {addresses_fc}")
    if near_table:
        print(f"Near table:             {near_table}")
    print("CSV exports written to:", out_folder)


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except Exception as exc:
        traceback.print_exc()
        try:
            arcpy.AddError(str(exc))
        except Exception:
            pass
        sys.exit(1)
