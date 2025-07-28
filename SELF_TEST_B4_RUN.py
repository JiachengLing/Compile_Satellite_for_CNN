# self_test program before you run the CNN SDM data compilation
# As the readme.md stated, you will need 3 data stored in the Works/[Your package name] folder.


# Your satellite map as the predictor (The only predictor you will need)
# A boundary of your research area as .shp file
# The species presence data downloaded from GBIF.

# You do not need any extra work for processing them (even though your satellite maps are in pieces), just make sure
# satellite map is .tif or .tiff
# boundary is in .shp
# GBIF presence data (and other presence data as csv.)

# This self check is compiled as an exe file, which is more intuitive for you to use.

# it includes:


# 1. Self check: make sure if everything's complete

# 2. The boundary needs to be converted to fishnet, with a assigned cropping window size, and then clipped with the boundary.
# Algorithm: (1) extract the lon, lat range from the boundary
#            (2) Clip with boundary
# be aware, boundary might only include the terrestrial area, the boundary will be 1 km buffered before it was clipped,
# so there will raise a alarm before you do this, the satellite map you get from GEE, ideally, should be 1 km buffered
# as well.

# The grid cells in the fishnet are also assigned a SiteID (6 digit capitalized letters and numbers).
# Then you will be directed to the image_processing.exe ...


# Packages
import PySimpleGUI as sg, inspect
#print("Loaded from:", getattr(sg, "__file__", "unknown"))
#print("Some attrs :", [a for a in ("Text","Window","Button","Input","theme") if hasattr(sg,a)])

import os
import sys
import glob
import math
import random
import string
import traceback
import numpy as np
import pandas as pd

import PySimpleGUI as sg
import geopandas as gpd
from shapely.geometry import box
from shapely.ops import transform
from pyproj import CRS, Transformer
from pyproj.exceptions import CRSError


# ---------- runtime paths for bundled GDAL/PROJ (for .exe builds) ----------
if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    _BASE = sys._MEIPASS
elif getattr(sys, "frozen", False):
    _BASE = os.path.dirname(sys.executable)
else:
    _BASE = os.path.dirname(__file__)

os.environ.setdefault("GDAL_DATA", os.path.join(_BASE, "gdal_data"))
os.environ.setdefault("PROJ_LIB", os.path.join(_BASE, "proj_data"))


# Tool boxes
def find_files(work_dir):
    tif_list = sorted(glob.glob(os.path.join(work_dir, "*.tif"))) + \
               sorted(glob.glob(os.path.join(work_dir, "*.tiff")))
    shp_list = sorted(glob.glob(os.path.join(work_dir, "*.shp")))
    csv_list = sorted(glob.glob(os.path.join(work_dir, "*.csv")))
    return tif_list, shp_list, csv_list

def shapefile_sidecars_ok(shp_path):
    base = os.path.splitext(shp_path)[0]
    required = [base + ".dbf", base + ".shx"]
    missing = [p for p in required if not os.path.exists(p)]
    has_prj = os.path.exists(base + ".prj")
    return len(missing) == 0, missing, has_prj

def to_epsg4326(gdf):
    if gdf.crs is None:
        raise ValueError("Boundary has no CRS (.prj). Please provide/define one.")
    return gdf.to_crs(epsg=4326)

from pyproj import CRS

def choose_aeqd_crs_for_boundary(boundary_4326):
    """
    Build a local Azimuthal Equidistant (AEQD) CRS centered at the boundary centroid.
    This CRS uses meters, so distance-based operations (e.g., 1 km buffer) are meaningful.

    NOTE:
      - PROJ 9+ requires an ellipsoid/sphere to be specified.
        We set `+ellps=WGS84` to avoid: "Must specify ellipsoid or sphere".
    """
    c = boundary_4326.unary_union.centroid
    lon0, lat0 = float(c.x), float(c.y)
    proj_str = (
        f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} "
        f"+ellps=WGS84 +units=m +no_defs +type=crs"
    )
    return CRS.from_proj4(proj_str)



def buffer_geodesic_approx(boundary_4326, buffer_m):
    """
    Buffer the boundary by a metric distance (meters) using a local AEQD projection,
    then transform the result back to EPSG:4326.

    Why:
      - In geographic CRS (lon/lat degrees), `.buffer(1000)` would mean "1000 degrees",
        which is wrong. We project to a local, meter-based CRS first.

    Fallback:
      - If AEQD creation fails (rare), fall back to a spherical Earth radius (R=6371000 m).
    """
    if buffer_m <= 0:
        return boundary_4326.unary_union

    try:
        aeqd = choose_aeqd_crs_for_boundary(boundary_4326)
    except CRSError:
        c = boundary_4326.unary_union.centroid
        lon0, lat0 = float(c.x), float(c.y)
        aeqd = CRS.from_proj4(
            f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +R=6371000 +units=m +no_defs +type=crs"
        )

    fwd = Transformer.from_crs(boundary_4326.crs, aeqd, always_xy=True).transform
    inv = Transformer.from_crs(aeqd, boundary_4326.crs, always_xy=True).transform

    geom_proj = transform(fwd, boundary_4326.unary_union)
    geom_buff = geom_proj.buffer(buffer_m)
    return transform(inv, geom_buff)


def generate_site_ids(n, seed=20240728):
    rng = random.Random(seed)
    alphabet = string.ascii_uppercase + string.digits
    ids = []
    for i in range(n):
        base36 = np.base_repr(i, base=36).upper().rjust(6, '0')[-6:]
        rand2 = ''.join(rng.choices(alphabet, k=2))
        ids.append((base36[:4] + rand2).upper())
    ids = list(dict.fromkeys(ids))
    while len(ids) < n:
        ids.append(''.join(rng.choices(alphabet, k=6)))
    return ids[:n]

# ---------- CSV → point GeodataFrame ----------


_LAT_CANDS = [
    "decimallatitude", "latitude", "lat", "y", "lat_dd", "y_coord"
]
_LON_CANDS = [
    "decimallongitude", "longitude", "lon", "x", "lon_dd", "x_coord"
]

def _find_lat_lon_cols(df: pd.DataFrame):
    """
    Find likely latitude/longitude column names in a DataFrame (case-insensitive).
    Returns (lat_col_name, lon_col_name) or (None, None) if not found.
    """
    cols = {c.lower(): c for c in df.columns}
    lat_col = next((cols[c] for c in _LAT_CANDS if c in cols), None)
    lon_col = next((cols[c] for c in _LON_CANDS if c in cols), None)
    if lat_col is None or lon_col is None:
        # Also try to match GBIF's camel/snake-case permutations
        for c in df.columns:
            cl = c.lower().replace("_", "")
            if cl == "decimallatitude":
                lat_col = c if lat_col is None else lat_col
            if cl == "decimallongitude":
                lon_col = c if lon_col is None else lon_col
    return lat_col, lon_col

def read_presence_csv_to_points(csv_path: str, log=lambda s: None) -> gpd.GeoDataFrame:
    """
    Robustly read a presence CSV and convert to a GeoDataFrame of points (EPSG:4326).

    Features:
      - Tries multiple encodings and separators (comma, tab, semicolon, pipe).
      - Uses the python engine with sep=None to auto-detect if possible.
      - Skips bad lines to avoid parse failures on malformed rows.
      - Converts decimal commas ("12,345") to dots ("12.345") for lon/lat columns.
      - Validates lon/lat ranges and drops invalid rows.
    """
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    seps = [None, ",", "\t", ";", "|"]  # None → auto-detect with engine='python'
    df = None
    last_err = None
    tried = []

    for enc in encodings:
        for sep in seps:
            try:
                df = pd.read_csv(
                    csv_path,
                    sep=sep,
                    engine="python",        # more tolerant parser; allows sep=None auto-detect
                    encoding=enc,
                    quotechar='"',
                    escapechar="\\",
                    on_bad_lines="skip",    # skip malformed lines
                    dtype="object"          # read as strings first; we'll coerce later
                )
                if df.shape[1] >= 2 and len(df) > 0:
                    log(f"Parsed CSV with encoding={enc}, sep={'auto' if sep is None else repr(sep)} "
                        f"(rows={len(df)}, cols={df.shape[1]}).")
                    break
            except Exception as e:
                last_err = e
                tried.append(f"encoding={enc}, sep={'auto' if sep is None else repr(sep)} -> {type(e).__name__}")
        else:
            continue
        break

    if df is None:
        raise ValueError("Failed to parse CSV with tried options: " + "; ".join(tried)) from last_err

    # Detect lon/lat columns
    lat_col, lon_col = _find_lat_lon_cols(df)
    if not lat_col or not lon_col:
        raise ValueError(
            "Cannot find latitude/longitude columns. Expected names like "
            "decimalLatitude/decimalLongitude, latitude/longitude, lat/lon, etc."
        )
    log(f"Using CSV columns: lon='{lon_col}', lat='{lat_col}'")

    # Normalize decimal commas in lon/lat, then coerce to numeric
    # Normalize lon/lat text → strip spaces and turn decimal commas into dots
    for col in (lon_col, lat_col):
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.replace("\u00A0", "", regex=False)  # remove non‑breaking space if present
            .str.replace(",", ".", regex=False)  # convert decimal comma → dot
        )

    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    before = len(df)
    df = df.dropna(subset=[lon_col, lat_col])
    df = df[(df[lon_col] >= -180) & (df[lon_col] <= 180) & (df[lat_col] >= -90) & (df[lat_col] <= 90)]
    log(f"Presence points after cleaning: {len(df)}/{before}")

    # Build GeoDataFrame in EPSG:4326
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs="EPSG:4326"
    )
    return gdf
# Fishnet builders

def build_fishnet_angular(boundary_4326, dlon_deg, dlat_deg, log=lambda s: None):
    """Build grid in EPSG:4326 using angular steps (degrees)."""
    minx, miny, maxx, maxy = boundary_4326.total_bounds
    if dlon_deg <= 0 or dlat_deg <= 0:
        raise ValueError("Angular cell sizes must be positive.")
    xs = np.arange(minx, maxx, dlon_deg)
    ys = np.arange(miny, maxy, dlat_deg)
    cells = [box(x, y, x + dlon_deg, y + dlat_deg) for x in xs for y in ys]
    fishnet_all = gpd.GeoDataFrame({"geometry": cells}, crs=boundary_4326.crs)
    log(f"Fishnet generated (angular): {len(fishnet_all)} cells (unclipped).")
    return fishnet_all

def build_fishnet_metric(boundary_4326, cell_km, log=lambda s: None):
    """Build grid in a local metric CRS via AEQD, then transform back to EPSG:4326."""
    if cell_km <= 0:
        raise ValueError("Grid size (km) must be positive.")
    cell_m = float(cell_km) * 1000.0
    aeqd = choose_aeqd_crs_for_boundary(boundary_4326)
    fwd = Transformer.from_crs(boundary_4326.crs, aeqd, always_xy=True).transform
    inv = Transformer.from_crs(aeqd, boundary_4326.crs, always_xy=True).transform

    b_proj = transform(fwd, boundary_4326.unary_union)
    minx, miny, maxx, maxy = b_proj.bounds
    xs = np.arange(minx, maxx, cell_m)
    ys = np.arange(miny, maxy, cell_m)
    cells = [box(x, y, x + cell_m, y + cell_m) for x in xs for y in ys]
    fishnet_all_proj = gpd.GeoDataFrame({"geometry": cells}, crs=aeqd)
    fishnet_all = fishnet_all_proj.to_crs(boundary_4326.crs)
    log(f"Fishnet generated (metric): {len(fishnet_all)} cells (unclipped).")
    return fishnet_all

def make_fishnet(boundary_gdf, grid_mode, val_radians, val_degrees, val_km, buffer_m=1000.0, log=lambda s: None):
    """
    grid_mode: 'rad', 'deg', or 'km'
    Returns: (fishnet_all_4326, fishnet_clip_4326, buffered_geom_4326)
    """
    b4326 = to_epsg4326(boundary_gdf)
    geom_buffered = buffer_geodesic_approx(b4326, buffer_m) if buffer_m > 0 else b4326.unary_union
    log(f"Applied {buffer_m/1000:.3f} km buffer to boundary." if buffer_m > 0 else "No buffering.")

    if grid_mode == 'rad':
        dlon_deg = math.degrees(val_radians)
        dlat_deg = math.degrees(val_radians)
        fishnet_all = build_fishnet_angular(b4326, dlon_deg, dlat_deg, log=log)
        fishnet_all["cell_dlon_deg"] = dlon_deg
        fishnet_all["cell_dlat_deg"] = dlat_deg
        fishnet_all["cell_dlon_rad"] = val_radians
        fishnet_all["cell_dlat_rad"] = val_radians
    elif grid_mode == 'deg':
        fishnet_all = build_fishnet_angular(b4326, val_degrees, val_degrees, log=log)
        fishnet_all["cell_dlon_deg"] = val_degrees
        fishnet_all["cell_dlat_deg"] = val_degrees
        fishnet_all["cell_dlon_rad"] = math.radians(val_degrees)
        fishnet_all["cell_dlat_rad"] = math.radians(val_degrees)
    elif grid_mode == 'km':
        fishnet_all = build_fishnet_metric(b4326, val_km, log=log)
        fishnet_all["cell_km"] = val_km
    else:
        raise ValueError("Unknown grid mode.")

    fishnet_clip = gpd.overlay(
        fishnet_all,
        gpd.GeoDataFrame(geometry=[geom_buffered], crs=b4326.crs),
        how="intersection"
    ).reset_index(drop=True)

    site_ids = generate_site_ids(len(fishnet_clip))
    fishnet_clip["SiteID"] = site_ids
    fishnet_clip["cx"] = fishnet_clip.geometry.centroid.x
    fishnet_clip["cy"] = fishnet_clip.geometry.centroid.y
    b = fishnet_clip.geometry.bounds
    fishnet_clip["minx"] = b.minx
    fishnet_clip["miny"] = b.miny
    fishnet_clip["maxx"] = b.maxx
    fishnet_clip["maxy"] = b.maxy

    log(f"Clip done: {len(fishnet_clip)} cells remain.")
    return fishnet_all, fishnet_clip, geom_buffered

# ---------- Presence stats (points-in-grid) ----------

def assign_presence_to_grid(
    fishnet_clip: gpd.GeoDataFrame,
    points_4326: gpd.GeoDataFrame,
    log=lambda s: None,
    predicate: str = "intersects",   # default to intersects
):
    """
    Join presence points to grid cells using the given spatial predicate.
    With 'intersects', points lying exactly on grid boundaries may match multiple cells,
    so we de‑duplicate and keep only one cell per point (first by SiteID).
    """
    if points_4326.empty:
        log("Presence CSV has 0 valid points after cleaning.")
        out = fishnet_clip.copy()
        out["n_points"] = 0
        out["occ"] = pd.NA
        return out, out[["SiteID", "n_points", "occ"]].copy()

    grid = fishnet_clip[["SiteID", "geometry"]]
    # Keep an explicit point index so we can detect duplicates (same point ↔ multiple cells)
    pts = points_4326.reset_index().rename(columns={"index": "pt_idx"})

    # Spatial join
    join = gpd.sjoin(pts, grid, how="left", predicate=predicate)

    # De-duplicate for boundary cases when predicate != 'within'
    if predicate != "within":
        # Deterministic tie-break: keep the match with the smallest SiteID per point
        before = len(join)
        join = join.sort_values(["pt_idx", "SiteID"]).drop_duplicates("pt_idx", keep="first")
        after = len(join)
        if after < before:
            log(f"De-duplicated boundary matches: kept {after}/{before} point→cell links.")

    total_points = len(points_4326)
    in_grid = join["SiteID"].notna().sum()
    log(f"Presence points matched to grid: {in_grid}/{total_points} (predicate='{predicate}')")

    # Count per cell
    counts = join.groupby("SiteID", dropna=True).size().rename("n_points").reset_index()
    out = fishnet_clip.merge(counts, on="SiteID", how="left")
    out["n_points"] = out["n_points"].fillna(0).astype(int)
    # occ = 1 if there is at least one point, else NA
    out["occ"] = out["n_points"].apply(lambda n: 1 if n >= 1 else pd.NA)

    summary = out[["SiteID", "n_points", "occ"]].copy()
    return out, summary


# GUI


LAYOUT = [
    [sg.Text("Select data folder (Works/[Your package name]):"),
     sg.Input(key="-WORKDIR-"), sg.FolderBrowse("Browse…")],
    [sg.Text("Grid unit:"),
     sg.Radio("Radians", "UNIT", key="-U_RAD-", default=True),
     sg.Radio("Degrees", "UNIT", key="-U_DEG-"),
     sg.Radio("Kilometers", "UNIT", key="-U_KM-")],
    [sg.Text("Angular cell size (radians):"), sg.Input("0.01", size=(10,1), key="-CELL_RAD-")],
    [sg.Text("Angular cell size (degrees):"), sg.Input("0.5",  size=(10,1), key="-CELL_DEG-")],
    [sg.Text("Grid size (km):"), sg.Input("2", size=(10,1), key="-CELL_KM-")],
    [sg.Text("Boundary buffer (meters):"), sg.Input("1000", size=(10,1), key="-BUF_M-")],
    [sg.Button("Self check"), sg.Button("Build fishnet + Presence"), sg.Button("Exit")],
    [sg.ProgressBar(max_value=100, orientation="h", size=(40, 20), key="-PROG-")],
    [sg.Multiline(size=(100, 20), key="-LOG-", autoscroll=True, disabled=True)]
]

def main():
    sg.theme("SystemDefault")
    window = sg.Window("CNN SDM — Fishnet (angular/metric) + Presence from CSV (no raster)",
                       LAYOUT, finalize=True)

    def log(msg: str):
        window["-LOG-"].update(f"{msg}\n", append=True)

    def set_progress(n: int, total: int):
        pct = int(n * 100 / max(total, 1))
        window["-PROG-"].update(pct)

    while True:
        event, values = window.read(timeout=200)
        if event in (sg.WIN_CLOSED, "Exit"):
            break

        # ------------------------ Self check ------------------------
        if event == "Self check":
            window["-LOG-"].update("")
            workdir = (values.get("-WORKDIR-") or "").strip()
            if not workdir or not os.path.isdir(workdir):
                log("Please select a valid data folder first.")
                continue
            tifs, shps, csvs = find_files(workdir)
            log(f"Found rasters: {len(tifs)}; boundary: {len(shps)}; CSV: {len(csvs)}.")
            if len(shps) == 0:
                log("No boundary .shp found.")
            else:
                shp = shps[0]
                ok, missing, has_prj = shapefile_sidecars_ok(shp)
                if not ok: log(f"Missing Shapefile sidecars: {missing}")
                if not has_prj: log("Warning: .prj is missing. A defined CRS is strongly recommended.")
                try:
                    g = gpd.read_file(shp)
                    g4326 = to_epsg4326(g)
                    minx, miny, maxx, maxy = g4326.total_bounds
                    log("Boundary lon/lat bounds (EPSG:4326):")
                    log(f"  Lon: {minx:.6f} ~ {maxx:.6f}")
                    log(f"  Lat: {miny:.6f} ~ {maxy:.6f}")
                except Exception as e:
                    log(f"Failed to read boundary: {e}")
            if len(csvs) == 0:
                log("No .csv presence data found.")
            else:
                log(f"CSV example: {os.path.basename(csvs[0])}")
            log("Self check finished.")

        # -------------- Build fishnet + Presence from CSV ----------
        if event == "Build fishnet + Presence":
            window["-LOG-"].update("")
            window["-PROG-"].update(0)
            workdir = (values.get("-WORKDIR-") or "").strip()
            if not workdir or not os.path.isdir(workdir):
                log("Please select a valid data folder first.")
                continue

            # Parse units & values
            use_rad = values.get("-U_RAD-", False)
            use_deg = values.get("-U_DEG-", False)
            use_km  = values.get("-U_KM-",  False)

            try:
                buf_m = float(values.get("-BUF_M-", 1000))
            except Exception:
                log("Boundary buffer (meters) must be numeric.")
                continue

            try:
                val_rad = float(values.get("-CELL_RAD-", 0.01))
                val_deg = float(values.get("-CELL_DEG-", 0.5))
                val_km  = float(values.get("-CELL_KM_", 2))
                val_km  = float(values.get("-CELL_KM-", val_km))
            except Exception:
                log("Grid size inputs must be numeric.")
                continue

            # Need boundary
            tifs, shps, csvs = find_files(workdir)
            if len(shps) == 0:
                log("Boundary .shp is required.")
                continue
            try:
                boundary = gpd.read_file(shps[0])
            except Exception as e:
                log(f"Failed to read boundary: {e}")
                continue

            # Confirm buffering policy
            if buf_m > 0:
                sg.popup_ok(f"Note: boundary will be buffered by {buf_m/1000:.3f} km before clipping.")

            try:
                # Build fishnet
                if use_rad:
                    mode = 'rad'
                    fishnet_all, fishnet_clip, buffered_geom = make_fishnet(boundary, mode, val_rad, None, None, buffer_m=buf_m, log=log)
                elif use_deg:
                    mode = 'deg'
                    fishnet_all, fishnet_clip, buffered_geom = make_fishnet(boundary, mode, None, val_deg, None, buffer_m=buf_m, log=log)
                else:
                    mode = 'km'
                    fishnet_all, fishnet_clip, buffered_geom = make_fishnet(boundary, mode, None, None, val_km, buffer_m=buf_m, log=log)

                out_dir = workdir
                os.makedirs(out_dir, exist_ok=True)
                fishnet_all_path = os.path.join(out_dir, "fishnet_all.gpkg")
                fishnet_clip_path = os.path.join(out_dir, "fishnet_clip.gpkg")
                fishnet_all.to_file(fishnet_all_path, driver="GPKG")
                fishnet_clip.to_file(fishnet_clip_path, driver="GPKG")
                log(f"Wrote: {fishnet_all_path}")
                log(f"Wrote: {fishnet_clip_path}")

                # Presence CSV → points → stats
                if len(csvs) == 0:
                    log("No presence CSV found; skipping presence assignment.")
                    # Still export a minimal grid_index without presence fields
                    idx_csv = os.path.join(out_dir, "grid_index.csv")
                    fishnet_clip.drop(columns="geometry").to_csv(idx_csv, index=False, encoding="utf-8-sig")
                    log(f"Wrote (no presence): {idx_csv}")
                else:
                    csv_path = csvs[0]
                    log(f"Reading presence CSV: {os.path.basename(csv_path)}")
                    points = read_presence_csv_to_points(csv_path, log=log)

                    # Save points shapefile for inspection
                    pts_shp = os.path.join(out_dir, "presence_points.shp")
                    points.to_file(pts_shp, driver="ESRI Shapefile")
                    log(f"Wrote: {pts_shp}")

                    # Assign presence to grid
                    grid_with_occ, summary = assign_presence_to_grid(
                        fishnet_clip, points, log=log, predicate="intersects"
                    )

                    # Save enriched fishnet (GPKG + SHP)
                    fishnet_clip_occ_gpkg = os.path.join(out_dir, "fishnet_clip_with_occ.gpkg")
                    fishnet_clip_occ_shp  = os.path.join(out_dir, "fishnet_clip_with_occ.shp")
                    grid_with_occ.to_file(fishnet_clip_occ_gpkg, driver="GPKG")
                    grid_with_occ.to_file(fishnet_clip_occ_shp, driver="ESRI Shapefile")
                    log(f"Wrote: {fishnet_clip_occ_gpkg}")
                    log(f"Wrote: {fishnet_clip_occ_shp}")

                    # Save index CSV (to output/ and to the package folder as requested)
                    idx_out_csv = os.path.join(out_dir, "grid_presence.csv")
                    idx_pkg_csv = os.path.join(workdir,  "grid_presence.csv")
                    summary.to_csv(idx_out_csv, index=False, encoding="utf-8-sig")
                    summary.to_csv(idx_pkg_csv, index=False, encoding="utf-8-sig")
                    log(f"Wrote: {idx_out_csv}")
                    log(f"Wrote: {idx_pkg_csv}")

                window["-PROG-"].update(100)
            except Exception:
                log("An error occurred:\n" + traceback.format_exc())

    window.close()

if __name__ == "__main__":
    main()