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


# Package


import os, sys, glob, math, random, string, traceback
import numpy as np
import pandas as pd

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import geopandas as gpd
import shapely
import shapely._geos
import shapely._geometry
import shapely._geometry_helpers
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
os.environ.setdefault("PROJ_LIB",  os.path.join(_BASE, "proj_data"))

# Tool boxes
def find_files(work_dir):
    """Collect .tif/.tiff, .shp, .csv under work_dir (non-recursive)."""
    tif_list = sorted(glob.glob(os.path.join(work_dir, "*.tif"))) + \
               sorted(glob.glob(os.path.join(work_dir, "*.tiff")))
    shp_list = sorted(glob.glob(os.path.join(work_dir, "*.shp")))
    csv_list = sorted(glob.glob(os.path.join(work_dir, "*.csv")))
    return tif_list, shp_list, csv_list

def shapefile_sidecars_ok(shp_path):
    """Check Shapefile sidecars: .dbf, .shx required; .prj optional but recommended."""
    base = os.path.splitext(shp_path)[0]
    required = [base + ".dbf", base + ".shx"]
    missing = [p for p in required if not os.path.exists(p)]
    has_prj = os.path.exists(base + ".prj")
    return len(missing) == 0, missing, has_prj

def to_epsg4326(gdf):
    """Ensure boundary is in EPSG:4326."""
    if gdf.crs is None:
        raise ValueError("Boundary has no CRS (.prj). Please provide/define one.")
    return gdf.to_crs(epsg=4326)

def choose_aeqd_crs_for_boundary(boundary_4326):
    """
    Build a local Azimuthal Equidistant (AEQD) CRS centered at the boundary centroid.
    PROJ 9+ requires an ellipsoid/sphere -> use +ellps=WGS84.
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
    Buffer by meters using local AEQD, then back to EPSG:4326.
    In lon/lat CRS, .buffer(1000) would mean "1000 degrees" (wrong), so we project first.
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
    """Deterministic 6-char IDs (A–Z + 0–9)."""
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

# --------------------- CSV -> points (robust reader) ---------------------

_LAT_CANDS = ["decimallatitude", "latitude", "lat", "y", "lat_dd", "y_coord"]
_LON_CANDS = ["decimallongitude", "longitude", "lon", "x", "lon_dd", "x_coord"]

def _find_lat_lon_cols(df: pd.DataFrame):
    """Find likely latitude/longitude column names (case-insensitive)."""
    cols = {c.lower(): c for c in df.columns}
    lat_col = next((cols[c] for c in _LAT_CANDS if c in cols), None)
    lon_col = next((cols[c] for c in _LON_CANDS if c in cols), None)
    if lat_col is None or lon_col is None:
        for c in df.columns:
            cl = c.lower().replace("_", "")
            if cl == "decimallatitude" and lat_col is None: lat_col = c
            if cl == "decimallongitude" and lon_col is None: lon_col = c
    return lat_col, lon_col

def read_presence_csv_to_points(csv_path: str, log=print) -> gpd.GeoDataFrame:
    """
    Robustly read presence CSV and convert to EPSG:4326 point layer.
    - Tries multiple encodings and separators (auto-detect first).
    - Skips malformed lines.
    - Fixes decimal commas in lon/lat.
    """
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    seps = [None, ",", "\t", ";", "|"]
    df = None
    last_err, tried = None, []
    for enc in encodings:
        for sep in seps:
            try:
                df = pd.read_csv(
                    csv_path, sep=sep, engine="python", encoding=enc,
                    quotechar='"', escapechar="\\", on_bad_lines="skip", dtype="object"
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

    lat_col, lon_col = _find_lat_lon_cols(df)
    if not lat_col or not lon_col:
        raise ValueError("Cannot find latitude/longitude columns (e.g., decimalLatitude/decimalLongitude).")
    log(f"Using CSV columns: lon='{lon_col}', lat='{lat_col}'")

    # normalize text -> decimal dot
    for col in (lon_col, lat_col):
        df[col] = (
            df[col].astype(str).str.strip()
                 .str.replace("\u00A0", "", regex=False)
                 .str.replace(",", ".", regex=False)
        )

    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    before = len(df)
    df = df.dropna(subset=[lon_col, lat_col])
    df = df[(df[lon_col] >= -180) & (df[lon_col] <= 180) & (df[lat_col] >= -90) & (df[lat_col] <= 90)]
    log(f"Presence points after cleaning: {len(df)}/{before}")

    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon_col], df[lat_col]), crs="EPSG:4326")
    return gdf

# -------------------------- Fishnet builders --------------------------

def build_fishnet_angular(boundary_4326, dlon_deg, dlat_deg, log=print):
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

def build_fishnet_metric(boundary_4326, cell_km, log=print):
    """Build grid in local metric CRS (AEQD), then transform back to EPSG:4326."""
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

def make_fishnet(boundary_gdf, grid_mode, val_radians, val_degrees, val_km, buffer_m=1000.0, log=print):
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

    centroids = fishnet_all.geometry.centroid
    mask = centroids.within(geom_buffered)  
    fishnet_clip = fishnet_all.loc[mask].copy().reset_index(drop=True)

    site_ids = generate_site_ids(len(fishnet_clip))
    fishnet_clip["SiteID"] = site_ids
    fishnet_clip["cx"] = fishnet_clip.geometry.centroid.x
    fishnet_clip["cy"] = fishnet_clip.geometry.centroid.y
    b = fishnet_clip.geometry.bounds
    fishnet_clip["minx"] = b.minx; fishnet_clip["miny"] = b.miny
    fishnet_clip["maxx"] = b.maxx; fishnet_clip["maxy"] = b.maxy

    log(f"Clip done: {len(fishnet_clip)} cells remain.")
    return fishnet_all, fishnet_clip, geom_buffered

# ------------------ Presence stats (points-in-grid) ------------------

def assign_presence_to_grid(fishnet_clip: gpd.GeoDataFrame, points_4326: gpd.GeoDataFrame,
                            log=print, predicate: str = "intersects"):
    """
    Spatially join presence points to grid cells.
    With 'intersects', points on grid boundaries can match multiple cells -> de-duplicate.
    """
    if points_4326.empty:
        log("Presence CSV has 0 valid points after cleaning.")
        out = fishnet_clip.copy()
        out["n_points"] = 0
        out["occ"] = pd.NA
        return out, out[["SiteID", "n_points", "occ"]].copy()

    grid = fishnet_clip[["SiteID", "geometry"]]
    pts = points_4326.reset_index().rename(columns={"index": "pt_idx"})
    join = gpd.sjoin(pts, grid, how="left", predicate=predicate)

    if predicate != "within":
        before = len(join)
        join = join.sort_values(["pt_idx", "SiteID"]).drop_duplicates("pt_idx", keep="first")
        after = len(join)
        if after < before:
            log(f"De-duplicated boundary matches: kept {after}/{before} point→cell links.")

    total_points = len(points_4326)
    in_grid = join["SiteID"].notna().sum()
    log(f"Presence points matched to grid: {in_grid}/{total_points} (predicate='{predicate}')")

    counts = join.groupby("SiteID", dropna=True).size().rename("n_points").reset_index()
    out = fishnet_clip.merge(counts, on="SiteID", how="left")
    out["n_points"] = out["n_points"].fillna(0).astype(int)
    out["occ"] = out["n_points"].apply(lambda n: 1 if n >= 1 else pd.NA)
    summary = out[["SiteID", "n_points", "occ"]].copy()
    return out, summary

# GUI


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CNN SDM — Fishnet (angular/metric) + Presence from CSV (no raster)")
        self.geometry("900x600")

        # --- Top: Work folder selector ---
        self.workdir = self._add_path_row("Select data folder (Works/[Your package name]):", is_dir=True)

        # --- Grid unit + sizes ---
        unit_frame = ttk.LabelFrame(self, text="Grid unit")
        unit_frame.pack(fill="x", padx=10, pady=6)
        self.unit_var = tk.StringVar(value="rad")
        ttk.Radiobutton(unit_frame, text="Radians",   variable=self.unit_var, value="rad").pack(side="left", padx=6)
        ttk.Radiobutton(unit_frame, text="Degrees",   variable=self.unit_var, value="deg").pack(side="left", padx=6)
        ttk.Radiobutton(unit_frame, text="Kilometers",variable=self.unit_var, value="km").pack(side="left", padx=6)

        size_frame = ttk.Frame(self); size_frame.pack(fill="x", padx=10)
        self.cell_rad = self._add_labeled_entry(size_frame, "Angular cell size (radians):", "0.01", 12)
        self.cell_deg = self._add_labeled_entry(size_frame, "Angular cell size (degrees):", "0.5",  12)
        self.cell_km  = self._add_labeled_entry(size_frame, "Grid size (km):",               "2",    12)
        self.buf_m    = self._add_labeled_entry(size_frame, "Boundary buffer (meters):",     "1000", 12)

        # --- Buttons ---
        btn_frame = ttk.Frame(self); btn_frame.pack(fill="x", padx=10, pady=6)
        ttk.Button(btn_frame, text="Self check", command=self.run_self_check).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Build fishnet + Presence", command=self.run_build).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Exit", command=self.destroy).pack(side="right", padx=5)

        # --- Progress + log ---
        prog_frame = ttk.Frame(self); prog_frame.pack(fill="x", padx=10, pady=4)
        self.progress = ttk.Progressbar(prog_frame, mode="determinate", maximum=100.0)
        self.progress.pack(fill="x", side="left", expand=True)
        self.progress_label = ttk.Label(prog_frame, text="Ready  0%")
        self.progress_label.pack(side="left", padx=8)

        log_frame = ttk.LabelFrame(self, text="Run log"); log_frame.pack(fill="both", expand=True, padx=10, pady=6)
        self.txt = tk.Text(log_frame, wrap="word")
        self.txt.pack(fill="both", expand=True)

        self._set_running(False)

    # ---- UI helpers ----
    def _add_path_row(self, label, is_dir=False):
        frame = ttk.Frame(self); frame.pack(fill="x", padx=10, pady=2)
        ttk.Label(frame, text=label, width=36).pack(side="left")
        var = tk.StringVar()
        entry = ttk.Entry(frame, textvariable=var); entry.pack(side="left", fill="x", expand=True, padx=6)
        def browse():
            start_dir = var.get().strip() or os.getcwd()
            p = filedialog.askdirectory(initialdir=start_dir) if is_dir else filedialog.askopenfilename(initialdir=start_dir)
            if p: var.set(p)
        ttk.Button(frame, text="Browse…", command=browse).pack(side="left")
        return var

    def _add_labeled_entry(self, parent, label, default, width=10):
        frame = ttk.Frame(parent); frame.pack(side="left", padx=6, pady=4)
        ttk.Label(frame, text=label).pack(side="left")
        var = tk.StringVar(value=default)
        ttk.Entry(frame, textvariable=var, width=width).pack(side="left")
        return var

    def _log(self, msg):
        self.txt.insert("end", str(msg) + "\n")
        self.txt.see("end")
        self.update_idletasks()

    def _set_running(self, running: bool):
        state = "disabled" if running else "normal"
        # Simple state control; buttons can be enabled/disabled here if needed
        if not running:
            self.progress['value'] = 0
            self.progress_label.config(text="Ready  0%")

    def _set_progress(self, pct: float, label: str = None):
        pct = max(0.0, min(100.0, pct))
        self.progress['value'] = pct
        if label is not None:
            self.progress_label.config(text=f"{label}  {pct:.1f}%")
        self.update_idletasks()

    # ---- Actions ----
    def run_self_check(self):
        try:
            self._set_running(True)
            self.txt.delete("1.0", "end")
            workdir = self.workdir.get().strip()
            if not workdir or not os.path.isdir(workdir):
                self._log("Please select a valid data folder first.")
                return

            tifs, shps, csvs = find_files(workdir)
            self._log(f"Found rasters: {len(tifs)}; boundary: {len(shps)}; CSV: {len(csvs)}.")

            if len(shps) == 0:
                self._log("No boundary .shp found.")
            else:
                shp = shps[0]
                ok, missing, has_prj = shapefile_sidecars_ok(shp)
                if not ok: self._log(f"Missing Shapefile sidecars: {missing}")
                if not has_prj: self._log("Warning: .prj is missing. A defined CRS is strongly recommended.")
                try:
                    g = gpd.read_file(shp)
                    g4326 = to_epsg4326(g)
                    minx, miny, maxx, maxy = g4326.total_bounds
                    self._log("Boundary lon/lat bounds (EPSG:4326):")
                    self._log(f"  Lon: {minx:.6f} ~ {maxx:.6f}")
                    self._log(f"  Lat: {miny:.6f} ~ {maxy:.6f}")
                except Exception as e:
                    self._log(f"Failed to read boundary: {e}")

            if len(csvs) == 0:
                self._log("No .csv presence data found.")
            else:
                self._log(f"CSV example: {os.path.basename(csvs[0])}")

            if len(tifs) > 0 and len(shps) > 0 and len(csvs) > 0:
                self._log("✅ Self check passed: all three data types present.")
            else:
                self._log("ℹ️ Self check incomplete. You can still build the fishnet if boundary is present.")

        finally:
            self._set_progress(100, "Self check done")
            self._set_running(False)

    def run_build(self):
        try:
            self._set_running(True)
            self.txt.delete("1.0", "end")
            self._set_progress(0, "Start")

            workdir = self.workdir.get().strip()
            if not workdir or not os.path.isdir(workdir):
                self._log("Please select a valid data folder first.")
                return

            # Parse unit & params
            unit = self.unit_var.get()
            try:
                buf_m  = float(self.buf_m.get().strip() or "1000")
                val_rad = float(self.cell_rad.get().strip() or "0.01")
                val_deg = float(self.cell_deg.get().strip() or "0.5")
                val_km  = float(self.cell_km.get().strip()  or "2")
            except Exception:
                self._log("Grid/buffer inputs must be numeric.")
                return

            tifs, shps, csvs = find_files(workdir)
            if len(shps) == 0:
                self._log("Boundary .shp is required.")
                return

            # Read boundary
            try:
                boundary = gpd.read_file(shps[0])
            except Exception as e:
                self._log(f"Failed to read boundary: {e}")
                return

            if buf_m > 0:
                messagebox.showinfo("Notice", f"Boundary will be buffered by {buf_m/1000:.3f} km before clipping.")

            # 1) Build fishnet
            self._set_progress(10, "Building fishnet")
            if unit == "rad":
                fishnet_all, fishnet_clip, _ = make_fishnet(boundary, 'rad', val_rad, None, None, buffer_m=buf_m, log=self._log)
            elif unit == "deg":
                fishnet_all, fishnet_clip, _ = make_fishnet(boundary, 'deg', None, val_deg, None, buffer_m=buf_m, log=self._log)
            else:
                fishnet_all, fishnet_clip, _ = make_fishnet(boundary, 'km',  None, None, val_km,  buffer_m=buf_m, log=self._log)

            out_dir = workdir
            os.makedirs(out_dir, exist_ok=True)
            fishnet_all_path = os.path.join(out_dir, "fishnet_all.gpkg")
            fishnet_clip_path = os.path.join(out_dir, "fishnet_clip.gpkg")
            fishnet_all.to_file(fishnet_all_path, driver="GPKG")
            fishnet_clip.to_file(fishnet_clip_path, driver="GPKG")
            self._log(f"Wrote: {fishnet_all_path}")
            self._log(f"Wrote: {fishnet_clip_path}")

            # 2) Presence CSV -> points -> stats
            self._set_progress(60, "Presence join")
            if len(csvs) == 0:
                self._log("No presence CSV found; skipping presence assignment.")
                idx_csv = os.path.join(out_dir, "grid_presence.csv")
                fishnet_clip.drop(columns="geometry").to_csv(idx_csv, index=False, encoding="utf-8-sig")
                self._log(f"Wrote (no presence): {idx_csv}")
            else:
                csv_path = csvs[0]
                self._log(f"Reading presence CSV: {os.path.basename(csv_path)}")
                points = read_presence_csv_to_points(csv_path, log=self._log)

                # Save presence points for inspection
                pts_shp = os.path.join(out_dir, "presence_points.shp")
                points.to_file(pts_shp, driver="ESRI Shapefile")
                self._log(f"Wrote: {pts_shp}")

                # Join with intersects + de-dup
                grid_with_occ, summary = assign_presence_to_grid(fishnet_clip, points, log=self._log, predicate="intersects")

                # Save enriched fishnet
                fishnet_clip_occ_gpkg = os.path.join(out_dir, "fishnet_clip_with_occ.gpkg")
                fishnet_clip_occ_shp  = os.path.join(out_dir, "fishnet_clip_with_occ.shp")
                grid_with_occ.to_file(fishnet_clip_occ_gpkg, driver="GPKG")
                grid_with_occ.to_file(fishnet_clip_occ_shp,  driver="ESRI Shapefile")
                self._log(f"Wrote: {fishnet_clip_occ_gpkg}")
                self._log(f"Wrote: {fishnet_clip_occ_shp}")

                # Save CSV summary (SiteID, n_points, occ)
                idx_csv = os.path.join(out_dir, "grid_presence.csv")
                summary.to_csv(idx_csv, index=False, encoding="utf-8-sig")
                self._log(f"Wrote: {idx_csv}")

            self._set_progress(100, "Done")
            messagebox.showinfo("Done", f"All results written to:\n{out_dir}")

        except Exception as e:
            self._log("An error occurred:\n" + traceback.format_exc())
            messagebox.showerror("Error", str(e))
        finally:
            self._set_running(False)


if __name__ == "__main__":
    try:
        app = App()
        app.mainloop()
    except Exception as e:
        print("ERROR:", e)
        sys.exit(1)