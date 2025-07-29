# -*- coding: utf-8 -*-
"""
Raster Mosaic & Fishnet Clip (GDAL) — self-contained GUI
- Build mosaic from ZIP-contained TIFFs via /vsizip/
- Clip by fishnet polygons, name each tile by SiteID
- Export fishnet copy (GPKG) that includes: SiteID, TILE_PATH, HAS_TILE, presence (copied)
- Build tile_index.csv that also includes presence (1 or NA)
"""

import os
import sys
import zipfile
import random
import string
import threading
import csv
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from contextlib import contextmanager
from osgeo import gdal, ogr, osr

# ===================== Defaults =====================
DEFAULT_ZIP_FOLDER = ""
DEFAULT_MOSAIC_OUT = ""
DEFAULT_FISHNET     = ""
DEFAULT_OUT_DIR     = ""
DEFAULT_FIELD_NAME  = "SiteID"
DEFAULT_SRC_NODATA  = ""
DEFAULT_CACHE_MB    = "1024"  # 1 GB
DEFAULT_CREATION_OPTS = [
    "TILED=YES",
    "BIGTIFF=IF_SAFER",
    "COMPRESS=DEFLATE",
    "PREDICTOR=2",
    "BLOCKXSIZE=512",
    "BLOCKYSIZE=512",
]
# ========================================================================


# -------------- Backward-compatible GDAL env (works without gdal.Env) --------------
@contextmanager
def gdal_env(**kwargs):
    """
    Backport of gdal.Env for older GDAL versions.
    Temporarily set GDAL config options, then restore on exit.
    """
    old_vals = {k: gdal.GetConfigOption(k) for k in kwargs}
    try:
        for k, v in kwargs.items():
            gdal.SetConfigOption(k, str(v) if v is not None else None)
        yield
    finally:
        for k, v in old_vals.items():
            gdal.SetConfigOption(k, v if v is not None else None)


# toolboxes
def _call_with_progress(func, fallback_msg, log, **kwargs):
    """
    Call a GDAL function with 'callback' if supported by your GDAL.
    If TypeError (older GDAL), retry without callback and log an info once.
    """
    try:
        return func(**kwargs)
    except TypeError as te:
        if "callback" in kwargs:
            kwargs.pop("callback", None)
            kwargs.pop("callback_data", None)
            log(f"[Info] {fallback_msg} (your GDAL doesn’t accept 'callback'; running without it).")
            return func(**kwargs)
        raise te


# ---------------------------------- Mosaic helpers ----------------------------------
def list_tiffs_inside_zips(folder, log=print):
    """
    Return GDAL-readable /vsizip/ paths for all TIFF files inside ZIPs.
    Example: /vsizip/C:/path/to/file.zip/inner/path/file.tif
    """
    sources = []
    for fn in os.listdir(folder):
        if not fn.lower().endswith(".zip"):
            continue
        zip_abs = os.path.abspath(os.path.join(folder, fn)).replace("\\", "/")
        try:
            with zipfile.ZipFile(zip_abs) as z:
                tifs = [n for n in z.namelist() if n.lower().endswith((".tif", ".tiff"))]
                for inner in tifs:
                    sources.append(f"/vsizip/{zip_abs}/{inner}")
        except zipfile.BadZipFile:
            log(f"[Warn] Bad ZIP: {zip_abs}")
    return sources


def build_mosaic_vrt_and_translate(
    srcs, vrt_path, out_path, src_nodata=None, vrt_nodata=None,
    cache_mb=1024, creation_options=None, log=print, progress=None
):
    """
    1) Build a VRT (virtual mosaic).
    2) Translate to a tiled/compressed GeoTIFF (streaming, low memory).
    3) Optional: build overviews.
    Progress mapping: BuildVRT 0–30%, Translate 30–90%, Overviews 90–100%.
    """
    if not srcs:
        raise RuntimeError("No TIFFs found inside ZIP files.")
    gdal.UseExceptions()

    cfg = {
        "GDAL_CACHEMAX": str(cache_mb),
        "NUM_THREADS": "ALL_CPUS",
        "GDAL_NUM_THREADS": "ALL_CPUS",
    }
    creation_options = creation_options or DEFAULT_CREATION_OPTS

    cb_buildvrt = progress.make_gdal_cb(0, 30, "BuildVRT") if progress else None
    cb_translate = progress.make_gdal_cb(30, 90, "Translate") if progress else None
    cb_overviews = progress.make_gdal_cb(90, 100, "Overviews") if progress else None

    with gdal_env(**cfg):
        log(f"Building VRT: {vrt_path}")
        vrt = _call_with_progress(
            gdal.BuildVRT,
            "BuildVRT progress not supported by this GDAL",
            log,
            destName=vrt_path,
            srcDSOrSrcDSTab=srcs,
            srcNodata=src_nodata,
            VRTNodata=vrt_nodata,
            resampleAlg="near",
            separate=False,
            callback=cb_buildvrt
        )
        if vrt is None:
            raise RuntimeError("GDAL BuildVRT failed.")
        vrt = None

        log(f"Writing mosaic GeoTIFF (streaming): {out_path}")
        _call_with_progress(
            gdal.Translate,
            "Translate progress not supported by this GDAL",
            log,
            destName=out_path,
            srcDS=vrt_path,
            options=gdal.TranslateOptions(creationOptions=creation_options),
            callback=cb_translate
        )

        ds = gdal.Open(out_path, gdal.GA_Update)
        if ds:
            log("Building overviews: [2,4,8,16]")
            try:
                ds.BuildOverviews("AVERAGE", [2, 4, 8, 16], callback=cb_overviews)
            except TypeError:
                ds.BuildOverviews("AVERAGE", [2, 4, 8, 16])
            ds = None


def mosaic_from_zip(
    zip_folder, out_tif, out_vrt=None, src_nodata=None,
    cache_mb=1024, creation_options=None, log=print, progress=None
):
    """
    Find TIFFs inside ZIPs under 'zip_folder' and mosaic them.
    """
    out_vrt = out_vrt or os.path.join(zip_folder, "mosaic.vrt")
    srcs = list_tiffs_inside_zips(zip_folder, log=log)
    log(f"Found {len(srcs)} TIFF(s) inside ZIPs.")
    for s in srcs[:5]:
        log(f"  {s}")
    if not srcs:
        raise RuntimeError("No TIFFs found.")
    os.makedirs(os.path.dirname(out_tif), exist_ok=True)
    build_mosaic_vrt_and_translate(
        srcs, out_vrt, out_tif,
        src_nodata=src_nodata, vrt_nodata=src_nodata,
        cache_mb=cache_mb, creation_options=creation_options, log=log, progress=progress
    )
    log(f"Mosaic done: {out_tif}")
    if progress:
        progress.set(100, "Mosaic done")
    return out_tif


# ----------------------------------- Clip helpers -----------------------------------
def pick_or_build_raster_source(folder, explicit_path=None, log=print):
    """
    If explicit_path exists: return it.
    Else: search TIF/VRT in folder:
      - Single TIF/VRT -> return it
      - Multiple TIFs -> build a VRT from them and return the VRT path
    """
    if explicit_path and os.path.exists(explicit_path):
        return explicit_path

    tifs, vrts = [], []
    for fn in os.listdir(folder):
        if fn.lower().endswith((".tif", ".tiff")):
            tifs.append(os.path.join(folder, fn))
        elif fn.lower().endswith(".vrt"):
            vrts.append(os.path.join(folder, fn))

    if len(vrts) == 1 and not tifs:
        return vrts[0]
    if len(tifs) == 1 and not vrts:
        return tifs[0]
    if len(tifs) == 0 and len(vrts) == 0:
        raise RuntimeError("No TIF/VRT found in the folder.")

    vrt_path = os.path.join(folder, "auto_mosaic.vrt")
    log(f"Building VRT from {len(tifs)} TIF(s): {vrt_path}")
    gdal.BuildVRT(vrt_path, tifs)
    return vrt_path


def get_raster_info(path):
    """
    Open raster and return (xres, yres, nodata_from_band1, srs_wkt).
    """
    ds = gdal.Open(path)
    if ds is None:
        raise RuntimeError(f"Cannot open raster: {path}")
    gt = ds.GetGeoTransform()
    xres = gt[1]
    yres = abs(gt[5])
    band = ds.GetRasterBand(1)
    nodata = band.GetNoDataValue() if band else None
    srs_wkt = ds.GetProjection()
    ds = None
    return xres, yres, nodata, srs_wkt


def ensure_siteid_field(fishnet_path, field_name="SiteID", log=print):
    """
    Ensure a 'SiteID' string field exists and is filled with unique 6-char
    alphanumeric codes (uppercase+digits) for any missing values.
    """
    ds = ogr.Open(fishnet_path, update=1)
    if ds is None:
        raise RuntimeError(f"Cannot open fishnet for update: {fishnet_path}")
    lyr = ds.GetLayer(0)
    defn = lyr.GetLayerDefn()

    # Create field if missing
    field_idx = defn.GetFieldIndex(field_name)
    if field_idx < 0:
        fd = ogr.FieldDefn(field_name, ogr.OFTString)
        fd.SetWidth(6)
        if lyr.CreateField(fd) != 0:
            raise RuntimeError(f"Failed to create '{field_name}' field.")
        log(f"Created field '{field_name}'")

    # Collect existing IDs
    existing = set()
    lyr.ResetReading()
    for feat in lyr:
        val = feat.GetField(field_name)
        if val:
            existing.add(str(val))
    lyr.ResetReading()

    # Fill missing IDs
    try:
        lyr.StartTransaction()
    except Exception:
        pass  # some drivers ignore transactions

    changed = 0
    for feat in lyr:
        val = feat.GetField(field_name)
        if not val or str(val).strip() == "":
            new_id = _gen_unique_id(existing, 6)
            feat.SetField(field_name, new_id)
            lyr.SetFeature(feat)
            existing.add(new_id)
            changed += 1

    try:
        lyr.CommitTransaction()
    except Exception:
        pass

    lyr.SyncToDisk()
    ds = None
    if changed:
        log(f"Filled {changed} missing {field_name}(s).")
    else:
        log(f"'{field_name}' field already complete.")
    return field_name


def _gen_unique_id(existing_set, length=6):
    alphabet = string.ascii_uppercase + string.digits
    while True:
        code = "".join(random.choices(alphabet, k=length))
        if code not in existing_set:
            return code


def list_siteids(fishnet_path, field_name="SiteID"):
    ds = ogr.Open(fishnet_path)
    if ds is None:
        raise RuntimeError(f"Cannot open fishnet: {fishnet_path}")
    lyr = ds.GetLayer(0)
    ids = [str(feat.GetField(field_name)) for feat in lyr]
    ds = None
    return ids


def clip_by_fishnet_per_cell(
    src_raster, fishnet_path, out_dir, field_name="SiteID",
    creation_opts=None, cache_mb=1024, src_nodata_hint=None, log=print, progress=None
):
    """
    For each polygon in the fishnet, clip using gdal.Warp with a WHERE filter.
    Overall progress = per-feature progress aggregated to 0–100%.
    """
    xres, yres, nodata_from_src, _ = get_raster_info(src_raster)
    dst_nodata = src_nodata_hint if src_nodata_hint is not None else nodata_from_src

    cfg = {
        "GDAL_CACHEMAX": str(cache_mb),
        "NUM_THREADS": "ALL_CPUS",
        "GDAL_NUM_THREADS": "ALL_CPUS",
    }
    creation_opts = creation_opts or DEFAULT_CREATION_OPTS
    os.makedirs(out_dir, exist_ok=True)

    siteids = list_siteids(fishnet_path, field_name=field_name)
    if not siteids:
        raise RuntimeError("No features found in fishnet.")

    N = len(siteids)
    with gdal_env(**cfg):
        for i, sid in enumerate(siteids):
            out_tif = os.path.join(out_dir, f"{sid}.tif")
            if os.path.exists(out_tif):
                log(f"[Skip] exists: {out_tif}")
                if progress:
                    progress.set((i + 1) / N * 100.0, f"Skip {sid}")
                continue

            # overall range for this tile in 0–100
            tile_start = (i / N) * 100.0
            tile_end   = ((i + 1) / N) * 100.0
            cb_tile = progress.make_gdal_cb(tile_start, tile_end, f"Warp {sid}") if progress else None

            where = f"{field_name}='{sid}'"
            warp_opts = gdal.WarpOptions(
                cutlineDSName=fishnet_path,
                cutlineWhere=where,
                cropToCutline=True,
                dstNodata=dst_nodata,
                multithread=True,
                xRes=xres, yRes=yres,
                targetAlignedPixels=True,
                creationOptions=creation_opts
            )
            log(f"Clipping SiteID={sid} -> {out_tif}")
            _call_with_progress(
                gdal.Warp,
                "Warp progress not supported by this GDAL",
                log,
                destNameOrDestDS=out_tif,
                srcDSOrSrcDSTab=src_raster,
                options=warp_opts,
                callback=cb_tile
            )
    log(f"Clipping done. Tiles in: {out_dir}")
    if progress:
        progress.set(100, "Clipping done")


# ------------------ Fishnet export + CSV index (presence included) ------------------
def export_fishnet_with_siteid(
    fishnet_path,
    out_dir,
    field_name="SiteID",
    driver_name="GPKG",       # "GPKG" | "ESRI Shapefile"
    layer_name="fishnet",
    include_only_existing=True,
    tile_ext=".tif",
    match_raster_crs=False,   # if True, reproject output fishnet to raster CRS
    raster_path=None,
    log=print
):
    """
    Export a copy of the fishnet into out_dir, keeping geometry + SiteID,
    and (optionally) only the cells that have a generated tile in out_dir.
    Also writes TILE_PATH, HAS_TILE, and presence (copied from source if present).
    """
    ds_in = ogr.Open(fishnet_path)
    if ds_in is None:
        raise RuntimeError(f"Cannot open fishnet: {fishnet_path}")
    lyr_in = ds_in.GetLayer(0)
    srs_in = lyr_in.GetSpatialRef()
    geom_type = lyr_in.GetGeomType()

    # Discover 'presence' field (case-insensitive / synonyms)
    defn = lyr_in.GetLayerDefn()
    name_map = {defn.GetFieldDefn(i).GetName().lower(): defn.GetFieldDefn(i).GetName()
                for i in range(defn.GetFieldCount())}
    pres_src_name = None
    for cand in ["presence", "pres", "label"]:
        if cand in name_map:
            pres_src_name = name_map[cand]
            break

    # Detect existing tiles
    tile_names = {os.path.splitext(fn)[0] for fn in os.listdir(out_dir) if fn.lower().endswith(tile_ext)}
    log(f"[Index] Found {len(tile_names)} tile(s) in: {out_dir}")

    # Output path & driver
    if driver_name.upper() == "GPKG":
        out_path = os.path.join(out_dir, "fishnet_with_SiteID.gpkg")
        layer_out_name = layer_name
    else:
        out_path = os.path.join(out_dir, "fishnet_with_SiteID.shp")
        layer_out_name = os.path.splitext(os.path.basename(out_path))[0]

    drv = ogr.GetDriverByName(driver_name)
    if drv is None:
        raise RuntimeError(f"OGR driver not found: {driver_name}")
    if os.path.exists(out_path):
        try:
            drv.DeleteDataSource(out_path)
        except Exception:
            pass

    # Target SRS (optional reproject to raster CRS)
    srs_target = srs_in
    coord_tx = None
    if match_raster_crs and raster_path and os.path.exists(raster_path):
        ds_r = gdal.Open(raster_path)
        if ds_r:
            wkt = ds_r.GetProjection()
            if wkt:
                srs_target = osr.SpatialReference()
                srs_target.ImportFromWkt(wkt)
                if srs_in and not srs_in.IsSame(srs_target):
                    coord_tx = osr.CoordinateTransformation(srs_in, srs_target)
            ds_r = None

    ds_out = drv.CreateDataSource(out_path)
    if ds_out is None:
        raise RuntimeError(f"Cannot create output: {out_path}")
    lyr_out = ds_out.CreateLayer(layer_out_name, srs_target, geom_type)
    if lyr_out is None:
        raise RuntimeError("Cannot create output layer.")

    # Fields
    f_site = ogr.FieldDefn(field_name, ogr.OFTString); f_site.SetWidth(64); lyr_out.CreateField(f_site)
    f_path = ogr.FieldDefn("TILE_PATH", ogr.OFTString); f_path.SetWidth(254); lyr_out.CreateField(f_path)
    f_has  = ogr.FieldDefn("HAS_TILE",  ogr.OFTInteger); lyr_out.CreateField(f_has)
    # presence: replicate source type if available; else create string field
    if pres_src_name is not None:
        src_fdef = defn.GetFieldDefn(defn.GetFieldIndex(pres_src_name))
        f_presence = ogr.FieldDefn("presence", src_fdef.GetType())
        lyr_out.CreateField(f_presence)
    else:
        f_presence = ogr.FieldDefn("presence", ogr.OFTString); f_presence.SetWidth(8); lyr_out.CreateField(f_presence)

    # Copy features
    cnt_total, cnt_written = 0, 0
    for feat in lyr_in:
        cnt_total += 1
        sid = feat.GetField(field_name)
        if sid is None:
            continue
        sid = str(sid).strip()
        tile_path = os.path.join(out_dir, f"{sid}{tile_ext}")
        has_tile = int(os.path.exists(tile_path))
        if include_only_existing and not has_tile:
            continue

        geom = feat.GetGeometryRef()
        if geom is None:
            continue
        geom2 = geom.Clone()
        if coord_tx is not None:
            geom2.Transform(coord_tx)

        # presence: copy as-is; keep NULL when absent
        pres_val = feat.GetField(pres_src_name) if pres_src_name else None

        f_out = ogr.Feature(lyr_out.GetLayerDefn())
        f_out.SetField(field_name, sid)
        f_out.SetField("TILE_PATH", tile_path if has_tile else "")
        f_out.SetField("HAS_TILE", has_tile)
        if pres_val is None or (isinstance(pres_val, str) and pres_val.strip() == ""):
            # leave NULL in GPKG; CSV 会转写为 "NA"
            pass
        else:
            f_out.SetField("presence", pres_val)
        f_out.SetGeometry(geom2)
        lyr_out.CreateFeature(f_out)
        f_out = None
        cnt_written += 1

    lyr_out.SyncToDisk()
    ds_out = None
    ds_in = None
    log(f"[Index] Fishnet written: {out_path}  (kept {cnt_written} / {cnt_total} features; presence copied)")
    return out_path


def _wkt_to_epsg(wkt):
    """Best-effort EPSG detection (may return None)."""
    if not wkt:
        return None
    srs = osr.SpatialReference()
    try:
        srs.ImportFromWkt(wkt)
        srs.AutoIdentifyEPSG()
        code = srs.GetAuthorityCode('PROJCS') or srs.GetAuthorityCode('GEOGCS')
        return int(code) if code else None
    except Exception:
        return None


def _read_presence_map(vector_path, field_name="SiteID", pres_candidates=("presence","pres","label"), log=print):
    """Return dict SiteID -> presence ('1' or 'NA')."""
    try:
        ds = ogr.Open(vector_path)
        if ds is None:
            return {}
        lyr = ds.GetLayer(0)
        defn = lyr.GetLayerDefn()
        # resolve presence field name (case-insensitive)
        name_map = {defn.GetFieldDefn(i).GetName().lower(): defn.GetFieldDefn(i).GetName()
                    for i in range(defn.GetFieldCount())}
        pres_name = None
        for c in pres_candidates:
            if c in name_map:
                pres_name = name_map[c]; break
        if pres_name is None:
            return {}
        m = {}
        for feat in lyr:
            sid = feat.GetField(field_name)
            if sid is None: continue
            sid = str(sid).strip()
            val = feat.GetField(pres_name)
            if val in [None, "", " ", "NA"]:
                m[sid] = "NA"
            else:
                # treat numeric 1 as presence; others -> NA
                try:
                    v = int(val)
                    m[sid] = "1" if v == 1 else "NA"
                except Exception:
                    v = str(val).strip()
                    m[sid] = "1" if v == "1" else "NA"
        ds = None
        return m
    except Exception as e:
        log(f"[Warn] read presence map failed: {e}")
        return {}


def build_tiles_index_csv(out_dir, csv_name="tile_index.csv", log=print):
    """
    Scan out_dir for .tif and write a CSV index with footprint & raster metadata + presence.
    Columns: SiteID, path, width, height, xres, yres, minx, miny, maxx, maxy, epsg, presence
    """
    # try to read presence map from the fishnet copy in out_dir
    vec_gpkg = os.path.join(out_dir, "fishnet_with_SiteID.gpkg")
    vec_shp  = os.path.join(out_dir, "fishnet_with_SiteID.shp")
    pres_map = {}
    if os.path.exists(vec_gpkg):
        pres_map = _read_presence_map(vec_gpkg, log=log)
    elif os.path.exists(vec_shp):
        pres_map = _read_presence_map(vec_shp, log=log)

    rows = []
    tifs = [fn for fn in os.listdir(out_dir) if fn.lower().endswith(".tif")]
    for fn in tifs:
        sid = os.path.splitext(fn)[0]
        path = os.path.join(out_dir, fn)
        try:
            ds = gdal.Open(path)
            if not ds:
                continue
            gt = ds.GetGeoTransform()
            w, h = ds.RasterXSize, ds.RasterYSize
            x0, y0 = gt[0], gt[3]
            x1 = x0 + gt[1] * w + gt[2] * h
            y1 = y0 + gt[4] * w + gt[5] * h
            minx, maxx = (x0, x1) if x0 <= x1 else (x1, x0)
            miny, maxy = (y1, y0) if y1 <= y0 else (y0, y1)
            epsg = _wkt_to_epsg(ds.GetProjection())
            presence = pres_map.get(sid, "NA")
            rows.append([sid, path, w, h, gt[1], gt[5], minx, miny, maxx, maxy, epsg or "", presence])
            ds = None
        except Exception as e:
            log(f"[Warn] index for {fn} failed: {e}")

    out_csv = os.path.join(out_dir, csv_name)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["SiteID","path","width","height","xres","yres","minx","miny","maxx","maxy","epsg","presence"])
        writer.writerows(rows)

    log(f"[Index] CSV written: {out_csv}  ({len(rows)} rows, presence included)")
    return out_csv


# -------------------------------------- GUI --------------------------------------
class UIProgress:
    """
    Bridge GDAL progress (0..1) to Tkinter Progressbar (0..100) safely from worker threads.
    """
    def __init__(self, app):
        self.app = app

    def set(self, pct, label=None):
        """
        Set absolute percentage [0..100].
        """
        def _update():
            self.app.progress['value'] = max(0.0, min(100.0, pct))
            if label is not None:
                self.app.progress_label.config(text=f"{label}  {self.app.progress['value']:.1f}%")
        self.app.after(0, _update)

    def make_gdal_cb(self, start_pct, end_pct, label_prefix):
        """
        Create a GDAL callback mapping 0..1 to [start_pct..end_pct] in UI.
        """
        span = max(0.0, end_pct - start_pct)
        def _cb(complete, message=None, data=None):
            # 'complete' is in [0..1]
            try:
                frac = float(complete)
            except Exception:
                frac = 0.0
            pct = start_pct + span * max(0.0, min(1.0, frac))
            self.set(pct, label_prefix)
            return 1  # continue
        return _cb


# GUI

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Raster Mosaic & Fishnet Clip (GDAL)")
        self.geometry("920x640")

        # Inputs
        self.zip_folder = self._add_path_row("ZIP folder:", DEFAULT_ZIP_FOLDER, is_dir=True)
        self.mosaic_out = self._add_path_row("Mosaic output (tif):", DEFAULT_MOSAIC_OUT, save_file=True, filetypes=[("GeoTIFF","*.tif")])

        self.fishnet     = self._add_path_row("Fishnet vector:", DEFAULT_FISHNET, is_file=True, filetypes=[("Vector","*.shp *.gpkg *.geojson *.json"), ("All","*.*")])
        self.raster_clip = self._add_path_row("Raster for clipping (optional):", "", is_file=True, filetypes=[("Raster","*.tif *.vrt"), ("All","*.*")])
        self.out_dir     = self._add_path_row("Clip output folder:", DEFAULT_OUT_DIR, is_dir=True)

        # Options
        opt_frame = ttk.LabelFrame(self, text="Options")
        opt_frame.pack(fill="x", padx=10, pady=6)

        self.field_name = self._add_labeled_entry(opt_frame, "ID field name:", DEFAULT_FIELD_NAME, width=12)
        self.src_nodata = self._add_labeled_entry(opt_frame, "Source NoData (optional):", DEFAULT_SRC_NODATA, width=8)
        self.cache_mb   = self._add_labeled_entry(opt_frame, "GDAL cache (MB):", DEFAULT_CACHE_MB, width=8)

        # Buttons
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill="x", padx=10, pady=6)
        self.btn_mosaic = ttk.Button(btn_frame, text="Build Mosaic", command=self.run_mosaic)
        self.btn_clip   = ttk.Button(btn_frame, text="Clip Only", command=self.run_clip)
        self.btn_all    = ttk.Button(btn_frame, text="Mosaic then Clip (ALL)", command=self.run_all)
        self.btn_mosaic.pack(side="left", padx=5)
        self.btn_clip.pack(side="left", padx=5)
        self.btn_all.pack(side="left", padx=5)

        # Progress (determinate)
        prog_frame = ttk.Frame(self)
        prog_frame.pack(fill="x", padx=10, pady=4)
        self.progress = ttk.Progressbar(prog_frame, mode="determinate", maximum=100.0)
        self.progress.pack(fill="x", side="left", expand=True)
        self.progress_label = ttk.Label(prog_frame, text="Ready  0.0%")
        self.progress_label.pack(side="left", padx=8)

        # Log
        log_frame = ttk.LabelFrame(self, text="Run log")
        log_frame.pack(fill="both", expand=True, padx=10, pady=6)
        self.txt = tk.Text(log_frame, wrap="word")
        self.txt.pack(fill="both", expand=True)
        self._log(f"GDAL version: {gdal.VersionInfo('--version')}")

        self._set_running(False)

    # UI helpers
    def _add_path_row(self, label, default, is_dir=False, is_file=False, save_file=False, filetypes=None):
        frame = ttk.Frame(self)
        frame.pack(fill="x", padx=10, pady=2)
        ttk.Label(frame, text=label, width=28).pack(side="left")
        var = tk.StringVar(value=default)
        entry = ttk.Entry(frame, textvariable=var)
        entry.pack(side="left", fill="x", expand=True, padx=6)
        def browse():
            if is_dir:
                p = filedialog.askdirectory(initialdir=default or os.getcwd())
            elif save_file:
                p = filedialog.asksaveasfilename(defaultextension=".tif", filetypes=filetypes or [("All","*.*")])
            elif is_file:
                p = filedialog.askopenfilename(filetypes=filetypes or [("All","*.*")])
            else:
                p = filedialog.askopenfilename()
            if p:
                var.set(p)
        ttk.Button(frame, text="Browse…", command=browse).pack(side="left")
        return var

    def _add_labeled_entry(self, parent, label, default, width=10):
        frame = ttk.Frame(parent)
        frame.pack(side="left", padx=6, pady=4)
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
        for b in (self.btn_mosaic, self.btn_clip, self.btn_all):
            b.configure(state=state)
        if not running:
            # Reset progress when idle
            self.progress['value'] = 0
            self.progress_label.config(text="Ready  0.0%")

    # Actions (run in background threads)
    def run_mosaic(self):
        zf = self.zip_folder.get().strip()
        out_tif = self.mosaic_out.get().strip()
        src_nodata = self.src_nodata.get().strip()
        cache_mb = int(self.cache_mb.get().strip() or "1024")
        src_nodata_val = float(src_nodata) if src_nodata != "" else None
        ui_prog = UIProgress(self)

        def worker():
            try:
                self._log(f"Starting mosaic: ZIP folder = {zf}")
                mosaic_from_zip(
                    zip_folder=zf,
                    out_tif=out_tif,
                    out_vrt=os.path.join(os.path.dirname(out_tif), "mosaic.vrt"),
                    src_nodata=src_nodata_val,
                    cache_mb=cache_mb,
                    creation_options=DEFAULT_CREATION_OPTS,
                    log=self._log,
                    progress=ui_prog
                )
                messagebox.showinfo("Done", f"Mosaic written:\n{out_tif}")
            except Exception as e:
                self._log(f"[ERROR] {e}")
                messagebox.showerror("Error", str(e))
            finally:
                self._set_running(False)

        if not os.path.isdir(zf):
            messagebox.showerror("Error", "ZIP folder does not exist.")
            return
        self._set_running(True)
        threading.Thread(target=worker, daemon=True).start()

    def run_clip(self):
        fishnet = self.fishnet.get().strip()
        out_dir = self.out_dir.get().strip()
        raster = self.raster_clip.get().strip()
        field = self.field_name.get().strip() or "SiteID"
        src_nodata = self.src_nodata.get().strip()
        cache_mb = int(self.cache_mb.get().strip() or "1024")
        src_nodata_val = float(src_nodata) if src_nodata != "" else None
        ui_prog = UIProgress(self)

        if raster == "":
            raster = self.mosaic_out.get().strip()

        def worker():
            try:
                if not os.path.exists(raster):
                    folder = os.path.dirname(raster) if raster else DEFAULT_ZIP_FOLDER
                    self._log(f"[Info] Raster not found; trying to auto-pick in folder: {folder}")
                    raster_resolved = pick_or_build_raster_source(folder, explicit_path=None, log=self._log)
                else:
                    raster_resolved = raster
                self._log(f"Raster source for clipping: {raster_resolved}")

                ensure_siteid_field(fishnet, field_name=field, log=self._log)
                clip_by_fishnet_per_cell(
                    src_raster=raster_resolved,
                    fishnet_path=fishnet,
                    out_dir=out_dir,
                    field_name=field,
                    creation_opts=DEFAULT_CREATION_OPTS,
                    cache_mb=cache_mb,
                    src_nodata_hint=src_nodata_val,
                    log=self._log,
                    progress=ui_prog
                )

                export_fishnet_with_siteid(
                    fishnet_path=fishnet,
                    out_dir=out_dir,
                    field_name=field,
                    driver_name="GPKG",  # or "ESRI Shapefile"
                    include_only_existing=True,  # only cells that have a tile
                    match_raster_crs=False,      # set True to reproject fishnet to raster CRS
                    raster_path=raster_resolved, # used when match_raster_crs=True
                    log=self._log
                )
                build_tiles_index_csv(out_dir, csv_name="tile_index.csv", log=self._log)

                messagebox.showinfo("Done", f"Clipping finished. Tiles in:\n{out_dir}")
            except Exception as e:
                self._log(f"[ERROR] {e}")
                messagebox.showerror("Error", str(e))
            finally:
                self._set_running(False)

        if not os.path.isfile(fishnet):
            messagebox.showerror("Error", "Invalid fishnet path.")
            return
        if not os.path.isdir(out_dir):
            try:
                os.makedirs(out_dir, exist_ok=True)
            except Exception as e:
                messagebox.showerror("Error", f"Cannot create output folder: {e}")
                return

        self._set_running(True)
        threading.Thread(target=worker, daemon=True).start()

    def run_all(self):
        zf = self.zip_folder.get().strip()
        out_tif = self.mosaic_out.get().strip()
        fishnet = self.fishnet.get().strip()
        out_dir = self.out_dir.get().strip()
        field = self.field_name.get().strip() or "SiteID"
        src_nodata = self.src_nodata.get().strip()
        cache_mb = int(self.cache_mb.get().strip() or "1024")
        src_nodata_val = float(src_nodata) if src_nodata != "" else None
        ui_prog = UIProgress(self)

        def worker():
            try:
                self._log("Starting ALL: mosaic then clip")
                mosaic_from_zip(
                    zip_folder=zf,
                    out_tif=out_tif,
                    out_vrt=os.path.join(os.path.dirname(out_tif), "mosaic.vrt"),
                    src_nodata=src_nodata_val,
                    cache_mb=cache_mb,
                    creation_options=DEFAULT_CREATION_OPTS,
                    log=self._log,
                    progress=ui_prog
                )
                ensure_siteid_field(fishnet, field_name=field, log=self._log)
                clip_by_fishnet_per_cell(
                    src_raster=out_tif,
                    fishnet_path=fishnet,
                    out_dir=out_dir,
                    field_name=field,
                    creation_opts=DEFAULT_CREATION_OPTS,
                    cache_mb=cache_mb,
                    src_nodata_hint=src_nodata_val,
                    log=self._log,
                    progress=ui_prog
                )
                export_fishnet_with_siteid(
                    fishnet_path=fishnet,
                    out_dir=out_dir,
                    field_name=field,
                    driver_name="GPKG",
                    include_only_existing=True,
                    match_raster_crs=False,
                    raster_path=out_tif,
                    log=self._log
                )
                build_tiles_index_csv(out_dir, csv_name="tile_index.csv", log=self._log)

                messagebox.showinfo("Done", f"All done.\nMosaic: {out_tif}\nTiles in: {out_dir}")
            except Exception as e:
                self._log(f"[ERROR] {e}")
                messagebox.showerror("Error", str(e))
            finally:
                self._set_running(False)

        if not os.path.isdir(zf):
            messagebox.showerror("Error", "ZIP folder does not exist.")
            return
        if not os.path.isfile(fishnet):
            messagebox.showerror("Error", "Invalid fishnet path.")
            return
        if not os.path.isdir(os.path.dirname(out_tif)):
            try:
                os.makedirs(os.path.dirname(out_tif), exist_ok=True)
            except Exception as e:
                messagebox.showerror("Error", f"Cannot create mosaic output folder: {e}")
                return
        if not os.path.isdir(out_dir):
            try:
                os.makedirs(out_dir, exist_ok=True)
            except Exception as e:
                messagebox.showerror("Error", f"Cannot create clipping output folder: {e}")
                return

        self._set_running(True)
        threading.Thread(target=worker, daemon=True).start()


if __name__ == "__main__":
    try:
        app = App()
        app.mainloop()
    except Exception as e:
        print("ERROR:", e)
        sys.exit(1)
