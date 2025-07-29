# CNN-Based Species Distribution Modeling Toolkit 

This toolkit implements a **Convolutional Neural Network (CNN)** - based approach to Species Distribution Modeling (SDM), with satellite imagery as input. It also includes GUI tools for data preprocessing and self-checks.

---

## Contents

| Script                    | Function                                                                 |
|---------------------------|--------------------------------------------------------------------------|
| `CNN_based_SDM.py`        | CNN training + prediction with ResNet-18 based SDM                       |
| `SELF_TEST_B4_RUN.py`     | Self-check + Fishnet grid builder + presence data assignment (GUI)      |
| `IMAGE_CLIPPING.py`       | Raster mosaic & tile clipping tool via shapefile/fishnet (GUI)          |

---

## Installation

### Option 1: Minimal Install (Only CNN model)

For users who only want to run the CNN model (`S3_CNN_based_SDM.py`):

```bash
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # macOS/Linux

pip install -r requirements.txt
```

### Option 2: Full Install (All scripts including GUI tools)

```bash
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # macOS/Linux

pip install -r requirements_gui.txt
```

### Tip: For better compatibility on Windows, consider installing GDAL, geopandas etc. via Conda if pip fails:

conda install -c conda-forge gdal geopandas pyproj shapely rasterio

## Usage
### 1. CNN Model (CNN_based_SDM.py)

Train SDM using image tiles + presence data:
```bash
python CNN_based_SDM.py --mode single --ratio 10 --boots 5
```


Benchmark mode (multiple background ratios):
```bash
python S3_CNN_based_SDM.py --mode benchmark --folder ./your_data --ratios 10,50,100 --boots 5
```

Expected input folder structure:
```
your_data/
├── tile_index.csv            ← includes SiteID, presence, minx, maxx...
├── 0012AB.tif                ← GeoTIFF tiles (one satellite image for each grid cell)
├── 0034XY.tif
└── fishnet_with_SiteID.gpkg  ← (optional) spatial blocks for CV
```

### 2. Self-check + Fishnet Builder (SELF_TEST_B4_RUN.py)
Run before satellite map clipping to ensure:
1. all data exist
2. boundary/fishnet generation
3. presence data assignment from CSV

A GUI will appear. Select your working folder, set grid size, and run.


### 3. Mosaic & Clipping Tool (IMAGE_CLIPPING.py)
Create mosaic from zipped TIFFs and clip with a shapefile/fishnet
Functionality:
1. Mosaic TIFFs from .zip (If you get separated and large .zip satellite maps from, for instance, Google Earth Engine)
2. Clip by fishnet shapefile (each polygon becomes one tile)
3. Export processed fishnet + tile_index.csv

## Project Folder Layout

```
YourProject/
├── S3_CNN_based_SDM.py
├── SELF_TEST_B4_RUN.py
├── IMAGE_CLIPPING.py
├── requirements.txt
├── requirements_gui.txt
├── install.bat / install.sh (optional)
└── README.md
```

## Authors
This program implements a Convolutional Neural Network (CNN) based approach using ResNet-18, a widely used deep learning architecture originally developed for image recognition tasks.

Unlike traditional SDMs that rely heavily on environmental variables such as temperature or precipitation, this method leverages remote sensing imagery to directly extract features from habitat structure.

This is particularly aimed for modeling shorebirds, whose distribution depends more on fine-scale habitat patterns (e.g., wetland morphology, vegetation patches) than on stable climatic zones, due to their highly migratory nature.

The CNN model automatically learns and extracts important visual features from satellite images (e.g., RGB bands and other spectral bands), and uses them as predictors for presence/absence modeling:

    Remote Sensing Imagery  →  Image Features (learned by CNN)  →  SDM Prediction

Feel free to adapt or extend.

