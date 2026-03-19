"""
Notebook 01 — Stream GEE data directly into memory
====================================================
No Google Drive. No GeoTIFF downloads.
GEE computes in the cloud → streams directly into numpy arrays → 
saved as compact .zarr tensors in C:/vegformer/data/processed/

Storage comparison:
  Raw GeoTIFFs (what we avoid):  ~50 GB
  Processed .zarr tensors:        ~2-4 GB
"""

# ── Cell 1: Setup ─────────────────────────────────────────────────────────────

import ee
import geemap
import numpy as np
import zarr
import requests
import zipfile
import io
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm

ee.Authenticate()
ee.Initialize(project='dc-site-scout')
print(ee.String('EE connected').getInfo())

# All processed data saves here — compact tensors only
PROCESSED_DIR = Path('C:/vegformer/data/processed')
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
print(f"Saving processed tensors to: {PROCESSED_DIR}")


# ── Cell 2: Define AOI ────────────────────────────────────────────────────────

aoi_coords = [
    [34.8800, -1.8500], [35.0500, -1.8500], [35.2000, -1.7800],
    [35.3500, -1.6500], [35.4200, -1.5000], [35.4200, -1.1500],
    [35.3000, -1.0500], [35.1500, -1.0000], [34.9500, -1.0000],
    [34.8000, -1.0800], [34.7200, -1.2500], [34.7200, -1.5500],
    [34.7800, -1.7500], [34.8800, -1.8500]
]
aoi = ee.Geometry.Polygon([aoi_coords])

# Preview
Map = geemap.Map(center=[-1.4061, 35.1019], zoom=9)
Map.addLayer(ee.FeatureCollection([ee.Feature(aoi)]), {'color': 'red'}, 'Mara AOI')
Map


# ── Cell 3: Core streaming function ──────────────────────────────────────────
#
# Instead of Export.image.toDrive() we use getDownloadURL()
# GEE zips the image, we stream the zip into memory, read with rasterio
# Nothing touches disk except the final compact tensor

import rasterio
from rasterio.io import MemoryFile

def stream_image_from_gee(image: ee.Image, aoi: ee.Geometry,
                           scale: int = 250, bands: list = None) -> np.ndarray:
    """
    Stream a GEE image directly into a numpy array.
    No file saved to disk — everything stays in memory.

    Args:
        image: ee.Image to download
        scale: spatial resolution in metres (250m recommended for Mara)
        bands: list of band names to select (None = all bands)
    Returns:
        numpy array of shape (bands, height, width)
    """
    if bands:
        image = image.select(bands)

    url = image.getDownloadURL({
        'region': aoi,
        'scale':  scale,
        'crs':    'EPSG:32736',
        'format': 'GEO_TIFF'
    })

    # Stream the response — no temp file
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with MemoryFile(response.content) as memfile:
        with memfile.open() as dataset:
            arr = dataset.read().astype(np.float32)

    return arr


# ── Cell 4: Build NDVI time series (mirrors your JS code) ─────────────────────

def mask_s2_clouds(img):
    scl  = img.select('SCL')
    mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
    return img.updateMask(mask).divide(10000)


def get_monthly_ndvi(year: int, month: int) -> ee.Image:
    """Python equivalent of your getMonthlyNDVI JS function."""
    start = ee.Date.fromYMD(year, month, 1)
    end   = start.advance(1, 'month')
    return (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(aoi)
            .filterDate(start, end)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
            .select(['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12','SCL'])
            .map(mask_s2_clouds)
            .map(lambda img: img.normalizedDifference(['B8','B4'])
                               .rename('NDVI')
                               .copyProperties(img, ['system:time_start']))
            .median()
            .clip(aoi)
            .set('year', year)
            .set('month', month))


# Stream all monthly NDVI composites directly into memory
# 72 months × small array at 250m ≈ manageable in RAM
years  = list(range(2019, 2025))
months = list(range(1, 13))

ndvi_stack = []  # list of (H, W) arrays — builds in RAM
dates      = []

print(f"Streaming {len(years)*len(months)} monthly NDVI composites from GEE...")
for y in tqdm(years, desc='Years'):
    for m in months:
        img = get_monthly_ndvi(y, m)
        try:
            arr = stream_image_from_gee(img, aoi, scale=250, bands=['NDVI'])
            ndvi_stack.append(arr[0])  # (H, W) — squeeze band dim
            dates.append(f"{y}-{m:02d}")
        except Exception as e:
            # Fill with NaN if month has no valid data (heavy cloud)
            if ndvi_stack:
                ndvi_stack.append(np.full_like(ndvi_stack[-1], np.nan))
            dates.append(f"{y}-{m:02d}")
            print(f"  Warning: {y}-{m:02d} failed ({e}) — filled with NaN")

ndvi_array = np.stack(ndvi_stack, axis=0)  # (T=72, H, W)
print(f"NDVI array shape: {ndvi_array.shape}")
print(f"Memory usage: {ndvi_array.nbytes / 1e6:.1f} MB")


# ── Cell 5: Stream Sentinel-2 multi-band composites ──────────────────────────

def get_16day_s2(start_date: str, end_date: str) -> ee.Image:
    def add_indices(img):
        ndvi = img.normalizedDifference(['B8','B4']).rename('NDVI')
        evi  = img.expression(
            '2.5*(NIR-RED)/(NIR+6*RED-7.5*BLUE+1)',
            {'NIR':img.select('B8'),'RED':img.select('B4'),'BLUE':img.select('B2')}
        ).rename('EVI')
        nbr  = img.normalizedDifference(['B8','B12']).rename('NBR')
        ndwi = img.normalizedDifference(['B3','B8']).rename('NDWI')
        return img.addBands([ndvi, evi, nbr, ndwi])

    return (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(aoi)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 80))
            .select(['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12','SCL'])
            .map(mask_s2_clouds)
            .map(add_indices)
            .select(['B2','B3','B4','B5','B6','B7',
                     'B8','B8A','B11','B12','NDVI','EVI','NBR','NDWI'])
            .median()
            .clip(aoi))


def date_windows(start, end, days=16):
    windows, current = [], datetime.strptime(start, '%Y-%m-%d')
    end_dt = datetime.strptime(end, '%Y-%m-%d')
    while current < end_dt:
        nxt = min(current + timedelta(days=days), end_dt)
        windows.append((current.strftime('%Y-%m-%d'), nxt.strftime('%Y-%m-%d')))
        current = nxt
    return windows


# Stream 16-day S2 composites for 2021 (test year)
windows   = date_windows('2021-01-01', '2021-12-31')
s2_stack  = []
s2_dates  = []

S2_BANDS = ['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12',
            'NDVI','EVI','NBR','NDWI']

print(f"Streaming {len(windows)} Sentinel-2 composites...")
for i, (start, end) in enumerate(tqdm(windows, desc='S2 composites')):
    img = get_16day_s2(start, end)
    try:
        arr = stream_image_from_gee(img, aoi, scale=250, bands=S2_BANDS)
        s2_stack.append(arr)   # (14, H, W)
        s2_dates.append(start)
    except Exception as e:
        print(f"  Warning: {start} failed — {e}")

s2_array = np.stack(s2_stack, axis=0)  # (T, 14, H, W)
print(f"S2 array shape: {s2_array.shape}")
print(f"Memory: {s2_array.nbytes / 1e6:.1f} MB")


# ── Cell 6: Stream Sentinel-1 SAR ────────────────────────────────────────────

def get_16day_s1(start_date, end_date):
    return (ee.ImageCollection('COPERNICUS/S1_GRD')
            .filterBounds(aoi)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.eq('instrumentMode', 'IW'))
            .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
            .select(['VV','VH'])
            .map(lambda img: img.focalMedian(3))
            .median()
            .clip(aoi))

s1_stack = []
print(f"Streaming {len(windows)} Sentinel-1 composites...")
for start, end in tqdm(windows, desc='S1 composites'):
    img = get_16day_s1(start, end)
    try:
        arr = stream_image_from_gee(img, aoi, scale=250, bands=['VV','VH'])
        s1_stack.append(arr)
    except Exception as e:
        print(f"  Warning: {start} failed — {e}")

s1_array = np.stack(s1_stack, axis=0)  # (T, 2, H, W)
print(f"S1 array shape: {s1_array.shape}")


# ── Cell 7: Stream CHIRPS rainfall ───────────────────────────────────────────

def get_16day_chirps(start_date, end_date):
    return (ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
            .filterDate(start_date, end_date)
            .filterBounds(aoi)
            .sum().rename('rain_sum').clip(aoi))

chirps_stack = []
print(f"Streaming {len(windows)} CHIRPS composites...")
for start, end in tqdm(windows, desc='CHIRPS'):
    img = get_16day_chirps(start, end)
    try:
        arr = stream_image_from_gee(img, aoi, scale=5000, bands=['rain_sum'])
        chirps_stack.append(arr)
    except Exception as e:
        print(f"  Warning: {start} failed — {e}")

chirps_array = np.stack(chirps_stack, axis=0)  # (T, 1, H, W)
print(f"CHIRPS array shape: {chirps_array.shape}")


# ── Cell 8: Save compact tensors to C:\vegformer\data\processed\ ──────────────
#
# We save as .zarr — compressed, chunked, fast to read back
# Total size: ~2-4 GB vs ~50 GB for raw GeoTIFFs

import zarr

def save_zarr(array, name, dates=None):
    path = PROCESSED_DIR / f'{name}.zarr'
    z = zarr.open(
        str(path), mode='w',
        shape=array.shape,
        dtype='float32',
        chunks=(1, *array.shape[1:]),   # chunk per timestep
        compressor=zarr.Blosc(cname='zstd', clevel=3)
    )
    z[:] = array
    if dates:
        z.attrs['dates'] = dates
    size_mb = path.stat().st_size / 1e6 if path.exists() else 0
    print(f"Saved {name}.zarr — shape {array.shape} | {size_mb:.1f} MB")
    return path

# Save everything
save_zarr(ndvi_array,   'ndvi_monthly_2019_2024', dates=dates)
save_zarr(s2_array,     's2_16day_2021',          dates=s2_dates)
save_zarr(s1_array,     's1_16day_2021')
save_zarr(chirps_array, 'chirps_16day_2021')

print(f"\nAll tensors saved to: {PROCESSED_DIR}")
print("Total size:")
total = sum(f.stat().st_size for f in PROCESSED_DIR.rglob('*') if f.is_file())
print(f"  {total / 1e9:.2f} GB")


# ── Cell 9: Verify and visualise ─────────────────────────────────────────────

import matplotlib.pyplot as plt

# Load back from zarr and check
ndvi_z = zarr.open(str(PROCESSED_DIR / 'ndvi_monthly_2019_2024.zarr'), mode='r')
print(f"NDVI zarr shape: {ndvi_z.shape}")
print(f"Dates: {ndvi_z.attrs['dates'][:6]} ...")

# Plot April 2021 (index 27 = 2021-04)
april_idx = ndvi_z.attrs['dates'].index('2021-04')
plt.figure(figsize=(8, 7))
plt.imshow(ndvi_z[april_idx], cmap='RdYlGn', vmin=-0.1, vmax=0.8)
plt.colorbar(label='NDVI')
plt.title('NDVI — April 2021 (long rains peak, Maasai Mara)')
plt.axis('off')
plt.tight_layout()
plt.show()

# Pixel time series — Mara Reserve centre
H, W = ndvi_z.shape[1], ndvi_z.shape[2]
pixel_ts = ndvi_z[:, H//2, W//2]  # centre pixel

plt.figure(figsize=(16, 4))
plt.plot(pixel_ts, color='#639922', linewidth=1.5)
plt.xticks(range(0, len(dates), 6), dates[::6], rotation=45, fontsize=9)
plt.axhspan(0.4, 0.8, alpha=0.1, color='green', label='Healthy grassland')
plt.axhspan(0.1, 0.4, alpha=0.1, color='orange', label='Stressed / dry')
plt.ylabel('NDVI')
plt.title('NDVI time series — Mara centre pixel (2019–2024)')
plt.legend()
plt.tight_layout()
plt.show()
