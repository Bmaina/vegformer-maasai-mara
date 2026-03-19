# =============================================================================
# NOTEBOOK 02 — MULTI-SENSOR DATA STREAMING (Sentinel-2, Sentinel-1, CHIRPS)
# =============================================================================
#
# PURPOSE
# -------
# Stream Sentinel-2 multispectral, Sentinel-1 SAR, and CHIRPS rainfall
# composites from GEE directly into memory. This notebook builds the full
# multi-sensor dataset that VegFormer uses for training.
#
# WHAT THIS NOTEBOOK DOES
# -----------------------
# 1. Streams 137 sixteen-day Sentinel-2 composites (14 bands + spectral indices)
# 2. Streams 137 sixteen-day Sentinel-1 SAR composites (VV + VH)
# 3. Fixes SAR orbit gaps by combining ascending + descending passes
# 4. Removes SAR specular reflection artifacts (0 dB values)
# 5. Streams 137 sixteen-day CHIRPS rainfall composites (4 climate variables)
# 6. Saves all three sensors as compressed zarr tensors
#
# WHY THESE SENSORS?
# ------------------
# Sentinel-2 (optical):
#   - 10 spectral bands capture vegetation structure, chlorophyll, water content
#   - 4 derived indices (NDVI, EVI, NBR, NDWI) provide additional signal
#   - Limitation: blind during cloud cover (common in Mara rainy seasons)
#
# Sentinel-1 (SAR):
#   - Microwave radar penetrates clouds — active during cloud-covered periods
#   - VV backscatter sensitive to vegetation vertical structure
#   - VH backscatter sensitive to vegetation volume and soil moisture
#   - Provides complementary signal to optical during cloud gaps
#
# CHIRPS rainfall:
#   - Vegetation responds to rainfall with a 2-4 week lag
#   - 90-day SPI captures drought stress accumulation
#   - Dry day count distinguishes intense storms from gentle rains
#
# ASSUMPTIONS
# -----------
# - 16-day composites align S2 and S1 to the same temporal grid
# - 250m scale used for streaming (native: S2=10m, S1=10m, CHIRPS=5km)
# - CHIRPS is exported at 5km (native) then mapped to S2 pixel grid in notebook 03
# - Both ascending and descending S1 orbits are combined to fill data gaps
#   over the Mara (DESCENDING orbit alone misses ~55% of composites)
# - EVI values clipped to [-1, 2] to remove rare computation artifacts
#
# OUTPUT
# ------
# C:/vegformer/data/processed/s2_16day_2019_2024.zarr     (615 MB)
# C:/vegformer/data/processed/s1_16day_2019_2024.zarr      (86 MB)
# C:/vegformer/data/processed/chirps_16day_2019_2024.zarr   (0.4 MB)
# =============================================================================


# -----------------------------------------------------------------------------
# CELL 1 — Environment Setup
# -----------------------------------------------------------------------------
# Initializes GEE, defines AOI and streaming function, generates 16-day
# date windows covering the full 2019–2024 period.
# This cell must run before any other cell in this notebook.

import ee
import numpy as np
import zarr
import requests
import rasterio
from rasterio.io import MemoryFile
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

ee.Initialize(project='dc-site-scout')

PROCESSED_DIR = Path('C:/vegformer/data/processed')
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

aoi_coords = [
    [34.8800, -1.8500], [35.0500, -1.8500], [35.2000, -1.7800],
    [35.3500, -1.6500], [35.4200, -1.5000], [35.4200, -1.1500],
    [35.3000, -1.0500], [35.1500, -1.0000], [34.9500, -1.0000],
    [34.8000, -1.0800], [34.7200, -1.2500], [34.7200, -1.5500],
    [34.7800, -1.7500], [34.8800, -1.8500]
]
aoi = ee.Geometry.Polygon([aoi_coords])

def stream_image_from_gee(image, aoi, scale=250, bands=None):
    if bands:
        image = image.select(bands)
    url = image.getDownloadURL({
        'region': aoi, 'scale': scale,
        'crs': 'EPSG:32736', 'format': 'GEO_TIFF'
    })
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with MemoryFile(response.content) as memfile:
        with memfile.open() as dataset:
            return dataset.read().astype(np.float32)

def date_windows(start, end, days=16):
    windows, current = [], datetime.strptime(start, '%Y-%m-%d')
    end_dt = datetime.strptime(end, '%Y-%m-%d')
    while current < end_dt:
        nxt = min(current + timedelta(days=days), end_dt)
        windows.append((current.strftime('%Y-%m-%d'), nxt.strftime('%Y-%m-%d')))
        current = nxt
    return windows

windows = date_windows('2019-01-01', '2024-12-31')
print(f"EE connected")
print(f"16-day windows: {len(windows)} composites")
print(f"Period: {windows[0][0]} to {windows[-1][1]}")


# -----------------------------------------------------------------------------
# CELL 2 — Sentinel-2 Composite Function
# -----------------------------------------------------------------------------
# Builds one cloud-masked 16-day Sentinel-2 median composite.
#
# Bands exported:
#   B2  Blue  490nm    — atmospheric scattering, water
#   B3  Green 560nm    — vegetation green peak
#   B4  Red   665nm    — chlorophyll absorption
#   B5  RE1   705nm    — red edge — sensitive to chlorophyll content
#   B6  RE2   740nm    — red edge — vegetation stress
#   B7  RE3   783nm    — red edge — LAI estimation
#   B8  NIR   842nm    — vegetation structure (NDVI numerator)
#   B8A RE4   865nm    — narrow NIR — atmospheric correction
#   B11 SWIR1 1610nm   — soil moisture, fire scars
#   B12 SWIR2 2190nm   — senescent vegetation, burned areas
#   NDVI — (B8-B4)/(B8+B4) — overall greenness
#   EVI  — enhanced vegetation index — reduces soil background effects
#   NBR  — (B8-B12)/(B8+B12) — normalized burn ratio
#   NDWI — (B3-B8)/(B3+B8)  — water content / flooding

def mask_s2_clouds(img):
    scl  = img.select('SCL')
    mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
    return img.updateMask(mask).divide(10000)

def get_s2_composite(start_date, end_date):
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

S2_BANDS = ['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12',
            'NDVI','EVI','NBR','NDWI']

# Test one composite
print("Testing S2 composite on April 2021...")
test = get_s2_composite('2021-04-01', '2021-04-16')
arr  = stream_image_from_gee(test, aoi, scale=250, bands=S2_BANDS)
print(f"Shape:          {arr.shape}  — (14 bands, H, W)")
print(f"NDVI range:     {arr[10].min():.3f} to {arr[10].max():.3f}")
print(f"Memory:         {arr.nbytes/1e6:.1f} MB per composite")
print(f"Full stack est: {arr.nbytes/1e6 * len(windows):.0f} MB in RAM")


# -----------------------------------------------------------------------------
# CELL 3 — Stream Full Sentinel-2 Time Series
# -----------------------------------------------------------------------------
# Streams all 137 sixteen-day S2 composites.
# Expected runtime: 20–30 minutes.
#
# The cloud cover threshold is set to 80% to maximize data availability —
# the cloud masking function removes cloud pixels within each image, so
# even a 70% cloudy image contributes valid pixels in the clear areas.
#
# A checkpoint is saved every 20 composites to protect against crashes.
# If this cell is interrupted, restart and check if a checkpoint exists.

s2_stack  = []
s2_dates  = []
s2_failed = []

print(f"Streaming {len(windows)} S2 composites (20-30 minutes)...")
for i, (start, end) in enumerate(tqdm(windows, desc='S2')):
    img = get_s2_composite(start, end)
    try:
        arr = stream_image_from_gee(img, aoi, scale=250, bands=S2_BANDS)
        s2_stack.append(arr)
        s2_dates.append(start)
    except Exception as e:
        fill = np.full_like(s2_stack[-1], np.nan) if s2_stack else np.full((14,377,313), np.nan)
        s2_stack.append(fill)
        s2_dates.append(start)
        s2_failed.append((start, str(e)))

    if (i+1) % 20 == 0:
        np.save(str(PROCESSED_DIR / 's2_checkpoint.npy'), np.stack(s2_stack))
        print(f"  Checkpoint saved at {i+1}/{len(windows)}")

s2_array = np.stack(s2_stack, axis=0)   # (137, 14, 377, 313)
print(f"\nDone. Shape: {s2_array.shape} | Memory: {s2_array.nbytes/1e6:.0f} MB")
print(f"Failed: {len(s2_failed)}")


# -----------------------------------------------------------------------------
# CELL 4 — Fix EVI Outliers and Save Sentinel-2
# -----------------------------------------------------------------------------
# EVI can produce extreme outlier values (up to 1,119,484) when the
# denominator approaches zero — typically at water edges or cloud shadows.
# These are clipped to the physically valid range [-1, 2].
#
# Impact: only 498 pixels out of 226 million (~0.0002%) are affected.

print(f"EVI outliers > 2:   {(s2_array[:, 11] > 2).sum():,}")
print(f"EVI outliers < -1:  {(s2_array[:, 11] < -1).sum():,}")
s2_array[:, 11] = np.clip(s2_array[:, 11], -1.0, 2.0)
print(f"EVI clipped to [-1, 2]")

# Fix failed composites by interpolating from neighbours
for i, (date, _) in enumerate(s2_failed):
    if np.isnan(s2_array[i]).all():
        before = next((j for j in range(i-1,-1,-1) if not np.isnan(s2_array[j]).all()), None)
        after  = next((j for j in range(i+1,len(s2_array)) if not np.isnan(s2_array[j]).all()), None)
        if before is not None and after is not None:
            s2_array[i] = (s2_array[before] + s2_array[after]) / 2
            print(f"  Interpolated {date}")

# Save
z_s2 = zarr.open(str(PROCESSED_DIR / 's2_16day_2019_2024.zarr'), mode='w',
                  shape=s2_array.shape, dtype='float32',
                  chunks=(1, 14, 377, 313))
z_s2[:] = s2_array
z_s2.attrs['dates']  = s2_dates
z_s2.attrs['bands']  = S2_BANDS
z_s2.attrs['scale']  = '250m'
z_s2.attrs['crs']    = 'EPSG:32736'
size = sum(f.stat().st_size for f in (PROCESSED_DIR/'s2_16day_2019_2024.zarr').rglob('*'))
print(f"\nSaved: s2_16day_2019_2024.zarr | {size/1e6:.1f} MB")
print(f"NaNs:  {np.isnan(s2_array).sum():,}")


# -----------------------------------------------------------------------------
# CELL 5 — Sentinel-1 SAR Composite Function
# -----------------------------------------------------------------------------
# SAR (Synthetic Aperture Radar) measures microwave backscatter.
# Unlike optical sensors, SAR is not affected by cloud cover.
#
# Key issue with the Mara:
#   The DESCENDING orbit does not cover the Mara AOI for ~55% of 16-day
#   windows, causing 75 out of 137 composites to fail when using DESCENDING
#   only. The fix is to combine both orbits (no orbit filter) — both passes
#   are valid for vegetation monitoring.
#
# Speckle filtering:
#   SAR images have 'speckle' noise from coherent interference. A 3×3
#   focal median filter reduces speckle while preserving spatial edges.
#
# Units: dB (decibels). Typical values:
#   Dense forest:     -10 to -5 dB (VV)
#   Open grassland:   -15 to -10 dB (VV)
#   Bare soil:        -20 to -15 dB (VV)
#   Water (calm):     -25 to -20 dB (VV) — can produce 0 dB specular return

def get_s1_composite(start_date, end_date):
    """Combines ascending and descending orbits to maximize coverage."""
    return (ee.ImageCollection('COPERNICUS/S1_GRD')
            .filterBounds(aoi)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.eq('instrumentMode', 'IW'))
            .select(['VV','VH'])
            .map(lambda img: img.focalMedian(3))   # speckle filter
            .median()
            .clip(aoi))

# Test
print("Testing S1 composite on April 2021...")
test = get_s1_composite('2021-04-01', '2021-04-16')
arr  = stream_image_from_gee(test, aoi, scale=250, bands=['VV','VH'])
print(f"Shape:      {arr.shape}  — (2 bands, H, W)")
print(f"VV range:   {arr[0].min():.2f} to {arr[0].max():.2f} dB")
print(f"VH range:   {arr[1].min():.2f} to {arr[1].max():.2f} dB")


# -----------------------------------------------------------------------------
# CELL 6 — Stream Full Sentinel-1 Time Series
# -----------------------------------------------------------------------------
# Streams all 137 SAR composites. Faster than S2 (2 bands vs 14).
# Expected runtime: 15–20 minutes.

s1_stack  = []
s1_dates  = []
s1_failed = []

print(f"Streaming {len(windows)} S1 composites (15-20 minutes)...")
for i, (start, end) in enumerate(tqdm(windows, desc='S1')):
    img = get_s1_composite(start, end)
    try:
        arr = stream_image_from_gee(img, aoi, scale=250, bands=['VV','VH'])
        s1_stack.append(arr)
        s1_dates.append(start)
    except Exception as e:
        fill = np.full_like(s1_stack[-1], np.nan) if s1_stack else np.full((2,377,313), np.nan)
        s1_stack.append(fill)
        s1_dates.append(start)
        s1_failed.append((start, str(e)))

    if (i+1) % 20 == 0:
        np.save(str(PROCESSED_DIR / 's1_checkpoint.npy'), np.stack(s1_stack))
        print(f"  Checkpoint saved at {i+1}/{len(windows)}")

s1_array = np.stack(s1_stack, axis=0)   # (137, 2, 377, 313)
print(f"\nDone. Shape: {s1_array.shape} | Memory: {s1_array.nbytes/1e6:.0f} MB")
print(f"Failed: {len(s1_failed)}")


# -----------------------------------------------------------------------------
# CELL 7 — Fix SAR Artifacts and Save Sentinel-1
# -----------------------------------------------------------------------------
# Two SAR-specific artifacts are fixed:
#
# 1. Specular reflection (0.0 dB values):
#    Calm water surfaces return all radar energy away from the sensor,
#    producing a near-zero backscatter value. The Mara River and seasonal
#    wetlands cause this. These are replaced with NaN and interpolated.
#
# 2. Per-pixel temporal interpolation:
#    Same interpolation as notebook 01, applied per SAR band.
#    Fixes ~310,000 artifact values across the full time series.
#
# Valid SAR ranges after fixing:
#   VV: -33 to -4 dB  (physically realistic for savanna vegetation)
#   VH: -46 to -10 dB (physically realistic for savanna vegetation)

print("Fixing SAR specular reflection artifacts (0.0 dB values)...")
s1_array[s1_array == 0.0] = np.nan
print(f"NaN count after zero-fix: {np.isnan(s1_array).sum():,}")

def interpolate_sar(array):
    T, B, H, W = array.shape
    filled = array.copy()
    fixed  = 0
    for b in range(B):
        for r in range(H):
            for c in range(W):
                pixel = array[:, b, r, c]
                nan_m = np.isnan(pixel)
                valid = ~nan_m
                if nan_m.any() and valid.sum() >= 2:
                    f = interp1d(np.where(valid)[0], pixel[valid],
                                 kind='linear', bounds_error=False,
                                 fill_value=(pixel[valid][0], pixel[valid][-1]))
                    filled[nan_m, b, r, c] = f(np.where(nan_m)[0])
                    fixed += nan_m.sum()
    print(f"  Fixed {fixed:,} artifact values")
    return filled

print("Interpolating artifacts (~4 minutes)...")
s1_array = interpolate_sar(s1_array)

z_s1 = zarr.open(str(PROCESSED_DIR / 's1_16day_2019_2024.zarr'), mode='w',
                  shape=s1_array.shape, dtype='float32',
                  chunks=(1, 2, 377, 313))
z_s1[:] = s1_array
z_s1.attrs['dates'] = s1_dates
z_s1.attrs['bands'] = ['VV', 'VH']
z_s1.attrs['units'] = 'dB'
z_s1.attrs['scale'] = '250m'
size = sum(f.stat().st_size for f in (PROCESSED_DIR/'s1_16day_2019_2024.zarr').rglob('*'))
print(f"\nSaved: s1_16day_2019_2024.zarr | {size/1e6:.1f} MB")
print(f"VV range: {s1_array[:,0].min():.2f} to {s1_array[:,0].max():.2f} dB")
print(f"VH range: {s1_array[:,1].min():.2f} to {s1_array[:,1].max():.2f} dB")


# -----------------------------------------------------------------------------
# CELL 8 — CHIRPS Rainfall Composite Function
# -----------------------------------------------------------------------------
# CHIRPS (Climate Hazards Group InfraRed Precipitation with Station data)
# provides daily rainfall estimates at ~5km resolution since 1981.
#
# Four derived climate variables are computed per 16-day window:
#
#   rain_sum:  Total rainfall in mm — direct moisture input
#   dry_days:  Count of days with <1mm — drought stress indicator
#   rain_mean: Mean daily rainfall — intensity measure
#   rain_90d:  Cumulative 90-day rainfall — captures seasonal drought memory
#              (vegetation responds to accumulated moisture deficit, not just
#               current rainfall — 90 days captures the full seasonal cycle)
#
# CHIRPS at 5km over the Mara AOI produces a 19×16 pixel grid —
# much coarser than the 377×313 S2 grid. Each CHIRPS pixel represents
# an area of ~25 km². In notebook 03, each S2 patch is mapped to its
# nearest CHIRPS pixel.

def get_chirps_composite(start_date, end_date):
    chirps    = (ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
                 .filterDate(start_date, end_date)
                 .filterBounds(aoi))
    rain_sum  = chirps.sum().rename('rain_sum')
    dry_days  = chirps.map(lambda img: img.lt(1)).sum().rename('dry_days')
    rain_mean = chirps.mean().rename('rain_mean')
    start_90  = ee.Date(start_date).advance(-90, 'day')
    rain_90d  = (ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
                 .filterDate(start_90, end_date)
                 .filterBounds(aoi).sum().rename('rain_90d'))
    return rain_sum.addBands([dry_days, rain_mean, rain_90d]).clip(aoi)

CHIRPS_BANDS = ['rain_sum', 'dry_days', 'rain_mean', 'rain_90d']

print("Testing CHIRPS composite on April 2021...")
test = get_chirps_composite('2021-04-01', '2021-04-16')
arr  = stream_image_from_gee(test, aoi, scale=5000, bands=CHIRPS_BANDS)
print(f"Shape:          {arr.shape}  — (4 vars, 19 rows, 16 cols)")
print(f"Rain sum range: {arr[0].min():.1f} to {arr[0].max():.1f} mm")
print(f"Dry days range: {arr[1].min():.0f} to {arr[1].max():.0f} days")


# -----------------------------------------------------------------------------
# CELL 9 — Stream and Save CHIRPS
# -----------------------------------------------------------------------------
# CHIRPS streams quickly — only 0.4MB total vs 700MB for optical/SAR.
# Expected runtime: 5 minutes.

chirps_stack = []
chirps_dates = []

print(f"Streaming {len(windows)} CHIRPS composites (~5 minutes)...")
for start, end in tqdm(windows, desc='CHIRPS'):
    img = get_chirps_composite(start, end)
    try:
        arr = stream_image_from_gee(img, aoi, scale=5000, bands=CHIRPS_BANDS)
        chirps_stack.append(arr)
    except Exception as e:
        fill = np.full_like(chirps_stack[-1], np.nan) if chirps_stack else np.full((4,19,16), np.nan)
        chirps_stack.append(fill)
        print(f"  Warning: {start} failed")
    chirps_dates.append(start)

chirps_array = np.stack(chirps_stack, axis=0)   # (137, 4, 19, 16)

z_ch = zarr.open(str(PROCESSED_DIR / 'chirps_16day_2019_2024.zarr'), mode='w',
                  shape=chirps_array.shape, dtype='float32',
                  chunks=(1, 4, 19, 16))
z_ch[:] = chirps_array
z_ch.attrs['dates'] = chirps_dates
z_ch.attrs['bands'] = CHIRPS_BANDS
z_ch.attrs['units'] = 'mm / days / mm / mm'
z_ch.attrs['scale'] = '5000m'

size = sum(f.stat().st_size for f in (PROCESSED_DIR/'chirps_16day_2019_2024.zarr').rglob('*'))
print(f"\nSaved: chirps_16day_2019_2024.zarr | {size/1e6:.2f} MB")
print(f"Shape: {chirps_array.shape}")


# -----------------------------------------------------------------------------
# CELL 10 — Final Summary
# -----------------------------------------------------------------------------
# Verify all three sensors are saved correctly and print a full summary.
# Clean up checkpoint files created during streaming.

import os
for ckpt in ['s2_checkpoint.npy', 's1_checkpoint.npy']:
    p = PROCESSED_DIR / ckpt
    if p.exists():
        os.remove(p)
        print(f"Removed checkpoint: {ckpt}")

print("\n" + "="*55)
print("  NOTEBOOK 02 COMPLETE")
print("="*55)
datasets = [
    ('s2_16day_2019_2024.zarr',     'Sentinel-2',  s2_array.shape),
    ('s1_16day_2019_2024.zarr',     'Sentinel-1',  s1_array.shape),
    ('chirps_16day_2019_2024.zarr', 'CHIRPS',       chirps_array.shape),
]
for fname, name, shape in datasets:
    z    = zarr.open(str(PROCESSED_DIR / fname), mode='r')
    mb   = sum(f.stat().st_size for f in (PROCESSED_DIR/fname).rglob('*')) / 1e6
    nans = np.isnan(z[:]).sum()
    print(f"  {name:<12} shape={str(shape):<25} NaNs={nans:<8} {mb:.1f} MB")
print("="*55)
print("Next: Open 03_patch_extraction.ipynb")
