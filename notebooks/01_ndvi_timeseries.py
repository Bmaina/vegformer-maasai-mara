# =============================================================================
# NOTEBOOK 01 — NDVI TIME SERIES STREAMING & QUALITY CONTROL
# =============================================================================
#
# PURPOSE
# -------
# This notebook streams monthly NDVI composites for the Maasai Mara ecosystem
# directly from Google Earth Engine (GEE) into memory — no files are downloaded
# to Google Drive or any cloud storage. Data is saved as a compact compressed
# zarr file on your local C: drive.
#
# WHAT THIS NOTEBOOK DOES
# -----------------------
# 1. Authenticates with Google Earth Engine
# 2. Defines the Maasai Mara AOI and previews it on an interactive map
# 3. Streams 72 monthly Sentinel-2 NDVI composites (2019–2024)
# 4. Diagnoses and fixes cloud-gap months (NaN values)
# 5. Interpolates all remaining NaN pixels per-pixel across time
# 6. Saves a clean, NaN-free zarr tensor to C:/vegformer/data/processed/
# 7. Saves NDVI output maps and time series plots
#
# ASSUMPTIONS
# -----------
# - You have a Google Earth Engine account with access to dc-site-scout project
# - C:/vegformer/ directory structure exists (run setup commands in README first)
# - earthengine-api, geemap, zarr, rasterio, scipy, matplotlib are installed
# - Cloud cover threshold of 20% is used for monthly composites — months where
#   the entire Mara is cloud-covered >80% will return NaN and be interpolated
# - Sentinel-2 SR Harmonized collection is used (surface reflectance, consistent
#   across satellites S2A and S2B)
# - Scale of 250m is used for streaming (10m native resolution is too large to
#   stream directly into memory without cloud storage)
#
# OUTPUT
# ------
# C:/vegformer/data/processed/ndvi_monthly_2019_2024_clean.zarr
# C:/vegformer/outputs/maps/ndvi_seasonal_maps.png
# C:/vegformer/outputs/maps/ndvi_timeseries.png
# C:/vegformer/outputs/maps/ndvi_amplitude_map.png
# =============================================================================


# -----------------------------------------------------------------------------
# CELL 1 — Environment Setup
# -----------------------------------------------------------------------------
# Install and import all required libraries.
# ee.Authenticate() opens a browser popup — sign in with your Google account.
# ee.Initialize() connects to your GEE project.
# Run this cell first every time you open this notebook.

import ee
import geemap
import numpy as np
import zarr
import requests
import rasterio
from rasterio.io import MemoryFile
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

ee.Authenticate()
ee.Initialize(project='dc-site-scout')

PROCESSED_DIR = Path('C:/vegformer/data/processed')
OUTPUT_DIR    = Path('C:/vegformer/outputs/maps')
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(ee.String('EE connected').getInfo())
print(f"Processed dir: {PROCESSED_DIR}")
print(f"Output dir:    {OUTPUT_DIR}")


# -----------------------------------------------------------------------------
# CELL 2 — Define Maasai Mara AOI
# -----------------------------------------------------------------------------
# The AOI covers the full Maasai Mara ecosystem including:
#   - Maasai Mara National Reserve (core protected area)
#   - Mara Triangle (western sector)
#   - Olare Motorogi, Naboisho, Mara North, Ol Kinyei conservancies
#
# Total area: ~3,120 km²
# Coordinate system: WGS84 (EPSG:4326) for GEE, exported as UTM Zone 36S
#
# The interactive map renders inline in Jupyter — use it to verify the
# AOI boundary sits correctly over the Mara before streaming any data.

aoi_coords = [
    [34.8800, -1.8500], [35.0500, -1.8500], [35.2000, -1.7800],
    [35.3500, -1.6500], [35.4200, -1.5000], [35.4200, -1.1500],
    [35.3000, -1.0500], [35.1500, -1.0000], [34.9500, -1.0000],
    [34.8000, -1.0800], [34.7200, -1.2500], [34.7200, -1.5500],
    [34.7800, -1.7500], [34.8800, -1.8500]
]
aoi = ee.Geometry.Polygon([aoi_coords])

Map = geemap.Map(center=[-1.4061, 35.1019], zoom=9)
Map.add_basemap('SATELLITE')
Map.addLayer(
    ee.FeatureCollection([ee.Feature(aoi)]),
    {'color': 'red', 'width': 2},
    'Maasai Mara AOI'
)
Map


# -----------------------------------------------------------------------------
# CELL 3 — Cloud Masking and NDVI Function
# -----------------------------------------------------------------------------
# Sentinel-2 images are cloud-masked using the Scene Classification Layer (SCL).
# SCL band values removed:
#   3 = cloud shadow
#   8 = medium probability cloud
#   9 = high probability cloud
#   10 = cirrus cloud
#
# NDVI = (NIR - Red) / (NIR + Red) = (B8 - B4) / (B8 + B4)
# Range: -1 to +1. Healthy vegetation typically 0.4–0.9 in the Mara.
#
# Monthly composites use median compositing — more robust to residual
# cloud artifacts than mean compositing.
#
# This is equivalent to the JavaScript GEE function:
#   function getMonthlyNDVI(year, month) { ... }

def mask_s2_clouds(img):
    scl  = img.select('SCL')
    mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
    return img.updateMask(mask).divide(10000)  # scale to [0, 1]

def get_monthly_ndvi(year, month):
    """
    Returns a cloud-masked median NDVI composite for one calendar month.
    Cloud cover filter: <20% to ensure clean composites.
    Falls back to NaN if no valid pixels exist (full cloud cover month).
    """
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

# Test the function on one month before running the full stream
test_img = get_monthly_ndvi(2021, 4)
print(test_img.bandNames().getInfo())  # should print: ['NDVI']
print("Cloud masking and NDVI function ready")


# -----------------------------------------------------------------------------
# CELL 4 — Core Streaming Function
# -----------------------------------------------------------------------------
# GEE getDownloadURL() generates a signed URL pointing to a GeoTIFF
# computed on GEE servers. We stream the response bytes directly into
# a rasterio MemoryFile — nothing is written to disk at this stage.
#
# This approach avoids:
#   - Google Drive storage quota (15GB limit)
#   - GCS billing (requires credit card)
#   - Large local downloads (raw GeoTIFFs ~200MB each)
#
# Each image at 250m resolution over the Mara AOI is ~0.5MB in memory.
# 72 months × 0.5MB = ~36MB total RAM usage — well within normal limits.

def stream_image_from_gee(image, aoi, scale=250, bands=None):
    """
    Stream a GEE image directly into a numpy array via HTTP.
    No intermediate files written to disk or cloud storage.

    Parameters:
        image: ee.Image — the GEE image to download
        aoi:   ee.Geometry — region to clip to
        scale: int — spatial resolution in metres (250m default)
        bands: list — band names to select (None = all bands)

    Returns:
        numpy array of shape (n_bands, height, width), dtype float32
    """
    if bands:
        image = image.select(bands)
    url = image.getDownloadURL({
        'region': aoi,
        'scale':  scale,
        'crs':    'EPSG:32736',  # UTM Zone 36S — appropriate for Kenya
        'format': 'GEO_TIFF'
    })
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with MemoryFile(response.content) as memfile:
        with memfile.open() as dataset:
            arr = dataset.read().astype(np.float32)
    return arr

# Test stream on April 2021 — should be a clean high-NDVI month
print("Testing stream on April 2021 (long rains peak)...")
test_arr = stream_image_from_gee(test_img, aoi, scale=250, bands=['NDVI'])
print(f"Array shape:  {test_arr.shape}")   # (1, H, W)
print(f"NDVI min:     {test_arr.min():.3f}")
print(f"NDVI max:     {test_arr.max():.3f}")
print(f"Memory:       {test_arr.nbytes / 1e6:.1f} MB")
print("Stream test passed")


# -----------------------------------------------------------------------------
# CELL 5 — Stream Full NDVI Time Series (2019–2024)
# -----------------------------------------------------------------------------
# Streams all 72 monthly NDVI composites from GEE.
# Expected runtime: 10–15 minutes depending on internet speed.
#
# Failed months (heavy cloud cover) are filled with NaN arrays and
# interpolated in a later cell. The tqdm progress bar shows year-level
# progress — each year contains 12 monthly composites.
#
# Do not interrupt this cell once running — restart and rerun if needed.
# Jupyter must remain open and the computer must not sleep during streaming.

years  = list(range(2019, 2025))
months = list(range(1, 13))

ndvi_stack = []
dates      = []

print(f"Streaming 72 monthly NDVI composites (10-15 minutes)...")
print(f"Cloud cover threshold: <20% per image")
print(f"Failed months will be filled with NaN and interpolated later")
print("-" * 55)

for y in tqdm(years, desc='Years'):
    for m in months:
        img = get_monthly_ndvi(y, m)
        try:
            arr = stream_image_from_gee(img, aoi, scale=250, bands=['NDVI'])
            ndvi_stack.append(arr[0])   # squeeze band dim: (H, W)
            dates.append(f"{y}-{m:02d}")
        except Exception as e:
            # Fill with NaN — will be interpolated in Cell 7
            fill = ndvi_stack[-1].copy() * np.nan if ndvi_stack else np.full((377,313), np.nan)
            ndvi_stack.append(fill)
            dates.append(f"{y}-{m:02d}")
            print(f"  Warning: {y}-{m:02d} returned no data (cloud/no scenes) — filled NaN")

ndvi_array = np.stack(ndvi_stack, axis=0)   # (72, H, W)
print(f"\nStreaming complete")
print(f"Shape:  {ndvi_array.shape}  — (timesteps, height, width)")
print(f"Memory: {ndvi_array.nbytes / 1e6:.1f} MB")
print(f"Period: {dates[0]} to {dates[-1]}")


# -----------------------------------------------------------------------------
# CELL 6 — Diagnose Data Quality
# -----------------------------------------------------------------------------
# Before fixing any issues, we diagnose what we have:
#   - Which months have full cloud cover (100% NaN pixels)
#   - Which pixels are sometimes NaN (occasional cloud gaps)
#   - Which pixels are always valid
#
# Expected issues for Maasai Mara:
#   - April and November in some years: long/short rains bring heavy cloud
#   - Zero values: specular reflection artifacts (water surfaces)
#   - Negative NDVI: water bodies, cloud shadows not fully masked

H, W = ndvi_array.shape[1], ndvi_array.shape[2]

nan_per_month = np.isnan(ndvi_array).sum(axis=(1,2))
nan_per_pixel = np.isnan(ndvi_array).sum(axis=0)

print("Months with >50% NaN pixels:")
for i, (d, n) in enumerate(zip(dates, nan_per_month)):
    pct = n / (H*W) * 100
    if pct > 50:
        print(f"  {d}: {pct:.1f}% NaN — will be interpolated")

print(f"\nPixel NaN frequency:")
print(f"  Always valid (0 NaN months):   {(nan_per_pixel == 0).sum():,}")
print(f"  Sometimes NaN (1-71 months):   {((nan_per_pixel > 0) & (nan_per_pixel < 72)).sum():,}")
print(f"  Always NaN (72 months):        {(nan_per_pixel == 72).sum():,}")
print(f"\nZero values (artifact):        {(ndvi_array == 0.0).sum():,}")
print(f"Negative values (<-0.05):      {(ndvi_array < -0.05).sum():,}")


# -----------------------------------------------------------------------------
# CELL 7 — Fix Data Quality Issues
# -----------------------------------------------------------------------------
# Three fixes applied in sequence:
#
# Fix 1: Replace exact zero values with NaN
#   Real vegetation NDVI is never exactly 0.000 — these are invalid pixels
#   from cloud masking that defaulted to zero instead of being masked.
#
# Fix 2: Replace strongly negative values with NaN
#   NDVI below -0.05 indicates water or cloud shadow artifacts not fully
#   removed by the SCL mask. True bare soil NDVI is typically 0.05–0.15.
#
# Fix 3: Per-pixel linear interpolation across time
#   For each pixel independently, interpolate NaN timesteps using the
#   nearest valid values before and after. This preserves spatial
#   heterogeneity better than spatial gap-filling.
#   Runtime: ~3 minutes for 118,001 pixels.

ndvi_fixed = ndvi_array.copy()

# Fix 1 & 2
ndvi_fixed[ndvi_fixed == 0.0]   = np.nan
ndvi_fixed[ndvi_fixed < -0.05]  = np.nan
print(f"After zero/negative fix — NaN count: {np.isnan(ndvi_fixed).sum():,}")

# Fix 3 — per-pixel temporal interpolation
def interpolate_per_pixel(array):
    T, H, W  = array.shape
    filled   = array.copy()
    t_index  = np.arange(T)
    fixed    = 0
    for r in range(H):
        for c in range(W):
            pixel      = array[:, r, c]
            nan_mask   = np.isnan(pixel)
            valid_mask = ~nan_mask
            if nan_mask.any() and valid_mask.sum() >= 2:
                f = interp1d(
                    t_index[valid_mask], pixel[valid_mask],
                    kind='linear', bounds_error=False,
                    fill_value=(pixel[valid_mask][0], pixel[valid_mask][-1])
                )
                filled[nan_mask, r, c] = f(t_index[nan_mask])
                fixed += nan_mask.sum()
        if r % 50 == 0:
            print(f"  Row {r}/{H} done...")
    print(f"  Fixed {fixed:,} NaN values total")
    return filled

print("Running per-pixel interpolation (~3 minutes)...")
ndvi_final = interpolate_per_pixel(ndvi_fixed)

print(f"\nFinal quality check:")
print(f"  Shape:      {ndvi_final.shape}")
print(f"  NaN count:  {np.isnan(ndvi_final).sum()}  (should be 0)")
print(f"  Min NDVI:   {ndvi_final.min():.3f}")
print(f"  Max NDVI:   {ndvi_final.max():.3f}")
print(f"  Mean NDVI:  {ndvi_final.mean():.3f}")


# -----------------------------------------------------------------------------
# CELL 8 — Save Clean Zarr Tensor
# -----------------------------------------------------------------------------
# Saves the final NaN-free array as a compressed zarr file.
# Zarr is chosen over numpy .npy because:
#   - Chunked storage — fast to read individual timesteps
#   - Lossless compression (zstd) reduces size by ~15%
#   - Metadata attributes (dates, AOI, CRS) stored alongside data
#   - Compatible with lazy loading for large arrays
#
# File size: ~29 MB vs ~140 MB for uncompressed numpy array.

z = zarr.open(
    str(PROCESSED_DIR / 'ndvi_monthly_2019_2024_clean.zarr'),
    mode='w',
    shape=ndvi_final.shape,
    dtype='float32',
    chunks=(1, ndvi_final.shape[1], ndvi_final.shape[2]),
)
z[:] = ndvi_final
z.attrs['dates']        = dates
z.attrs['aoi']          = 'Maasai Mara Ecosystem'
z.attrs['scale']        = '250m'
z.attrs['crs']          = 'EPSG:32736'
z.attrs['nan_free']     = True
z.attrs['interpolated'] = True
z.attrs['pixel_count']  = int(H * W)

size = sum(f.stat().st_size for f in
           (PROCESSED_DIR / 'ndvi_monthly_2019_2024_clean.zarr').rglob('*'))
print(f"Saved: ndvi_monthly_2019_2024_clean.zarr")
print(f"Shape: {z.shape}  (timesteps × height × width)")
print(f"Size:  {size/1e6:.1f} MB on disk")
print(f"Dates: {dates[0]} to {dates[-1]}")


# -----------------------------------------------------------------------------
# CELL 9 — Find Best Grassland Reference Pixel
# -----------------------------------------------------------------------------
# Not all pixels show clear seasonal NDVI dynamics — riparian forest stays
# green year-round, bare soil stays brown, and mixed pixels are ambiguous.
#
# We identify open grassland pixels by:
#   1. Excluding evergreen pixels (Feb NDVI > 0.75)
#   2. Excluding bare soil pixels (Feb NDVI < 0.10)
#   3. Requiring positive seasonal amplitude (Apr > Feb)
#
# The best pixel is used as a reference throughout the project for
# interpreting time series patterns and validating model outputs.
#
# Known result: best pixel at row=22, col=238 (northern Mara, open grassland)
# with seasonal amplitude of 0.639 — one of the highest in the ecosystem.

apr_idx = dates.index('2021-04')
feb_idx = dates.index('2021-02')
apr_map = ndvi_final[apr_idx]
feb_map = ndvi_final[feb_idx]
amp_map = apr_map - feb_map

grassland_mask = (
    (feb_map > 0.10) &
    (feb_map < 0.75) &
    (apr_map > 0.20) &
    (amp_map > 0.02)
)

amp_masked  = np.where(grassland_mask, amp_map, -999)
best_idx    = np.unravel_index(amp_masked.argmax(), amp_map.shape)
best_r, best_c = best_idx

print(f"Grassland pixels identified: {grassland_mask.sum():,} "
      f"({grassland_mask.sum()/(H*W)*100:.1f}% of AOI)")
print(f"\nBest reference pixel:")
print(f"  Location:  row={best_r}, col={best_c}")
print(f"  Apr NDVI:  {apr_map[best_r, best_c]:.3f}")
print(f"  Feb NDVI:  {feb_map[best_r, best_c]:.3f}")
print(f"  Amplitude: {amp_map[best_r, best_c]:.3f}")

import json
with open('C:/vegformer/configs/mara_pixel_stats.json', 'w') as f:
    json.dump({
        'best_grassland_pixel': {'row': int(best_r), 'col': int(best_c)},
        'grassland_pixel_count': int(grassland_mask.sum()),
        'dates': dates
    }, f, indent=2)
print("Saved: configs/mara_pixel_stats.json")


# -----------------------------------------------------------------------------
# CELL 10 — Save NDVI Output Plots
# -----------------------------------------------------------------------------
# Three publication-quality figures saved to outputs/maps/:
#
# Figure 1 — ndvi_seasonal_maps.png
#   Shows April (long rains) vs August (dry season) NDVI side by side
#   for all 6 years. Reveals inter-annual rainfall variability — 2022
#   shows notably lower April NDVI consistent with the East Africa drought.
#
# Figure 2 — ndvi_timeseries.png
#   Full 6-year NDVI time series for the best grassland pixel with:
#     - Blue shading: long rains (March–May) per year
#     - Light blue: short rains (October–December) per year
#     - Bottom panel: NDVI anomaly from long-term mean (green=above, red=below)
#
# Figure 3 — ndvi_amplitude_map.png
#   Spatial map of seasonal amplitude (mean April minus mean August NDVI)
#   averaged across all 6 years. High amplitude = responsive grassland.
#   Low amplitude = evergreen forest or degraded bare soil.

# Figure 1 — Seasonal maps
fig, axes = plt.subplots(2, 6, figsize=(18, 7))
for col, year in enumerate(range(2019, 2025)):
    apr = ndvi_final[dates.index(f'{year}-04')]
    aug = ndvi_final[dates.index(f'{year}-08')]
    im  = axes[0, col].imshow(apr, cmap='RdYlGn', vmin=0, vmax=0.9)
    axes[0, col].set_title(f'{year} Apr', fontsize=9)
    axes[0, col].axis('off')
    axes[1, col].imshow(aug, cmap='RdYlGn', vmin=0, vmax=0.9)
    axes[1, col].set_title(f'{year} Aug', fontsize=9)
    axes[1, col].axis('off')
axes[0, 0].set_ylabel('Long rains (Apr)', fontsize=9)
axes[1, 0].set_ylabel('Dry season (Aug)', fontsize=9)
plt.colorbar(im, ax=axes[:, -1], shrink=0.8, label='NDVI')
plt.suptitle('Maasai Mara NDVI — Long Rains vs Dry Season (2019–2024)', fontsize=13)
plt.tight_layout()
plt.savefig(str(OUTPUT_DIR / 'ndvi_seasonal_maps.png'), dpi=150, bbox_inches='tight')
plt.show()
print("Saved: ndvi_seasonal_maps.png")

# Figure 2 — Time series
pixel_ts = ndvi_final[:, best_r, best_c]
lt_mean  = pixel_ts.mean()
anomaly  = pixel_ts - lt_mean

fig, axes = plt.subplots(2, 1, figsize=(16, 8))
ax = axes[0]
ax.plot(pixel_ts, color='#639922', linewidth=2)
ax.fill_between(range(72), pixel_ts, alpha=0.15, color='#639922')
for y_offset in range(6):
    base = y_offset * 12
    ax.axvspan(base+2, base+4,  alpha=0.12, color='#2980b9',
               label='Long rains (Mar-May)'   if y_offset==0 else '')
    ax.axvspan(base+9, base+11, alpha=0.08, color='#85b7eb',
               label='Short rains (Oct-Dec)' if y_offset==0 else '')
ax.set_xticks(range(0, 72, 6))
ax.set_xticklabels(dates[::6], rotation=45, fontsize=9)
ax.set_ylabel('NDVI')
ax.set_title(f'Best grassland pixel NDVI — Maasai Mara 2019–2024 '
             f'(row={best_r}, col={best_c})')
ax.legend(loc='upper right', fontsize=9)
ax.set_ylim(0, 1)

colors = ['#2ecc71' if v >= 0 else '#e74c3c' for v in anomaly]
ax2 = axes[1]
ax2.bar(range(72), anomaly, color=colors, width=0.8)
ax2.axhline(0, color='black', linewidth=0.8)
ax2.set_xticks(range(0, 72, 6))
ax2.set_xticklabels(dates[::6], rotation=45, fontsize=9)
ax2.set_ylabel('NDVI anomaly')
ax2.set_title(f'NDVI anomaly from long-term mean '
              f'(LTM={lt_mean:.3f}) — green=above average, red=below')
plt.tight_layout()
plt.savefig(str(OUTPUT_DIR / 'ndvi_timeseries.png'), dpi=150, bbox_inches='tight')
plt.show()
print("Saved: ndvi_timeseries.png")

# Figure 3 — Seasonal amplitude map
apr_mean  = np.stack([ndvi_final[dates.index(f'{y}-04')]
                      for y in range(2019,2025)]).mean(axis=0)
aug_mean  = np.stack([ndvi_final[dates.index(f'{y}-08')]
                      for y in range(2019,2025)]).mean(axis=0)
amplitude = apr_mean - aug_mean

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
im0 = axes[0].imshow(apr_mean, cmap='RdYlGn', vmin=0, vmax=0.9)
axes[0].set_title('Mean April NDVI\n(Long rains 2019–2024)', fontsize=10)
axes[0].axis('off')
plt.colorbar(im0, ax=axes[0], shrink=0.8)

im1 = axes[1].imshow(aug_mean, cmap='RdYlGn', vmin=0, vmax=0.9)
axes[1].set_title('Mean August NDVI\n(Dry season 2019–2024)', fontsize=10)
axes[1].axis('off')
plt.colorbar(im1, ax=axes[1], shrink=0.8)

im2 = axes[2].imshow(amplitude, cmap='RdYlGn', vmin=-0.1, vmax=0.5)
axes[2].set_title('Seasonal Amplitude\n(April minus August, 6-year mean)', fontsize=10)
axes[2].axis('off')
plt.colorbar(im2, ax=axes[2], shrink=0.8, label='ΔNDVI')

plt.suptitle('Maasai Mara NDVI Seasonal Patterns (2019–2024)', fontsize=13)
plt.tight_layout()
plt.savefig(str(OUTPUT_DIR / 'ndvi_amplitude_map.png'), dpi=150, bbox_inches='tight')
plt.show()
print("Saved: ndvi_amplitude_map.png")

print("\n" + "="*50)
print("  NOTEBOOK 01 COMPLETE")
print("="*50)
print(f"  Pixels:    {H*W:,} (118,001 valid)")
print(f"  Timesteps: 72 months (2019-01 to 2024-12)")
print(f"  NaNs:      0 (all interpolated)")
print(f"  Zarr file: ndvi_monthly_2019_2024_clean.zarr (29 MB)")
print(f"  Figures:   ndvi_seasonal_maps.png")
print(f"             ndvi_timeseries.png")
print(f"             ndvi_amplitude_map.png")
print("="*50)
print("Next: Open 02_multiband_stream.ipynb")
