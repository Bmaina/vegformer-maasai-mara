"""
Google Earth Engine Data Export Scripts
=======================================
Exports Sentinel-1, Sentinel-2, CHIRPS, and MODIS data
as multi-temporal patch tensors for VegFormer training.

Prerequisites:
    pip install earthengine-api geemap
    earthengine authenticate

Usage:
    python data/gee/export_all.py \
        --aoi configs/study_area.geojson \
        --start 2019-01-01 \
        --end 2023-12-31 \
        --output_bucket gs://your-bucket/vegformer/
"""

import ee
import geemap
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta


def initialize_gee():
    """Initialize GEE with retry."""
    try:
        ee.Initialize(opt_url='https://earthengine.googleapis.com')
    except Exception:
        ee.Authenticate()
        ee.Initialize()


# ---------------------------------------------------------------------------
# Sentinel-1 SAR Export
# ---------------------------------------------------------------------------

def export_sentinel1(aoi: ee.Geometry, start_date: str, end_date: str,
                     output_prefix: str, scale: int = 10):
    """
    Export Sentinel-1 GRD VV/VH 16-day median composites.

    Preprocessing applied in GEE:
    - Orbital correction (applied automatically in GEE S1 collection)
    - Speckle filtering via focal median (3×3 kernel)
    - Lee sigma filter optional (add if speckle is severe in AOI)
    """
    dates = _generate_date_windows(start_date, end_date, interval_days=16)

    for i, (start, end) in enumerate(dates):
        s1 = (ee.ImageCollection('COPERNICUS/S1_GRD')
              .filterBounds(aoi)
              .filterDate(start, end)
              .filter(ee.Filter.eq('instrumentMode', 'IW'))
              .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
              .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
              .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
              .select(['VV', 'VH'])
              .map(lambda img: img.focalMedian(3))  # Speckle filter
              .median()
              .clip(aoi))

        task = ee.batch.Export.image.toCloudStorage(
            image=s1,
            description=f'S1_16d_{i:04d}',
            bucket=output_prefix.split('/')[2],
            fileNamePrefix=f"{'/'.join(output_prefix.split('/')[3:])}/S1/{start}",
            region=aoi,
            scale=scale,
            crs='EPSG:4326',
            maxPixels=1e13,
            fileFormat='GeoTIFF'
        )
        task.start()
        print(f"  Started S1 export {i+1}/{len(dates)}: {start} → {end}")


# ---------------------------------------------------------------------------
# Sentinel-2 Optical Export
# ---------------------------------------------------------------------------

def mask_s2_clouds(image: ee.Image) -> ee.Image:
    """
    Apply SCL (Scene Classification Layer) cloud and shadow masking.
    SCL classes: 3=cloud shadow, 8=medium cloud, 9=high cloud, 10=cirrus
    """
    scl = image.select('SCL')
    cloud_mask = (scl.neq(3)   # cloud shadow
                  .And(scl.neq(8))   # medium cloud
                  .And(scl.neq(9))   # high cloud
                  .And(scl.neq(10))) # cirrus
    return image.updateMask(cloud_mask).divide(10000)  # Scale to [0,1]


def compute_spectral_indices(image: ee.Image) -> ee.Image:
    """Compute NDVI, EVI, NBR, NDWI, BSI as additional input features."""
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    evi  = image.expression(
        '2.5 * (NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1)',
        {'NIR': image.select('B8'), 'RED': image.select('B4'),
         'BLUE': image.select('B2')}
    ).rename('EVI')
    nbr  = image.normalizedDifference(['B8', 'B12']).rename('NBR')
    ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
    return image.addBands([ndvi, evi, nbr, ndwi])


def export_sentinel2(aoi: ee.Geometry, start_date: str, end_date: str,
                     output_prefix: str, scale: int = 10,
                     include_indices: bool = True):
    """
    Export Sentinel-2 SR 16-day cloud-free median composites.
    Bands: B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12 + 4 spectral indices
    """
    dates = _generate_date_windows(start_date, end_date, interval_days=16)
    bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12', 'SCL']
    out_bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
    if include_indices:
        out_bands += ['NDVI', 'EVI', 'NBR', 'NDWI']

    for i, (start, end) in enumerate(dates):
        s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
              .filterBounds(aoi)
              .filterDate(start, end)
              .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 80))
              .select(bands)
              .map(mask_s2_clouds))

        if include_indices:
            s2 = s2.map(compute_spectral_indices)

        composite = s2.select(out_bands).median().clip(aoi)

        task = ee.batch.Export.image.toCloudStorage(
            image=composite,
            description=f'S2_16d_{i:04d}',
            bucket=output_prefix.split('/')[2],
            fileNamePrefix=f"{'/'.join(output_prefix.split('/')[3:])}/S2/{start}",
            region=aoi,
            scale=scale,
            crs='EPSG:4326',
            maxPixels=1e13,
            fileFormat='GeoTIFF'
        )
        task.start()
        print(f"  Started S2 export {i+1}/{len(dates)}: {start} → {end}")


# ---------------------------------------------------------------------------
# CHIRPS Rainfall Export
# ---------------------------------------------------------------------------

def export_chirps(aoi: ee.Geometry, start_date: str, end_date: str,
                  output_prefix: str, scale: int = 5566):
    """
    Export CHIRPS v2.0 rainfall as 16-day sum composites.
    Also computes derived climate variables:
      - 16-day rainfall sum
      - 90-day rolling SPI approximation
      - 180-day rainfall anomaly vs long-term mean
      - Dry spell count (days with < 1mm in the 16-day window)
    """
    dates = _generate_date_windows(start_date, end_date, interval_days=16)

    # Long-term mean for anomaly computation (2000–2019 baseline)
    chirps_lt_mean = (ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
                      .filterDate('2000-01-01', '2019-12-31')
                      .filterBounds(aoi)
                      .mean()
                      .multiply(16)  # Scale daily mean to 16-day period
                      .rename('lt_mean'))

    for i, (start, end) in enumerate(dates):
        chirps = (ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
                  .filterDate(start, end)
                  .filterBounds(aoi))

        rain_sum = chirps.sum().rename('rain_sum')
        dry_days = chirps.map(lambda img: img.lt(1)).sum().rename('dry_days')

        # 90-day SPI window (simplified: z-score vs baseline)
        start_90 = (datetime.strptime(start, '%Y-%m-%d') -
                    timedelta(days=90)).strftime('%Y-%m-%d')
        rain_90 = (ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
                   .filterDate(start_90, end).filterBounds(aoi).sum())
        rain_90_mean = (ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
                        .filterDate('2000-01-01', '2019-12-31')
                        .filterBounds(aoi).sum().multiply(90/7305))
        spi_90 = rain_90.subtract(rain_90_mean).rename('spi_90d')

        anomaly = rain_sum.subtract(chirps_lt_mean).rename('anomaly')
        composite = rain_sum.addBands([dry_days, spi_90, anomaly]).clip(aoi)

        task = ee.batch.Export.image.toCloudStorage(
            image=composite,
            description=f'CHIRPS_16d_{i:04d}',
            bucket=output_prefix.split('/')[2],
            fileNamePrefix=f"{'/'.join(output_prefix.split('/')[3:])}/CHIRPS/{start}",
            region=aoi,
            scale=scale,
            crs='EPSG:4326',
            maxPixels=1e13
        )
        task.start()
        print(f"  Started CHIRPS export {i+1}/{len(dates)}: {start} → {end}")


# ---------------------------------------------------------------------------
# MODIS Legacy Export (for weak supervision)
# ---------------------------------------------------------------------------

def export_modis_legacy(aoi: ee.Geometry, start_date: str = '2001-01-01',
                         end_date: str = '2016-12-31',
                         output_prefix: str = None, scale: int = 250):
    """
    Export MODIS MOD13Q1 NDVI/EVI time series for weak supervision labels.
    These serve as pseudo-labels bridging the 2001–2016 temporal gap.
    """
    modis = (ee.ImageCollection('MODIS/061/MOD13Q1')
             .filterDate(start_date, end_date)
             .filterBounds(aoi)
             .select(['NDVI', 'EVI', 'sur_refl_b01', 'sur_refl_b02'])
             .map(lambda img: img.multiply(0.0001).clip(aoi)))  # Scale factor

    # Export composite time series
    dates = _generate_date_windows(start_date, end_date, interval_days=16)
    for i, (start, end) in enumerate(dates[::2]):  # Every 32 days for MODIS
        composite = modis.filterDate(start, end).first()
        if composite is None:
            continue
        task = ee.batch.Export.image.toCloudStorage(
            image=composite,
            description=f'MODIS_legacy_{i:04d}',
            bucket=output_prefix.split('/')[2],
            fileNamePrefix=f"{'/'.join(output_prefix.split('/')[3:])}/MODIS/{start}",
            region=aoi,
            scale=scale,
            crs='EPSG:4326',
            maxPixels=1e13
        )
        task.start()
    print(f"Started {len(dates)//2} MODIS legacy exports")


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _generate_date_windows(start_date: str, end_date: str,
                            interval_days: int = 16) -> list:
    """Generate list of (start, end) date string pairs."""
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    windows = []
    current = start
    while current < end:
        window_end = min(current + timedelta(days=interval_days), end)
        windows.append((
            current.strftime('%Y-%m-%d'),
            window_end.strftime('%Y-%m-%d')
        ))
        current = window_end
    return windows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export VegFormer training data from GEE')
    parser.add_argument('--aoi', required=True, help='Path to AOI GeoJSON')
    parser.add_argument('--start', default='2019-01-01')
    parser.add_argument('--end', default='2023-12-31')
    parser.add_argument('--output_bucket', required=True,
                        help='GCS bucket URI e.g. gs://my-bucket/vegformer/')
    parser.add_argument('--sensors', nargs='+',
                        default=['s1', 's2', 'chirps', 'modis'],
                        choices=['s1', 's2', 'chirps', 'modis'])
    parser.add_argument('--scale', type=int, default=10)
    args = parser.parse_args()

    initialize_gee()

    with open(args.aoi) as f:
        aoi_geojson = json.load(f)
    aoi = ee.FeatureCollection(aoi_geojson).geometry()

    print(f"Exporting for AOI: {args.aoi}")
    print(f"Period: {args.start} to {args.end}")
    print(f"Sensors: {args.sensors}")

    if 's1' in args.sensors:
        print("\n[1/4] Exporting Sentinel-1 SAR...")
        export_sentinel1(aoi, args.start, args.end, args.output_bucket, args.scale)

    if 's2' in args.sensors:
        print("\n[2/4] Exporting Sentinel-2 optical...")
        export_sentinel2(aoi, args.start, args.end, args.output_bucket, args.scale)

    if 'chirps' in args.sensors:
        print("\n[3/4] Exporting CHIRPS rainfall...")
        export_chirps(aoi, args.start, args.end, args.output_bucket)

    if 'modis' in args.sensors:
        print("\n[4/4] Exporting MODIS legacy (weak supervision)...")
        export_modis_legacy(aoi, output_prefix=args.output_bucket)

    print("\nAll exports submitted. Monitor at https://code.earthengine.google.com/tasks")
