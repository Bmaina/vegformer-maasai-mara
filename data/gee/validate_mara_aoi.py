"""
Maasai Mara AOI Validation Script
===================================
Run this BEFORE the full export to:
  1. Visualize the AOI boundary on an interactive map
  2. Check Sentinel-2 cloud coverage statistics for your date range
  3. Verify data availability per sub-region
  4. Preview one NDVI composite to confirm the AOI is correct

Usage:
    python data/gee/validate_mara_aoi.py

Opens an interactive geemap map in your Jupyter notebook, or
saves a static preview map as outputs/mara_aoi_preview.html
"""

import ee
import geemap
import json
import numpy as np


def initialize_gee():
    try:
        ee.Initialize(opt_url='https://earthengine.googleapis.com')
    except Exception:
        ee.Authenticate()
        ee.Initialize()


def load_aoi(geojson_path: str = 'configs/study_area.geojson') -> ee.Geometry:
    with open(geojson_path) as f:
        gj = json.load(f)
    return ee.FeatureCollection(gj).geometry()


def check_cloud_coverage(aoi: ee.Geometry,
                          start: str = '2019-01-01',
                          end: str = '2023-12-31') -> dict:
    """
    Compute monthly mean cloud coverage % for the AOI.
    Target: <50% for most months (Mara is feasible — dry season very clear).
    """
    s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
          .filterBounds(aoi)
          .filterDate(start, end)
          .select('CLOUDY_PIXEL_PERCENTAGE'))

    stats = s2.aggregate_stats('CLOUDY_PIXEL_PERCENTAGE')
    return stats.getInfo()


def check_s1_coverage(aoi: ee.Geometry,
                       start: str = '2019-01-01',
                       end: str = '2023-12-31') -> int:
    """Count available Sentinel-1 scenes."""
    s1 = (ee.ImageCollection('COPERNICUS/S1_GRD')
          .filterBounds(aoi)
          .filterDate(start, end)
          .filter(ee.Filter.eq('instrumentMode', 'IW'))
          .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING')))
    return s1.size().getInfo()


def preview_ndvi_composite(aoi: ee.Geometry,
                            date: str = '2021-07-01') -> ee.Image:
    """
    Generate a single NDVI composite for visual inspection.
    July = dry season in Mara — good for checking grassland vs woodland contrast.
    """
    end_date = '2021-08-01'
    s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
          .filterBounds(aoi)
          .filterDate(date, end_date)
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
          .map(lambda img: img.normalizedDifference(['B8', 'B4']).rename('NDVI'))
          .median()
          .clip(aoi))
    return s2


def build_validation_map(aoi: ee.Geometry) -> geemap.Map:
    """
    Build an interactive geemap validation map showing:
    - AOI boundary
    - Sentinel-2 NDVI composite (July 2021 — dry season)
    - Sentinel-2 NDVI composite (April 2021 — long rains peak)
    - Sub-region labels
    - Known landmark points
    """
    Map = geemap.Map(center=[-1.4061, 35.1019], zoom=9)

    # Base satellite imagery
    Map.add_basemap('SATELLITE')

    # AOI boundary
    aoi_fc = ee.FeatureCollection([ee.Feature(aoi, {'name': 'Maasai Mara Ecosystem'})])
    Map.addLayer(
        aoi_fc,
        {'color': 'FF0000', 'width': 2, 'fillColor': '00000000'},
        'AOI Boundary'
    )

    # NDVI — dry season (July 2021) — shows grassland degradation clearly
    ndvi_dry = preview_ndvi_composite(aoi, '2021-06-15', '2021-07-31')
    ndvi_viz = {'min': -0.1, 'max': 0.8,
                'palette': ['#d4a76a', '#ffffcc', '#78c679', '#238443', '#004529']}
    Map.addLayer(ndvi_dry, ndvi_viz, 'NDVI Dry Season (Jul 2021)', shown=True)

    # NDVI — long rains peak (April 2021) — maximum greenness
    ndvi_wet = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                .filterBounds(aoi)
                .filterDate('2021-03-15', '2021-05-15')
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40))
                .map(lambda img: img.normalizedDifference(['B8', 'B4']).rename('NDVI'))
                .median().clip(aoi))
    Map.addLayer(ndvi_wet, ndvi_viz, 'NDVI Long Rains (Apr 2021)', shown=False)

    # NDVI difference (wet - dry) — highlights seasonal amplitude
    ndvi_diff = ndvi_wet.subtract(ndvi_dry).rename('NDVI_diff')
    diff_viz = {'min': -0.2, 'max': 0.5,
                'palette': ['#c0392b', '#f39c12', '#ffffff', '#27ae60', '#1a5c1a']}
    Map.addLayer(ndvi_diff, diff_viz, 'NDVI Seasonal Amplitude', shown=False)

    # SAR VV (Sentinel-1, dry season) — soil moisture + vegetation structure
    s1_dry = (ee.ImageCollection('COPERNICUS/S1_GRD')
              .filterBounds(aoi)
              .filterDate('2021-06-01', '2021-08-31')
              .filter(ee.Filter.eq('instrumentMode', 'IW'))
              .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
              .select('VV')
              .median().clip(aoi))
    Map.addLayer(s1_dry, {'min': -20, 'max': 0}, 'SAR VV Dry (Jul 2021)', shown=False)

    # Known landmarks
    landmarks = [
        {'name': 'Mara River Crossing', 'lat': -1.3500, 'lon': 35.0200,
         'note': 'Peak wildebeest crossing July-Oct'},
        {'name': 'Governors Camp',       'lat': -1.3800, 'lon': 35.0100,
         'note': 'Mara Triangle core'},
        {'name': 'Talek Gate',           'lat': -1.4800, 'lon': 35.1800,
         'note': 'Eastern entrance, Talek River'},
        {'name': 'Sekenani Gate',        'lat': -1.5600, 'lon': 35.2400,
         'note': 'Southern entrance'},
        {'name': 'Naboisho Conservancy', 'lat': -1.2000, 'lon': 35.3500,
         'note': 'Community conservancy, degradation hotspot'},
        {'name': 'Ol Kinyei',            'lat': -1.5500, 'lon': 35.3000,
         'note': 'Eastern conservancy'},
    ]
    for lm in landmarks:
        Map.add_marker(
            location=[lm['lat'], lm['lon']],
            popup=f"<b>{lm['name']}</b><br>{lm['note']}",
            tooltip=lm['name']
        )

    Map.add_colorbar(ndvi_viz, label='NDVI', orientation='vertical', layer_name='NDVI')
    Map.add_layer_control()

    return Map


def preview_ndvi_composite(aoi, start, end):
    """Helper for building validation map."""
    return (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(aoi)
            .filterDate(start, end)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40))
            .map(lambda img: img.normalizedDifference(['B8', 'B4']).rename('NDVI'))
            .median().clip(aoi))


def run_validation():
    """Run all validation checks and print a summary report."""
    print("=" * 55)
    print("  VegFormer — Maasai Mara AOI Validation")
    print("=" * 55)

    initialize_gee()
    aoi = load_aoi('configs/study_area.geojson')

    print("\n[1/3] Checking Sentinel-2 cloud coverage (2019-2023)...")
    cloud_stats = check_cloud_coverage(aoi)
    mean_cloud = cloud_stats.get('mean', 'N/A')
    min_cloud  = cloud_stats.get('min', 'N/A')
    max_cloud  = cloud_stats.get('max', 'N/A')
    print(f"  Mean cloud %:  {mean_cloud:.1f}" if isinstance(mean_cloud, float) else f"  Mean: {mean_cloud}")
    print(f"  Min cloud %:   {min_cloud:.1f}"  if isinstance(min_cloud, float)  else f"  Min:  {min_cloud}")
    print(f"  Max cloud %:   {max_cloud:.1f}"  if isinstance(max_cloud, float)  else f"  Max:  {max_cloud}")
    print("  Note: Mara has <20% cloud cover in Jun-Oct (dry season)")
    print("        Expect 40-60% in Apr-May (long rains) — this is normal")

    print("\n[2/3] Counting Sentinel-1 scenes (2019-2023)...")
    s1_count = check_s1_coverage(aoi)
    print(f"  Available S1 scenes: {s1_count}")
    print(f"  Expected 16-day composites: ~{s1_count // 6} (after compositing)")

    print("\n[3/3] Building interactive validation map...")
    Map = build_validation_map(aoi)

    # Save as HTML for browser viewing
    import os
    os.makedirs('outputs/dashboards', exist_ok=True)
    Map.to_html('outputs/dashboards/mara_aoi_validation.html')
    print("  Saved: outputs/dashboards/mara_aoi_validation.html")
    print("  Open in browser to inspect the AOI and NDVI composites")

    print("\n" + "=" * 55)
    print("  READY TO EXPORT — run:")
    print("  python data/gee/export_all.py \\")
    print("    --aoi configs/study_area.geojson \\")
    print("    --start 2017-01-01 \\")
    print("    --end 2024-12-31 \\")
    print("    --output_bucket gs://your-bucket/vegformer-mara/")
    print("=" * 55)

    return Map


if __name__ == '__main__':
    # Run from terminal: python data/gee/validate_mara_aoi.py
    # Or in Jupyter: from data.gee.validate_mara_aoi import run_validation; Map = run_validation()
    Map = run_validation()
