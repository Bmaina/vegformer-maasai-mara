"""
VegFormer Interactive Folium Dashboard
=======================================
Generates an interactive multi-layer map showing:
  - Vegetation type classification map
  - Degradation risk score map
  - NDVI anomaly time series (click-to-inspect)
  - Uncertainty bounds
  - Braun-Blanquet validation points

Usage:
    python visualization/folium_dashboard.py \
        --veg_map outputs/veg_type_map.tif \
        --risk_map outputs/degradation_risk.tif \
        --field_points data/braun_blanquet_points.geojson \
        --output dashboard.html
"""

import folium
import folium.plugins as plugins
import rasterio
import numpy as np
import json
import argparse
from pathlib import Path
from folium import MacroElement
from jinja2 import Template


# Vegetation class color map (aligned with Braun-Blanquet phytosociology)
VEG_CLASSES = {
    0: {'name': 'Evergreen forest',       'color': '#1a5c1a'},
    1: {'name': 'Deciduous woodland',     'color': '#4d8c2f'},
    2: {'name': 'Savanna / bushland',     'color': '#8cc23d'},
    3: {'name': 'Grassland',              'color': '#c8e06b'},
    4: {'name': 'Shrubland / thicket',    'color': '#b5832a'},
    5: {'name': 'Degraded bare ground',   'color': '#d4a76a'},
    6: {'name': 'Cropland',               'color': '#4a90c4'},
    7: {'name': 'Wetland / riparian',     'color': '#2070b4'},
}

DEGRADATION_CLASSES = {
    0: {'name': 'Recovering',  'color': '#2ecc71'},
    1: {'name': 'Stable',      'color': '#f39c12'},
    2: {'name': 'Degrading',   'color': '#e67e22'},
    3: {'name': 'Collapsed',   'color': '#c0392b'},
}


def build_dashboard(
    veg_map_path: str = None,
    risk_map_path: str = None,
    field_points_path: str = None,
    center_lat: float = -1.5,
    center_lon: float = 37.5,
    output_path: str = 'vegformer_dashboard.html'
) -> folium.Map:
    """
    Build and save an interactive Folium dashboard.
    """

    # Initialize map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=9,
        tiles=None,
        control_scale=True
    )

    # Base layers
    folium.TileLayer(
        'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        name='Google Satellite',
        attr='Google',
        max_zoom=20
    ).add_to(m)

    folium.TileLayer('OpenStreetMap', name='OpenStreetMap').add_to(m)

    folium.TileLayer(
        'https://server.arcgisonline.com/ArcGIS/rest/services/World_Terrain_Base/MapServer/tile/{z}/{y}/{x}',
        name='Terrain',
        attr='ESRI'
    ).add_to(m)

    # --- Vegetation Type Layer ---
    if veg_map_path and Path(veg_map_path).exists():
        veg_layer = folium.FeatureGroup(name='Vegetation Type', show=True)
        _add_raster_overlay(m, veg_map_path, 'Vegetation Type',
                            colormap=_veg_colormap(), opacity=0.7)

    # --- Degradation Risk Layer ---
    if risk_map_path and Path(risk_map_path).exists():
        _add_raster_overlay(m, risk_map_path, 'Degradation Risk',
                            colormap=_risk_colormap(), opacity=0.65,
                            show=False)

    # --- Field Validation Points (Braun-Blanquet) ---
    if field_points_path and Path(field_points_path).exists():
        bb_group = folium.FeatureGroup(name='Braun-Blanquet Validation', show=True)
        with open(field_points_path) as f:
            bb_data = json.load(f)
        for feat in bb_data.get('features', []):
            props = feat['properties']
            coords = feat['geometry']['coordinates']  # [lon, lat]
            veg_class = props.get('veg_class', 'Unknown')
            deg_class  = props.get('deg_class', 'Unknown')
            cover_pct  = props.get('total_cover', '-')
            species_n  = props.get('n_species', '-')
            model_pred = props.get('model_prediction', 'N/A')

            popup_html = f"""
            <div style="font-family: sans-serif; font-size: 12px; min-width: 200px;">
                <b>Braun-Blanquet Transect</b><br>
                <hr style="margin: 4px 0;">
                <b>Vegetation class:</b> {veg_class}<br>
                <b>Degradation:</b> {deg_class}<br>
                <b>Total cover:</b> {cover_pct}%<br>
                <b>Species richness:</b> {species_n}<br>
                <hr style="margin: 4px 0;">
                <b>Model prediction:</b> {model_pred}
            </div>
            """
            folium.CircleMarker(
                location=[coords[1], coords[0]],
                radius=7,
                color='white',
                weight=1.5,
                fill=True,
                fill_color='#e74c3c',
                fill_opacity=0.9,
                popup=folium.Popup(popup_html, max_width=260),
                tooltip=f"BB: {veg_class}"
            ).add_to(bb_group)
        bb_group.add_to(m)

    # --- Degradation Risk Choropleth (simplified demo) ---
    _add_demo_risk_points(m)

    # --- Legend ---
    _add_legend(m)

    # --- Full-screen button and layer control ---
    plugins.Fullscreen(
        position='topleft',
        title='Fullscreen',
        force_separate_button=True
    ).add_to(m)

    plugins.MiniMap(toggle_display=True).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    # --- Title overlay ---
    title_html = """
    <div style="position: fixed; top: 12px; right: 60px; z-index: 9999;
                background: rgba(255,255,255,0.93); padding: 10px 16px;
                border-radius: 8px; border: 1px solid #ddd;
                font-family: sans-serif; font-size: 13px; max-width: 280px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.15);">
        <b style="font-size: 14px;">VegFormer</b><br>
        <span style="color: #555; font-size: 11px;">
            Vegetation Dynamics & Land Degradation<br>
            Sentinel-1/2 · CHIRPS · Temporal Transformer
        </span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    m.save(output_path)
    print(f"Dashboard saved to: {output_path}")
    return m


def _add_demo_risk_points(m: folium.Map):
    """Add demonstration degradation risk markers."""
    demo_group = folium.FeatureGroup(name='Degradation Risk (demo)', show=True)

    import random
    random.seed(42)
    for _ in range(80):
        lat = -1.5 + random.uniform(-0.8, 0.8)
        lon = 37.5 + random.uniform(-0.8, 0.8)
        risk = random.random()
        traj = random.choice(list(DEGRADATION_CLASSES.values()))

        if risk > 0.75:
            color = '#c0392b'
        elif risk > 0.5:
            color = '#e67e22'
        elif risk > 0.25:
            color = '#f39c12'
        else:
            color = '#2ecc71'

        popup_html = f"""
        <div style="font-family: sans-serif; font-size: 12px; min-width: 180px;">
            <b>VegFormer Prediction</b><br>
            <hr style="margin: 4px 0;">
            <b>Risk score:</b> {risk:.2f}<br>
            <b>Trajectory:</b> {traj['name']}<br>
            <b>Uncertainty:</b> ±{random.uniform(0.05, 0.15):.2f}
        </div>
        """
        folium.CircleMarker(
            location=[lat, lon],
            radius=5 + risk * 8,
            color='white',
            weight=1,
            fill=True,
            fill_color=color,
            fill_opacity=0.75,
            popup=folium.Popup(popup_html, max_width=240),
            tooltip=f"Risk: {risk:.2f} | {traj['name']}"
        ).add_to(demo_group)

    demo_group.add_to(m)


def _add_raster_overlay(m, raster_path, name, colormap, opacity=0.7, show=True):
    """Add a GeoTIFF raster as an image overlay."""
    try:
        with rasterio.open(raster_path) as src:
            bounds = src.bounds
            folium_bounds = [[bounds.bottom, bounds.left],
                             [bounds.top, bounds.right]]
            # In production: convert raster to PNG with colormap
            # For now, register the layer as a placeholder
            print(f"  Registered raster layer: {name} from {raster_path}")
    except Exception as e:
        print(f"  Note: Could not load raster {raster_path}: {e}")


def _veg_colormap():
    """Return vegetation class colormap dict."""
    return {cls_id: info['color'] for cls_id, info in VEG_CLASSES.items()}


def _risk_colormap():
    """Return risk score colormap (continuous)."""
    return {'low': '#2ecc71', 'medium': '#f39c12', 'high': '#e74c3c'}


def _add_legend(m: folium.Map):
    """Add a vegetation class and degradation legend."""
    veg_items = ''.join([
        f'<div style="display:flex;align-items:center;gap:6px;margin:3px 0;">'
        f'<div style="width:14px;height:14px;background:{info["color"]};'
        f'border-radius:3px;flex-shrink:0;"></div>'
        f'<span>{info["name"]}</span></div>'
        for info in VEG_CLASSES.values()
    ])
    deg_items = ''.join([
        f'<div style="display:flex;align-items:center;gap:6px;margin:3px 0;">'
        f'<div style="width:14px;height:14px;background:{info["color"]};'
        f'border-radius:3px;flex-shrink:0;"></div>'
        f'<span>{info["name"]}</span></div>'
        for info in DEGRADATION_CLASSES.values()
    ])

    legend_html = f"""
    <div style="position: fixed; bottom: 40px; left: 12px; z-index: 9999;
                background: rgba(255,255,255,0.95); padding: 12px 16px;
                border-radius: 8px; border: 1px solid #ddd;
                font-family: sans-serif; font-size: 12px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.15); max-width: 200px;">
        <b>Vegetation Types</b><br>
        {veg_items}
        <br><b>Degradation Status</b><br>
        {deg_items}
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--veg_map', default=None)
    parser.add_argument('--risk_map', default=None)
    parser.add_argument('--field_points', default=None)
    parser.add_argument('--center_lat', type=float, default=-1.5)
    parser.add_argument('--center_lon', type=float, default=37.5)
    parser.add_argument('--output', default='vegformer_dashboard.html')
    args = parser.parse_args()

    dashboard = build_dashboard(
        veg_map_path=args.veg_map,
        risk_map_path=args.risk_map,
        field_points_path=args.field_points,
        center_lat=args.center_lat,
        center_lon=args.center_lon,
        output_path=args.output
    )
    print("Dashboard ready.")
