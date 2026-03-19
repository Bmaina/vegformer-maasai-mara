# VegFormer: Transformer-Based Vegetation Dynamics & Land Degradation Mapping

> **Multi-source satellite time-series analysis using cross-modal temporal transformers**  
> Sentinel-1 SAR · Sentinel-2 Optical · CHIRPS Rainfall · MODIS legacy labels

---

## Novelty Statement

This project makes four distinct contributions over prior NDVI-based vegetation mapping work:

1. **Masked Sensor Modeling (MSM)** — Self-supervised pre-training objective where random sensor-timestep blocks are masked across S1/S2/CHIRPS jointly. The model learns cross-sensor dependencies without labeled data.
2. **Cross-Modal Attention Fusion** — Instead of naive concatenation, a cross-attention module lets the SAR encoder query optical features and vice versa, learning *when* each modality is more informative (cloud cover events, soil moisture spikes).
3. **Degradation Trajectory Classification** — Predicts ordered sequences (`recovering → stable → degrading → collapsed`) rather than static land cover labels, validated against Braun-Blanquet phytosociological transects.
4. **MODIS Weak Supervision Bridge** — Your legacy MODIS NDVI labels (2000–2015) are used as noisy pseudo-labels to warm-start the trajectory head, bridging the MODIS → Sentinel resolution gap.

---

## Project Structure

```
vegformer/
├── data/
│   ├── gee/
│   │   ├── export_sentinel1.py        # GEE S1 SAR export (VV, VH, 16-day composites)
│   │   ├── export_sentinel2.py        # GEE S2 optical export (12 bands, cloud-masked)
│   │   ├── export_chirps.py           # CHIRPS rainfall export (daily → 16-day sum)
│   │   └── export_modis_legacy.py     # MODIS NDVI/EVI for weak supervision labels
│   ├── preprocessing/
│   │   ├── cloud_mask.py              # SCL-based cloud/shadow masking
│   │   ├── temporal_composite.py      # 16-day median compositing
│   │   ├── patch_extractor.py         # 64×64 px patch extraction with overlap
│   │   └── normalization.py           # Per-band z-score normalization
│   └── datasets/
│       ├── vegformer_dataset.py       # PyTorch Dataset: (T, C_s1, C_s2, C_clim) tensors
│       └── weak_label_dataset.py      # MODIS-aligned pseudo-label dataset
│
├── models/
│   ├── encoders/
│   │   ├── sar_encoder.py             # 1D temporal conv + bidirectional LSTM
│   │   ├── optical_encoder.py         # Prithvi/SatMAE backbone (frozen or fine-tuned)
│   │   └── climate_encoder.py         # Informer encoder for irregular rainfall series
│   ├── fusion/
│   │   └── cross_modal_attention.py   # Cross-attention: SAR queries optical KV, and vice versa
│   ├── transformer/
│   │   ├── vegformer.py               # Main VegFormer model
│   │   ├── positional_encoding.py     # Julian day + seasonal Fourier embedding
│   │   └── trajectory_head.py         # 4-class ordered sequence classification head
│   └── pretraining/
│       └── masked_sensor_modeling.py  # MSM objective: mask patches across sensors, predict
│
├── training/
│   ├── pretrain.py                    # Stage 1: self-supervised MSM pre-training
│   ├── finetune_weak.py               # Stage 2: fine-tune with MODIS weak labels
│   ├── finetune_full.py               # Stage 3: fine-tune with Braun-Blanquet labels
│   ├── losses.py                      # MSM reconstruction loss + trajectory CTC loss
│   └── metrics.py                     # OA, F1, kappa, degradation trajectory accuracy
│
├── pipelines/
│   ├── prefect_ingest.py              # Prefect flow: GEE export → local cache
│   ├── prefect_train.py               # Prefect flow: data → pretrain → finetune
│   └── prefect_inference.py           # Prefect flow: patch inference → mosaic → GeoTIFF
│
├── evaluation/
│   ├── braun_blanquet_validator.py    # Cross-validate against BB phytosociology
│   ├── confusion_matrix.py            # Per-class and trajectory confusion
│   └── attention_visualizer.py        # Attention weight maps over time
│
├── visualization/
│   ├── folium_dashboard.py            # Interactive Folium map (risk + vegetation type)
│   ├── umap_embeddings.py             # UMAP projection of learned time-series embeddings
│   └── degradation_timeline.py        # Pixel-level trajectory animation
│
├── tracking/
│   └── mlflow_config.py               # MLflow experiment tracking config
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_pretraining_experiments.ipynb
│   ├── 03_finetuning_ablations.ipynb
│   └── 04_results_visualization.ipynb
│
├── configs/
│   ├── pretrain_config.yaml
│   ├── finetune_config.yaml
│   └── inference_config.yaml
│
├── requirements.txt
├── environment.yml
└── README.md
```

---

## Model Architecture: VegFormer

```
Input: (B, T=46, C=20) tensor
  - T = 46 timesteps × 16 days ≈ 2 years
  - C = VV + VH (SAR) + 12 bands (optical) + 4 climate vars + 2 auxiliary

Stage 1 — Per-modality encoders:
  SAR encoder:      (B, T, 2)   → (B, T, 128)   [1D conv + BiLSTM]
  Optical encoder:  (B, T, 12)  → (B, T, 256)   [Prithvi patch embed + Transformer]
  Climate encoder:  (B, T, 4)   → (B, T, 64)    [Informer sparse attention]

Stage 2 — Cross-modal fusion:
  SAR ↔ Optical cross-attention   → (B, T, 384)
  + Climate residual concatenation → (B, T, 448)

Stage 3 — Temporal Transformer:
  Julian day positional encoding
  4× Transformer encoder layers (d=448, heads=8, FFN=1792)
  [CLS] token aggregation → (B, 448)

Stage 4 — Task heads:
  Vegetation type head:     Linear(448, 8)   → 8 vegetation classes
  Degradation trajectory:   Linear(448, 4)   → trajectory state
  Risk score:               Linear(448, 1)   → continuous risk [0,1]
```

---

## Training Protocol (3-Stage)

### Stage 1: Masked Sensor Modeling (MSM) — Self-supervised
- **Objective**: Randomly mask 40% of timestep-sensor blocks; reconstruct masked values
- **Data**: All available S1/S2/CHIRPS patches, no labels required
- **Duration**: ~50 epochs, lr=1e-4, batch=64
- **Loss**: MSE on masked positions (per-sensor normalized)

### Stage 2: Weak Supervision Fine-tuning
- **Data**: MODIS-aligned NDVI clusters → pseudo vegetation classes
- **Strategy**: Freeze encoder weights, train trajectory head only
- **Loss**: Cross-entropy with label smoothing (ε=0.1) for noisy labels
- **Duration**: ~20 epochs, lr=5e-5

### Stage 3: Full Fine-tuning with Field Labels
- **Data**: Braun-Blanquet field transects (your existing labels)
- **Strategy**: Unfreeze all, discriminative learning rates (encoder: 1e-5, head: 1e-4)
- **Loss**: Weighted cross-entropy + ordinal trajectory loss
- **Duration**: ~30 epochs with early stopping

---

## Key Implementation Details

### Cross-Modal Attention Fusion

```python
class CrossModalAttention(nn.Module):
    """
    SAR encoder queries optical key-value pairs to learn
    when SAR backscatter is more informative than optical reflectance
    (e.g., under persistent cloud cover or during soil moisture events).
    """
    def __init__(self, sar_dim=128, optical_dim=256, heads=4):
        super().__init__()
        self.sar_to_optical = nn.MultiheadAttention(
            embed_dim=sar_dim, kdim=optical_dim, vdim=optical_dim,
            num_heads=heads, batch_first=True
        )
        self.optical_to_sar = nn.MultiheadAttention(
            embed_dim=optical_dim, kdim=sar_dim, vdim=sar_dim,
            num_heads=heads, batch_first=True
        )

    def forward(self, sar_feats, optical_feats):
        # SAR attends to optical context
        sar_enhanced, sar_attn = self.sar_to_optical(
            sar_feats, optical_feats, optical_feats
        )
        # Optical attends to SAR context (soil moisture, structure)
        optical_enhanced, opt_attn = self.optical_to_sar(
            optical_feats, sar_feats, sar_feats
        )
        return sar_enhanced, optical_enhanced, sar_attn, opt_attn
```

### Seasonal Positional Encoding

```python
class SeasonalPositionalEncoding(nn.Module):
    """
    Encodes both absolute date (Julian day) and seasonal position
    using Fourier features — critical for vegetation phenology cycles.
    """
    def __init__(self, d_model, max_len=500):
        super().__init__()
        self.d_model = d_model

    def forward(self, x, dates):  # dates: (B, T) Julian days [1-365]
        # Annual cycle (365 days)
        sin_annual = torch.sin(2 * torch.pi * dates / 365)
        cos_annual = torch.cos(2 * torch.pi * dates / 365)
        # Semi-annual cycle (bimodal rainfall in East Africa)
        sin_semi = torch.sin(4 * torch.pi * dates / 365)
        cos_semi = torch.cos(4 * torch.pi * dates / 365)
        # Stack and project
        season_feats = torch.stack(
            [sin_annual, cos_annual, sin_semi, cos_semi], dim=-1
        )
        return x + self.project(season_feats)
```

### Degradation Trajectory Classes

```
Class 0: Recovering    — NDVI/SAR trend positive, reaching reference state
Class 1: Stable        — Low inter-annual variance, within normal range
Class 2: Degrading     — Sustained decline over ≥3 consecutive composites
Class 3: Collapsed     — Below minimum threshold, bare soil dominance
```

---

## Data Requirements & GEE Exports

| Source | Variables | Resolution | Period | Export format |
|--------|-----------|-----------|--------|---------------|
| Sentinel-1 GRD | VV, VH backscatter | 10m | 2017–2024 | 16-day median composite |
| Sentinel-2 SR | B2–B8A, B11, B12 | 10m | 2017–2024 | 16-day median, SCL-masked |
| CHIRPS v2.0 | Rainfall | ~5km | 2015–2024 | 16-day sum |
| MODIS MOD13Q1 | NDVI, EVI | 250m | 2001–2017 | 16-day composite |
| Field data | Braun-Blanquet transects | Point | Fieldwork | GeoJSON / CSV |

Recommended study area: **Arid/semi-arid drylands**, Kenya/Tanzania (aligns with your MODIS legacy data).  
Minimum recommended patches for pre-training: **50,000+** (achievable from any 50×50km GEE export).

---

## Outputs

### 1. Vegetation Type Map
- 8 phytosociology-aligned classes (mapped from Braun-Blanquet)
- GeoTIFF + WMS tile layer
- Per-class confidence score

### 2. Degradation Risk Map
- Continuous risk score [0, 1]
- 95% uncertainty bounds (MC Dropout or ensemble)
- Change detection layer vs. MODIS baseline period

### 3. Time-Series Embeddings
- UMAP projection of [CLS] token embeddings
- Cluster analysis (k-means / HDBSCAN) for unsupervised vegetation grouping
- Animatable trajectory plots per pixel

### 4. Interactive Folium Dashboard
- Multi-layer toggle: vegetation type / degradation risk / NDVI anomaly
- Click-to-inspect: pixel-level time series + trajectory prediction
- Export: PNG, CSV, GeoJSON per selected AOI

---

## MLflow Experiment Tracking

```python
# Key tracked metrics per run
mlflow.log_params({
    "model": "VegFormer-Base",
    "pretrain_epochs": 50,
    "masking_ratio": 0.4,
    "sensors": "S1+S2+CHIRPS",
    "weak_sup": True,
    "bb_labels_n": 847
})

mlflow.log_metrics({
    "veg_type_OA": 0.87,
    "degradation_F1": 0.79,
    "trajectory_accuracy": 0.73,
    "kappa": 0.81
})
```

---

## Publication Target

**Primary**: *Remote Sensing of Environment* (RSE) — highest impact in this domain  
**Secondary**: *IEEE Transactions on Geoscience and Remote Sensing* (TGRS)  
**Conference**: IGARSS 2026 (abstract deadline typically November)

**Suggested title**:  
*"VegFormer: Cross-Modal Masked Sensor Modeling for Vegetation Dynamics and Land Degradation Mapping in East African Drylands"*

---

## Requirements

```
torch>=2.1.0
torchvision>=0.16.0
einops>=0.7.0
earthengine-api>=0.1.380
geemap>=0.30.0
prefect>=2.14.0
mlflow>=2.9.0
folium>=0.15.0
umap-learn>=0.5.5
rasterio>=1.3.9
geopandas>=0.14.0
transformers>=4.37.0    # for Prithvi backbone
timm>=0.9.12
```

---

## Citation (template)

```bibtex
@article{vegformer2025,
  title={VegFormer: Cross-Modal Masked Sensor Modeling for Vegetation Dynamics 
         and Land Degradation Mapping},
  author={Your Name},
  journal={Remote Sensing of Environment},
  year={2025},
  note={Under review}
}
```
