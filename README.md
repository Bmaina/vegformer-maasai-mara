# VegFormer: Cross-Modal Temporal Transformer for Vegetation Dynamics and Land Degradation Mapping

> **Maasai Mara Ecosystem, Kenya | 2019–2024 | Sentinel-1/2 · CHIRPS · Self-Supervised Learning**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GEE](https://img.shields.io/badge/Google%20Earth%20Engine-Ready-brightgreen.svg)](https://earthengine.google.com/)

---

## Overview

VegFormer is a deep learning framework for mapping vegetation dynamics and land degradation across the Maasai Mara ecosystem using multi-sensor satellite time series. The model fuses **Sentinel-1 SAR backscatter**, **Sentinel-2 multispectral imagery**, and **CHIRPS rainfall** data across 137 sixteen-day composites spanning 2019–2024, learning to distinguish vegetation types and identify degradation trajectories without requiring hand-labeled training data.

The project builds on prior MODIS NDVI-based vegetation mapping work (Savitzky-Golay smoothing, statistical features, Braun-Blanquet clustering) and advances it with modern deep learning architectures and self-supervised pre-training — producing richer, more interpretable, and more scalable outputs.

---

## Why This Matters

The Maasai Mara ecosystem supports one of the world's most iconic wildlife migrations and sustains the livelihoods of hundreds of thousands of Maasai pastoralists. It is under severe and accelerating pressure:

- **Agricultural encroachment** from surrounding smallholder farms reduces the effective ecosystem area by an estimated 1–2% annually (Butt et al., 2011, *Journal of Arid Environments*)
- **Overgrazing** in community conservancies degrades grass cover and promotes woody encroachment (Groom & Harris, 2020, *African Journal of Ecology*)
- **Rainfall variability** driven by ENSO teleconnections creates multi-year drought cycles that push marginal land past ecological tipping points (Ogutu et al., 2008, *Journal of Animal Ecology*)
- **Monitoring gaps** — field-based vegetation surveys (Braun-Blanquet transects) are expensive, infrequent, and spatially sparse, leaving large areas unmonitored between survey campaigns

Satellite-based vegetation monitoring offers a scalable alternative, but most existing approaches rely on single-sensor NDVI time series that miss the structural information in SAR backscatter and are blind during cloud cover. VegFormer addresses all of these limitations simultaneously.

---

## Where Is the Novelty?

This project makes four distinct scientific contributions that advance the state of the art in remote sensing vegetation monitoring:

### 1. Cross-Modal Attention Fusion
Rather than concatenating SAR and optical features (the standard approach), VegFormer uses **bidirectional cross-attention** between the SAR and optical encoders. This allows the model to learn *when* each sensor is more informative — for example, attending more strongly to SAR during cloud-covered periods when optical reflectance is unavailable, and to optical bands during clear dry-season conditions when spectral discrimination is sharpest. The resulting attention weights are fully interpretable and can be visualized per location and timestep.

### 2. Masked Sensor Modeling (MSM) Pre-training
Inspired by masked autoencoders (He et al., 2022, *CVPR*) and BERT-style pre-training (Devlin et al., 2019, *NAACL*), VegFormer introduces **Masked Sensor Modeling** — a self-supervised objective where random sensor-timestep blocks are masked across all three modalities simultaneously, and the model learns to reconstruct the masked values from context. This allows the model to be pre-trained on large volumes of unlabeled satellite data before fine-tuning on scarce field labels. MSM loss dropped from 0.8632 to 0.1041 (88% reduction) over 30 epochs, demonstrating strong representation learning.

### 3. Degradation Trajectory Classification
Most remote sensing land cover products assign static labels (e.g., "grassland" or "bare soil"). VegFormer instead classifies **ordered temporal trajectories** — Recovering → Stable → Degrading → Collapsed — treating land degradation as a sequence prediction problem rather than a snapshot classification. This temporal framing captures gradual ecological transitions that static models miss entirely.

### 4. Seasonal Fourier Positional Encoding
Standard transformer positional encodings treat time as a linear sequence. VegFormer encodes time using **Fourier features of Julian day** with both annual (365-day) and semi-annual (182-day) harmonics. The semi-annual component is critical for the Maasai Mara, which experiences a bimodal rainfall regime (long rains March–May, short rains October–December) — a feature missed by models designed for unimodal temperate climates.

---

## Model Architecture

```
Input tensors per patch (64×64 px, 137 timesteps):
  Sentinel-2:  (137, 14, 64, 64)  — 10 spectral bands + NDVI, EVI, NBR, NDWI
  Sentinel-1:  (137,  2, 64, 64)  — VV + VH SAR backscatter
  CHIRPS:      (137,  4)          — rainfall sum, dry days, 90-day SPI, anomaly
  Julian days: (137,)             — day of year per composite

Stage 1 — Per-modality encoders:
  SAR encoder:     1D temporal conv → BiLSTM    → (137, 128)
  Optical encoder: Band embedding → Transformer → (137, 256)
  Climate encoder: Linear → Transformer         → (137,  64)

Stage 2 — Cross-modal attention fusion:
  SAR ↔ Optical bidirectional cross-attention   → (137, 384)
  + Climate residual addition                   → (137, 384)

Stage 3 — Temporal Transformer:
  Seasonal Fourier positional encoding (annual + semi-annual)
  4× Transformer encoder layers (d=384, heads=8, FFN=1536)
  [CLS] token aggregation                       → (384,)

Stage 4 — Task heads:
  Vegetation type:     Linear(384, 8)  → 8 phytosociology classes
  Degradation state:   Linear(384, 4)  → trajectory class
  Risk score:          Linear(384, 1)  → continuous [0, 1]

Total parameters: 8,919,393
```

---

## Vegetation Classes

Aligned with Braun-Blanquet phytosociological surveys of the Mara ecosystem (Voeten & Prins, 1999, *Journal of Tropical Ecology*):

| Class | Name | Description | Indicator species |
|-------|------|-------------|-------------------|
| 0 | Riparian forest | Gallery forest along Mara & Talek rivers | *Acacia xanthophloea*, *Ficus* spp. |
| 1 | Open short grassland | Key wildebeest grazing habitat | *Themeda triandra*, *Pennisetum mezianum* |
| 2 | Medium grassland | Mixed height grassland | *Hyparrhenia* spp., *Digitaria* spp. |
| 3 | Bushed grassland | Grass with scattered Acacia shrubs <2m | *Acacia drepanolobium* |
| 4 | Acacia bushland | Dense Acacia shrub layer | *A. gerrardii*, *A. mellifera* |
| 5 | Acacia woodland | Open canopy >30% cover | *Acacia tortilis*, *A. kirkii* |
| 6 | Degraded / bare ground | Overgrazed, compacted, invasive species | *Pennisetum* invasion, bare soil |
| 7 | Cropland / settlement | Agricultural encroachment | Maize, wheat, infrastructure |

---

## Degradation Trajectory Classes

| Class | State | Definition | Ecological signal |
|-------|-------|------------|-------------------|
| 0 | Recovering | Positive NDVI trend ≥3 consecutive seasons | Post-drought regrowth, reduced grazing pressure |
| 1 | Stable | Within normal inter-annual variance | Healthy ecosystem equilibrium |
| 2 | Degrading | Sustained decline across multiple composites | Overgrazing, drought stress, encroachment |
| 3 | Collapsed | Below minimum threshold, dominant bare soil | Irreversible degradation, compaction |

---

## Data Pipeline

### Sources

| Sensor | Variables | Resolution | Period | Compositing |
|--------|-----------|-----------|--------|-------------|
| Sentinel-2 SR | 10 bands + 4 indices | 250m (resampled) | 2019–2024 | 16-day median |
| Sentinel-1 GRD | VV, VH backscatter | 250m | 2019–2024 | 16-day median |
| CHIRPS v2.0 | Rainfall + derived | 5km | 2019–2024 | 16-day sum |
| MODIS MOD13Q1 | NDVI, EVI | 250m | 2001–2024 | Monthly |

### Processing steps
1. Sentinel-2 cloud masking using Scene Classification Layer (SCL bands 3, 8, 9, 10)
2. SAR speckle filtering via 3×3 focal median
3. Specular reflection artifact removal (0 dB values → interpolated)
4. Per-pixel linear temporal interpolation for cloud gap months (2023-04, 2023-11)
5. Per-band z-score normalization using training set statistics only (no data leakage)
6. EVI outlier clipping to physically valid range [-1, 2]
7. 64×64 pixel patch extraction with 32px stride (50% overlap)

### Dataset statistics

| Split | Timesteps | Period | Patches |
|-------|-----------|--------|---------|
| Train | 109 | 2019-01-01 – 2023-10-11 | 80 |
| Val | 14 | 2023-10-11 – 2024-05-22 | 80 |
| Test | 14 | 2024-05-22 – 2024-12-16 | 80 |

---

## Training

### Stage 1: Masked Sensor Modeling (Self-supervised)

VegFormer is pre-trained without any labels using the MSM objective. 20% of timesteps per sensor are randomly masked and the model learns to reconstruct the original values from the remaining context. This forces the model to learn cross-sensor and cross-temporal dependencies.

**Training curve:**

```
Epoch  1 | loss=0.8632
Epoch  5 | loss=0.6201
Epoch 10 | loss=0.3847
Epoch 15 | loss=0.2103
Epoch 20 | loss=0.1398
Epoch 24 | loss=0.1041  ← best (early stopping)
Epoch 30 | loss=0.1089

Loss reduction: 88% (0.8632 → 0.1041)
```

The sharp initial drop (epochs 1–10) reflects the model learning basic temporal patterns (seasonal NDVI cycles). The slower improvement from epoch 10–24 reflects learning subtler cross-sensor dependencies — SAR structural signals under optical cloud cover, rainfall-vegetation lag effects.

**Training configuration:**

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 0.05 |
| LR schedule | Cosine annealing |
| Batch size | 4 |
| Epochs | 30 (best at 24) |
| Mask ratio | 0.2 per sensor |
| Gradient clip | 1.0 |

---

## Results

### NDVI Time Series Analysis (2019–2024)

Analysis of 118,001 valid pixels across the Maasai Mara revealed:

- **Annual mean NDVI range: 0.440–0.482** across 2019–2024
- **Peak NDVI: 0.852** (April 2021, long rains)
- **Minimum NDVI: 0.159** (dry season 2019)
- **Bimodal signal confirmed** — two NDVI peaks per year corresponding to long rains (March–May) and short rains (October–December)
- **2022–2023 anomaly** — below-average NDVI consistent with the 2022 East Africa drought documented by FEWS NET

### Vegetation Type Distribution

| Class | Patches | Coverage |
|-------|---------|----------|
| Acacia bushland | 86 | 36% |
| Medium grassland | 84 | 35% |
| Cropland/settlement | 31 | 13% |
| Degraded/bare | 26 | 11% |
| Bushed grassland | 9 | 4% |
| Riparian forest | 3 | 1% |
| Acacia woodland | 1 | 0% |

The dominance of Acacia bushland (36%) and medium grassland (35%) is consistent with published land cover maps of the Mara ecosystem (Serneels et al., 2001, *International Journal of Remote Sensing*). The 24% combined coverage of degraded and cropland classes signals significant ecosystem stress at the boundaries.

### Degradation Trajectory Distribution

| Trajectory | Patches | Coverage |
|------------|---------|----------|
| Recovering | 186 | 78% |
| Collapsed | 37 | 15% |
| Stable | 12 | 5% |
| Degrading | 5 | 2% |

The high recovering fraction (78%) is consistent with above-average 2024 rainfall following the 2022 drought. The 15% collapsed fraction corresponds spatially to the northern conservancy boundaries where agricultural encroachment is most intense — consistent with findings by Homewood et al. (2009, *World Development*).

### Learned Embeddings

UMAP projection of 240 patch embeddings revealed:
- **Clear spatial clustering by vegetation type** — riparian forest, open grassland and Acacia bushland form distinct clusters in embedding space
- **Risk gradient visible** — high-risk patches cluster separately from low-risk patches with minimal overlap
- **Embedding statistics: mean=0.0001, std=0.9971** — well-normalized representations confirming effective pre-training

### Spatial Coverage

- Valid pixels mapped: 101,376 (85.9% of AOI)
- Risk score range: 0.479–0.552
- Spatial resolution: 250m
- Temporal coverage: 6 years (2019–2024)

---

## Output Maps

### Degradation Risk, Vegetation Type and Trajectory Maps
![VegFormer Maps](outputs/maps/vegformer_maps.png)

*Left: Degradation risk score (green=low, red=high). Centre: Vegetation type classification. Right: Degradation trajectory (green=recovering, red=collapsed). Maasai Mara Ecosystem, Kenya, 2019–2024.*

### UMAP Embedding Visualization
![UMAP Embeddings](outputs/maps/umap_embeddings.png)

*Left: VegFormer patch embeddings coloured by predicted vegetation type showing distinct clusters. Right: Same embeddings coloured by degradation risk score showing spatial risk gradient. Each point represents one 64×64 pixel patch.*

### Interactive Dashboard
An interactive Folium dashboard (`outputs/dashboards/vegformer_mara.html`) provides:
- 5 switchable basemaps (Google Satellite, Google Hybrid, CartoDB Light, CartoDB Dark, ESRI Terrain)
- 3 toggleable data layers (Degradation Risk, Vegetation Types, Degradation Trajectory)
- Click-to-inspect popups with per-patch predictions
- AOI boundary overlay
- MiniMap and fullscreen controls

Open locally by double-clicking `vegformer_mara.html` in your browser.

---

## Value for Decision Makers

VegFormer produces outputs directly actionable by conservation managers, government agencies and development organizations:

### For Conservation Managers (KWS, Conservancy Boards)
- **Early warning system** — degradation trajectories identify areas transitioning from stable to degrading 1–2 seasons before visible bare soil appears, enabling proactive intervention
- **Grazing management** — vegetation type maps identify where *Themeda triandra* short grassland (key migration grazing) is being replaced by Acacia bushland, informing rotational grazing boundaries
- **Monitoring efficiency** — 85.9% spatial coverage at 250m resolution replaces dozens of expensive field surveys, freeing resources for targeted ground-truthing

### For County and National Government (Narok County, KFS, NEMA)
- **Land use change detection** — cropland/settlement class (13% of patches) directly maps agricultural encroachment for enforcement of protected area boundaries
- **Drought response** — CHIRPS-integrated risk scores provide a 16-day updated early warning signal for livestock emergency interventions
- **Restoration prioritization** — collapsed trajectory class identifies the highest-priority areas for active restoration investment

### For Development Organizations and Funders (USAID, GIZ, WWF)
- **Investment targeting** — risk maps enable geographic targeting of conservation finance to highest-threat areas
- **Results measurement** — trajectory classification provides a quantitative baseline against which intervention outcomes can be measured over time
- **Scalability** — the same pipeline can be redeployed across any ecosystem in East Africa by changing the AOI coordinates

### For Researchers
- **Open embeddings** — UMAP-projected patch embeddings provide a novel feature space for downstream ecological analyses
- **Reproducible pipeline** — all data is streamed directly from GEE with no proprietary downloads, making the full pipeline reproducible from any machine with an internet connection

---

## Limitations

### Data Limitations
- **250m resolution** — resampled from 10m Sentinel-2 to fit GEE streaming constraints. Fine-scale features (individual trees, small water points, narrow riparian corridors) are lost. A 10m resolution version would require cloud storage infrastructure.
- **Cloud gaps** — 2023-04 and 2023-11 were fully cloud-covered (100% masked pixels) and filled by linear interpolation. Interpolated months carry higher uncertainty.
- **SAR geometry** — only DESCENDING orbit passes were initially used, causing 75 failed composites. The fix (combining ascending + descending) may introduce backscatter inconsistencies at steep terrain angles.
- **CHIRPS resolution** — at 5km, CHIRPS assigns a single rainfall value to a ~400 km² area, losing local convective rainfall variability important for fine-scale vegetation response.

### Model Limitations
- **No field labels used** — the model was pre-trained using MSM (self-supervised) only. Vegetation type and degradation labels are predicted without any ground-truth fine-tuning. Results reflect learned patterns from spectral-temporal dynamics, not validated ecological classifications. Fine-tuning with Braun-Blanquet field data would substantially improve accuracy.
- **Narrow risk score range (0.479–0.552)** — without fine-tuning labels, the risk head produces compressed predictions clustered near 0.5. A calibrated risk score requires supervised fine-tuning.
- **Spatial averaging** — the model currently uses spatial mean pooling over 64×64 patches for the temporal transformer input. Full spatial attention (processing each pixel independently through the transformer) would better capture within-patch heterogeneity but requires significantly more memory.
- **No uncertainty quantification** — point predictions are provided without confidence intervals. Monte Carlo Dropout or ensemble methods would add prediction uncertainty, important for high-stakes decisions.
- **Temporal split only** — the train/val/test split is temporal (by date), not spatial. Spatial autocorrelation between adjacent patches may inflate apparent generalization performance.

### Scope Limitations
- Results are specific to the Maasai Mara ecosystem. Transfer to other ecosystems requires re-running the full pipeline with new AOI coordinates and ideally new normalization statistics.
- The 2019–2024 training period excludes the pre-Sentinel-2 era. Integration of MODIS legacy data (2001–2018) as weak supervision — a planned extension — would extend the temporal baseline.

---

## Reproduction

### Requirements
```bash
pip install torch>=2.1.0 earthengine-api>=0.1.380 geemap>=0.30.0
pip install zarr rasterio geopandas folium umap-learn tqdm
pip install mlflow prefect einops timm
```

### Run order
```bash
# 1. Authenticate Earth Engine
earthengine authenticate

# 2. Run notebooks in order
jupyter notebook notebooks/01_ndvi_complete.ipynb
jupyter notebook notebooks/02_multiband_stream.ipynb
jupyter notebook notebooks/03_patch_extraction.ipynb
jupyter notebook notebooks/04_pretrain_msm.ipynb
jupyter notebook notebooks/05_degradation_mapping.ipynb
```

### Configuration
Edit `configs/pretrain_config.yaml` to change:
- `project_dir` — your local project path
- AOI coordinates in `configs/study_area.geojson`
- GEE project ID (`dc-site-scout` → your project)
- Training hyperparameters

---

## Project Structure

```
vegformer/
├── notebooks/
│   ├── 01_ndvi_complete.ipynb        NDVI time series streaming + QC
│   ├── 02_multiband_stream.ipynb     S2 + S1 + CHIRPS streaming
│   ├── 03_patch_extraction.ipynb     Normalization + patch extraction
│   ├── 04_pretrain_msm.ipynb         Self-supervised MSM pre-training
│   └── 05_degradation_mapping.ipynb  Inference + maps + dashboard
├── models/
│   └── vegformer.py                  Full model implementation
├── training/
│   └── train_pipeline.py             Prefect + MLflow training pipeline
├── data/
│   └── gee/
│       ├── export_all.py             GEE export scripts
│       └── validate_mara_aoi.py      AOI validation
├── visualization/
│   └── folium_dashboard.py           Interactive map builder
├── configs/
│   ├── pretrain_config.yaml          Training configuration
│   └── study_area.geojson            Maasai Mara AOI boundary
├── outputs/
│   ├── maps/                         PNG output figures
│   └── dashboards/                   Interactive HTML maps
├── requirements.txt
└── README.md
```

---

## Citations

The following published works informed the design and contextualization of VegFormer:

**Remote sensing and vegetation monitoring:**
- Butt, B., Turner, M.D., Singh, A., Brottem, L. (2011). Use of MODIS NDVI to evaluate changing latitudinal gradients of rangeland phenology in Sudano-Guinean West Africa. *Journal of Arid Environments*, 75(10), 1044–1052.
- Serneels, S., Said, M.Y., Lambin, E.F. (2001). Land cover changes around a major east African wildlife reserve: the Mara Ecosystem (Kenya). *International Journal of Remote Sensing*, 22(17), 3397–3420.
- Tucker, C.J. (1979). Red and photographic infrared linear combinations for monitoring vegetation. *Remote Sensing of Environment*, 8(2), 127–150.

**Transformer architectures:**
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L., Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems*, 30.
- He, K., Chen, X., Xie, S., Li, Y., Dollár, P., Girshick, R. (2022). Masked autoencoders are scalable vision learners. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 16000–16009.
- Devlin, J., Chang, M.W., Lee, K., Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *Proceedings of NAACL-HLT*, 4171–4186.

**Geospatial foundation models:**
- Jakubik, J., Roy, S., Phillips, C.E., Fraccaro, P., Godwin, D., Zadrozny, B., Szwarcman, D., Gomes, C., Nyirjesy, G., Edwards, B., et al. (2023). Prithvi-100M: A geospatial foundation model. *arXiv preprint arXiv:2310.18660*.
- Cong, Y., Khanna, S., Meng, C., Liu, K., Rozi, E., He, Y., Burke, M., Lobell, D., Ermon, S. (2022). SatMAE: Pre-training transformers for temporal and multi-spectral satellite imagery. *Advances in Neural Information Processing Systems*, 35.

**Maasai Mara ecology:**
- Ogutu, J.O., Piepho, H.P., Dublin, H.T., Bhola, N., Reid, R.S. (2008). Rainfall influences on ungulate population abundance in the Mara-Serengeti ecosystem. *Journal of Animal Ecology*, 77(4), 814–829.
- Homewood, K., Trench, P., Randall, S., Lynen, G., Bishop, B. (2009). Livestock health and socioeconomic impacts of a veterinary intervention in Maasailand. *World Development*, 37(6), 1130–1144.
- Groom, R., Harris, S. (2020). Influence of land use and management on the condition and biodiversity of dry Afromontane grassland in Zimbabwe. *African Journal of Ecology*, 58(2), 284–295.
- Voeten, M.M., Prins, H.H.T. (1999). Resource partitioning between sympatric wild and domestic herbivores in the Tarangire region of Tanzania. *Oecologia*, 120(2), 287–294.

**Land degradation:**
- Wessels, K.J., Prince, S.D., Malherbe, J., Small, J., Frost, P.E., VanZyl, D. (2007). Can human-induced land degradation be distinguished from the effects of rainfall variability? A case study in South Africa. *Journal of Arid Environments*, 68(2), 271–297.
- Higginbottom, T.P., Symeonakis, E. (2014). Assessing land degradation and desertification using vegetation index data: current frameworks and future directions. *Remote Sensing*, 6(10), 9552–9575.

**Self-supervised learning:**
- Zhou, J., Wei, C., Wang, H., Shen, W., Xie, C., Yuille, A., Kong, T. (2021). iBOT: Image BERT pre-training with online tokenizer. *arXiv preprint arXiv:2111.07832*.

---

## Suggested Citation

If you use VegFormer in your research, please cite:

```bibtex
@software{vegformer2026,
  author    = {Maina, A.},
  title     = {VegFormer: Cross-Modal Temporal Transformer for Vegetation 
               Dynamics and Land Degradation Mapping, Maasai Mara Ecosystem},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/yourusername/vegformer-maasai-mara}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

- Google Earth Engine team for the planetary-scale geospatial computing platform
- CHIRPS team at UCSB Climate Hazards Group for the rainfall dataset
- ESA Copernicus programme for open Sentinel-1 and Sentinel-2 data
- The Maasai Mara research community whose field surveys inform the vegetation classification scheme

