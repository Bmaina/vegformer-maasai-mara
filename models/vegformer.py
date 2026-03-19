"""
VegFormer: Cross-Modal Temporal Transformer for Vegetation Dynamics
====================================================================
Novel contributions:
  1. Cross-modal attention fusion (SAR ↔ Optical)
  2. Masked Sensor Modeling (MSM) self-supervised pre-training
  3. Seasonal Fourier positional encoding for phenology
  4. Degradation trajectory classification head
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------

class SeasonalPositionalEncoding(nn.Module):
    """
    Combines Julian day Fourier features (annual + semi-annual cycles)
    with a learned projection into d_model space.

    East Africa has a bimodal rainfall regime (long/short rains),
    so semi-annual (4π/365) encoding is critical, not just annual.
    """

    def __init__(self, d_model: int, n_fourier: int = 4):
        super().__init__()
        # n_fourier: number of Fourier harmonics per cycle
        in_dim = n_fourier      # each harmonic already generates sin+cos pair
        self.project = nn.Linear(in_dim, d_model)
        self.n_fourier = n_fourier

    def forward(self, x: torch.Tensor, dates: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:     (B, T, d_model) — input sequence
            dates: (B, T) — Julian day [1, 365]
        Returns:
            (B, T, d_model) — x + seasonal position encoding
        """
        B, T = dates.shape
        feats = []
        for k in range(1, self.n_fourier // 2 + 1):
            feats.append(torch.sin(2 * math.pi * k * dates / 365))
            feats.append(torch.cos(2 * math.pi * k * dates / 365))
        season_feats = torch.stack(feats, dim=-1)  # (B, T, n_fourier)
        pos_enc = self.project(season_feats)        # (B, T, d_model)
        return x + pos_enc


# ---------------------------------------------------------------------------
# Per-Modality Encoders
# ---------------------------------------------------------------------------

class SAREncoder(nn.Module):
    """
    Encodes Sentinel-1 SAR backscatter time series.
    Architecture: 1D temporal conv (local patterns) → BiLSTM (global sequence).
    SAR is sensitive to vegetation structure and soil moisture, complementary
    to optical at different time lags.
    """

    def __init__(self, in_channels: int = 2, out_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )
        self.lstm = nn.LSTM(
            input_size=64, hidden_size=out_dim // 2,
            num_layers=2, batch_first=True,
            bidirectional=True, dropout=0.1
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, 2) — VV, VH backscatter
        Returns:
            (B, T, out_dim)
        """
        x = rearrange(x, 'b t c -> b c t')
        x = self.conv(x)
        x = rearrange(x, 'b c t -> b t c')
        out, _ = self.lstm(x)
        return out


class OpticalEncoder(nn.Module):
    """
    Encodes Sentinel-2 multispectral time series.
    Uses a lightweight Transformer over the spectral dimension per timestep,
    then a temporal Transformer across timesteps.

    In production: swap spectral_transformer for a frozen Prithvi/SatMAE
    backbone, using its [CLS] token per timestep as the temporal feature.
    """

    def __init__(self, n_bands: int = 12, out_dim: int = 256, n_heads: int = 4):
        super().__init__()
        # Per-timestep spectral embedding
        self.band_embed = nn.Linear(n_bands, out_dim)
        self.spectral_norm = nn.LayerNorm(out_dim)

        # Temporal self-attention over optical sequence
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=out_dim, nhead=n_heads, dim_feedforward=out_dim * 2,
            dropout=0.1, batch_first=True, activation='gelu', norm_first=True
        )
        self.temporal_attn = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, n_bands)
        Returns:
            (B, T, out_dim)
        """
        x = self.spectral_norm(self.band_embed(x))
        return self.temporal_attn(x)


class ClimateEncoder(nn.Module):
    """
    Encodes CHIRPS rainfall and derived climate variables.
    Uses a simplified Informer-style sparse attention for long-range
    rainfall patterns (ENSO teleconnections, seasonal anomalies).
    """

    def __init__(self, n_vars: int = 4, out_dim: int = 64):
        super().__init__()
        self.input_proj = nn.Linear(n_vars, out_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=out_dim, nhead=2, dim_feedforward=out_dim * 2,
            dropout=0.1, batch_first=True, activation='gelu', norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, n_vars) — rainfall_sum, spi_90d, spi_180d, anomaly
        Returns:
            (B, T, out_dim)
        """
        return self.encoder(self.input_proj(x))


# ---------------------------------------------------------------------------
# Cross-Modal Attention Fusion (Novel Contribution)
# ---------------------------------------------------------------------------

class CrossModalAttentionFusion(nn.Module):
    """
    Bidirectional cross-attention between SAR and Optical encodings.

    The key insight: SAR is informative during cloud cover events (when optical
    is unavailable) and captures soil moisture and vegetation structure at
    different angles. Cross-attention lets each modality learn WHEN to attend
    to the other, rather than always combining them with equal weight.

    Attention weights can be visualized to show cloud cover compensation.
    """

    def __init__(self, sar_dim: int = 128, optical_dim: int = 256,
                 climate_dim: int = 64, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.fused_dim = sar_dim + optical_dim

        # SAR queries optical: "where does optical confirm SAR signal?"
        self.sar_queries_optical = nn.MultiheadAttention(
            embed_dim=sar_dim, kdim=optical_dim, vdim=optical_dim,
            num_heads=n_heads, batch_first=True, dropout=dropout
        )
        # Optical queries SAR: "where does SAR provide structure info?"
        self.optical_queries_sar = nn.MultiheadAttention(
            embed_dim=optical_dim, kdim=sar_dim, vdim=sar_dim,
            num_heads=n_heads, batch_first=True, dropout=dropout
        )

        # Climate is added as a residual (smaller, don't cross-attend)
        self.climate_proj = nn.Linear(climate_dim, self.fused_dim)

        self.norm_sar = nn.LayerNorm(sar_dim)
        self.norm_opt = nn.LayerNorm(optical_dim)
        self.out_norm = nn.LayerNorm(self.fused_dim)

    def forward(self, sar: torch.Tensor, optical: torch.Tensor,
                climate: torch.Tensor):
        """
        Args:
            sar:     (B, T, sar_dim)
            optical: (B, T, optical_dim)
            climate: (B, T, climate_dim)
        Returns:
            fused:         (B, T, sar_dim + optical_dim)
            sar_attn_w:    (B, T, T) — SAR→Optical attention weights
            optical_attn_w:(B, T, T) — Optical→SAR attention weights
        """
        # SAR enhanced by optical context
        sar_enhanced, sar_attn_w = self.sar_queries_optical(
            sar, optical, optical, need_weights=True, average_attn_weights=True
        )
        sar_out = self.norm_sar(sar + sar_enhanced)

        # Optical enhanced by SAR structure
        opt_enhanced, opt_attn_w = self.optical_queries_sar(
            optical, sar, sar, need_weights=True, average_attn_weights=True
        )
        opt_out = self.norm_opt(optical + opt_enhanced)

        # Concatenate and add climate residual
        fused = torch.cat([sar_out, opt_out], dim=-1)
        fused = fused + self.climate_proj(climate)
        fused = self.out_norm(fused)

        return fused, sar_attn_w, opt_attn_w


# ---------------------------------------------------------------------------
# Main VegFormer Model
# ---------------------------------------------------------------------------

class VegFormer(nn.Module):
    """
    VegFormer: Cross-Modal Temporal Transformer for Vegetation Dynamics.

    Input shape: dict with keys:
        'sar':     (B, T, 2)      — Sentinel-1 VV, VH
        'optical': (B, T, n_bands)— Sentinel-2 spectral bands
        'climate': (B, T, n_vars) — CHIRPS rainfall + derived vars
        'dates':   (B, T)         — Julian day per timestep

    Outputs dict with:
        'veg_type':    (B, n_veg_classes)
        'degradation': (B, n_deg_classes)
        'risk_score':  (B, 1)
        'embeddings':  (B, d_model)        — for UMAP / clustering
        'sar_attn':    (B, T, T)           — cross-modal attention weights
        'opt_attn':    (B, T, T)           — cross-modal attention weights
    """

    def __init__(
        self,
        n_bands: int = 12,
        n_climate_vars: int = 4,
        d_model: int = 448,
        n_heads: int = 8,
        n_layers: int = 4,
        n_veg_classes: int = 8,
        n_deg_classes: int = 4,   # recovering, stable, degrading, collapsed
        dropout: float = 0.1,
        max_seq_len: int = 92,    # ~4 years at 16-day composites
    ):
        super().__init__()
        self.d_model = d_model

        # Per-modality encoders
        self.sar_encoder = SAREncoder(in_channels=2, out_dim=128)
        self.optical_encoder = OpticalEncoder(n_bands=n_bands, out_dim=256)
        self.climate_encoder = ClimateEncoder(n_vars=n_climate_vars, out_dim=64)

        # Cross-modal fusion (→ 128 + 256 = 384 + 64 = 448 = d_model)
        self.fusion = CrossModalAttentionFusion(
            sar_dim=128, optical_dim=256, climate_dim=64,
            n_heads=4, dropout=dropout
        )
        # d_model = sar_dim + optical_dim (climate added as residual after fusion)

        # Seasonal positional encoding
        self.pos_enc = SeasonalPositionalEncoding(d_model)

        # [CLS] token for sequence-level prediction
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Main temporal Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
            activation='gelu', norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

        # Task heads
        self.veg_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_veg_classes)
        )
        self.deg_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_deg_classes)
        )
        self.risk_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # risk in [0, 1]
        )

        # MSM decoder (used during pre-training only)
        self.msm_decoder = nn.Linear(d_model, n_bands + 2 + n_climate_vars)

    def forward(self, batch: dict) -> dict:
        sar = batch['sar']       # (B, T, 2)
        optical = batch['optical']  # (B, T, n_bands)
        climate = batch['climate']  # (B, T, n_vars)
        dates = batch['dates']   # (B, T)

        B, T, _ = sar.shape

        # Encode each modality
        sar_feats = self.sar_encoder(sar)          # (B, T, 128)
        opt_feats = self.optical_encoder(optical)   # (B, T, 256)
        clim_feats = self.climate_encoder(climate)  # (B, T, 64)

        # Cross-modal fusion
        fused, sar_attn, opt_attn = self.fusion(
            sar_feats, opt_feats, clim_feats
        )  # (B, T, 448)

        # Add seasonal positional encoding
        fused = self.pos_enc(fused, dates)

        # Prepend [CLS] token
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        seq = torch.cat([cls, fused], dim=1)    # (B, T+1, d_model)

        # Temporal Transformer
        out = self.transformer(seq)  # (B, T+1, d_model)
        out = self.norm(out)

        # Extract [CLS] for sequence-level tasks
        cls_out = out[:, 0]  # (B, d_model)

        return {
            'veg_type':    self.veg_head(cls_out),     # (B, n_veg_classes)
            'degradation': self.deg_head(cls_out),      # (B, n_deg_classes)
            'risk_score':  self.risk_head(cls_out),     # (B, 1)
            'embeddings':  cls_out,                     # (B, d_model)
            'sar_attn':    sar_attn,                    # (B, T, T)
            'opt_attn':    opt_attn,                    # (B, T, T)
            'token_feats': out[:, 1:],                  # (B, T, d_model) for MSM
        }


# ---------------------------------------------------------------------------
# Masked Sensor Modeling (Self-Supervised Pre-training)
# ---------------------------------------------------------------------------

class MaskedSensorModeling(nn.Module):
    """
    Self-supervised pre-training objective.

    Masks random (sensor, timestep) blocks and predicts their values.
    Masking is applied independently per sensor, encouraging the model
    to learn cross-sensor dependencies (SAR ↔ Optical complementarity).

    Masking strategy:
      - For each batch element, randomly select 40% of timesteps per sensor
      - Replace with learned [MASK] token (not zeros — avoids shortcut)
      - Predict original values using MSM decoder head
    """

    def __init__(self, model: VegFormer, mask_ratio: float = 0.4):
        super().__init__()
        self.model = model
        self.mask_ratio = mask_ratio
        d_model = model.d_model

        # Learned mask tokens per modality
        self.sar_mask = nn.Parameter(torch.zeros(1, 1, 2))
        self.optical_mask = nn.Parameter(torch.zeros(1, 1, 14))
        self.climate_mask = nn.Parameter(torch.zeros(1, 1, 4))
        nn.init.trunc_normal_(self.sar_mask, std=0.02)
        nn.init.trunc_normal_(self.optical_mask, std=0.02)
        nn.init.trunc_normal_(self.climate_mask, std=0.02)

    def _apply_mask(self, x: torch.Tensor, mask_token: torch.Tensor):
        """Apply random temporal masking to a modality tensor."""
        B, T, C = x.shape
        n_mask = int(T * self.mask_ratio)
        mask = torch.zeros(B, T, dtype=torch.bool, device=x.device)
        for i in range(B):
            idx = torch.randperm(T, device=x.device)[:n_mask]
            mask[i, idx] = True
        x_masked = x.clone()
        x_masked[mask] = mask_token.expand(B, T, C)[mask]
        return x_masked, mask

    def forward(self, batch: dict) -> dict:
        sar_masked, sar_mask = self._apply_mask(batch['sar'], self.sar_mask)
        opt_masked, opt_mask = self._apply_mask(batch['optical'], self.optical_mask)
        clim_masked, clim_mask = self._apply_mask(batch['climate'], self.climate_mask)

        masked_batch = {**batch, 'sar': sar_masked,
                        'optical': opt_masked, 'climate': clim_masked}
        out = self.model(masked_batch)

        # Reconstruct original values at masked positions
        token_feats = out['token_feats']  # (B, T, d_model)
        recon = self.model.msm_decoder(token_feats)  # (B, T, 18)

        # Compute MSE only at masked positions (per modality)
        target = torch.cat([batch['sar'], batch['optical'], batch['climate']], dim=-1)
        all_mask = sar_mask | opt_mask | clim_mask
        loss = F.mse_loss(recon[all_mask], target[all_mask])

        return {'loss': loss, 'mask': all_mask, 'recon': recon}


# ---------------------------------------------------------------------------
# Model instantiation helper
# ---------------------------------------------------------------------------

def build_vegformer(config: dict = None) -> VegFormer:
    """
    Build VegFormer from config dict. Defaults match the paper configuration.

    Example:
        model = build_vegformer({'n_layers': 6, 'n_heads': 8})
    """
    defaults = dict(
        n_bands=12, n_climate_vars=4, d_model=448,
        n_heads=8, n_layers=4, n_veg_classes=8,
        n_deg_classes=4, dropout=0.1, max_seq_len=92
    )
    if config:
        defaults.update(config)
    return VegFormer(**defaults)


if __name__ == '__main__':
    # Smoke test
    model = build_vegformer()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"VegFormer parameters: {n_params:,}")

    B, T = 4, 46  # batch=4, 46 timesteps (~2 years)
    dummy_batch = {
        'sar':     torch.randn(B, T, 2),
        'optical': torch.randn(B, T, 12),
        'climate': torch.randn(B, T, 4),
        'dates':   torch.randint(1, 366, (B, T)).float()
    }
    out = model(dummy_batch)
    print("Output shapes:")
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {tuple(v.shape)}")
