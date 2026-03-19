"""
VegFormer Training Pipeline
===========================
Prefect flow orchestrating the 3-stage training protocol:
  Stage 1: Masked Sensor Modeling (self-supervised)
  Stage 2: Weak supervision fine-tuning (MODIS pseudo-labels)
  Stage 3: Full fine-tuning (Braun-Blanquet field labels)

MLflow tracks all experiments, metrics, and artifacts.

Usage:
    python training/train_pipeline.py --config configs/pretrain_config.yaml
"""

import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from prefect import flow, task, get_run_logger
import mlflow
import mlflow.pytorch

from models.vegformer import build_vegformer, MaskedSensorModeling


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

@task(name="load_config", retries=1)
def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


@task(name="setup_mlflow")
def setup_mlflow(config: dict) -> str:
    mlflow.set_tracking_uri(config.get('mlflow_uri', 'http://localhost:5000'))
    mlflow.set_experiment(config.get('experiment_name', 'vegformer'))
    run = mlflow.start_run(run_name=config.get('run_name', 'vegformer-run'))
    mlflow.log_params(config)
    return run.info.run_id


@task(name="pretrain_msm", timeout_seconds=86400)
def pretrain_masked_sensor_modeling(
    config: dict,
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device
) -> nn.Module:
    """Stage 1: Self-supervised Masked Sensor Modeling."""
    logger = get_run_logger()
    logger.info("Starting MSM pre-training")

    msm = MaskedSensorModeling(
        model=model,
        mask_ratio=config.get('mask_ratio', 0.4)
    ).to(device)

    optimizer = torch.optim.AdamW(
        msm.parameters(),
        lr=config.get('pretrain_lr', 1e-4),
        weight_decay=0.05
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.get('pretrain_epochs', 50)
    )

    best_loss = float('inf')
    for epoch in range(config.get('pretrain_epochs', 50)):
        msm.train()
        epoch_loss = 0.0

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            out = msm(batch)
            loss = out['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(msm.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        scheduler.step()

        mlflow.log_metric('pretrain_loss', epoch_loss, step=epoch)
        logger.info(f"Epoch {epoch+1}/{config['pretrain_epochs']} | MSM loss: {epoch_loss:.4f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            ckpt_path = Path(config.get('checkpoint_dir', 'checkpoints')) / 'msm_best.pt'
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)
            mlflow.log_artifact(str(ckpt_path))

    return model


@task(name="finetune_weak_labels", timeout_seconds=43200)
def finetune_weak_supervision(
    config: dict,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device
) -> nn.Module:
    """Stage 2: Fine-tune with MODIS weak pseudo-labels."""
    logger = get_run_logger()
    logger.info("Starting weak-supervision fine-tuning with MODIS labels")

    # Freeze encoders, train only task heads
    for name, param in model.named_parameters():
        if 'head' not in name and 'fusion' not in name:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters (heads only): {trainable:,}")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.get('weak_lr', 5e-5),
        weight_decay=0.01
    )

    # Label smoothing for noisy MODIS pseudo-labels
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    for epoch in range(config.get('weak_epochs', 20)):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out['veg_type'], batch['veg_label'])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(batch)
                preds = out['veg_type'].argmax(dim=-1)
                val_correct += (preds == batch['veg_label']).sum().item()
                val_total += len(batch['veg_label'])

        val_acc = val_correct / val_total
        mlflow.log_metric('weak_train_loss', train_loss / len(train_loader), step=epoch)
        mlflow.log_metric('weak_val_acc', val_acc, step=epoch)
        logger.info(f"Weak FT Epoch {epoch+1} | val acc: {val_acc:.3f}")

    # Unfreeze for next stage
    for param in model.parameters():
        param.requires_grad = True

    return model


@task(name="finetune_full", timeout_seconds=43200)
def finetune_full(
    config: dict,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device
) -> nn.Module:
    """Stage 3: Full fine-tuning with Braun-Blanquet field labels."""
    logger = get_run_logger()
    logger.info("Starting full fine-tuning with Braun-Blanquet labels")

    # Discriminative learning rates: lower for encoders, higher for heads
    param_groups = [
        {'params': model.sar_encoder.parameters(),     'lr': 1e-5},
        {'params': model.optical_encoder.parameters(), 'lr': 1e-5},
        {'params': model.climate_encoder.parameters(), 'lr': 1e-5},
        {'params': model.fusion.parameters(),          'lr': 2e-5},
        {'params': model.transformer.parameters(),     'lr': 2e-5},
        {'params': model.veg_head.parameters(),        'lr': 1e-4},
        {'params': model.deg_head.parameters(),        'lr': 1e-4},
        {'params': model.risk_head.parameters(),       'lr': 1e-4},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=5, factor=0.5
    )

    veg_criterion = nn.CrossEntropyLoss()
    deg_criterion = nn.CrossEntropyLoss()
    risk_criterion = nn.BCELoss()

    best_val_f1 = 0.0
    patience_counter = 0

    for epoch in range(config.get('finetune_epochs', 30)):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            out = model(batch)

            # Multi-task loss
            loss_veg  = veg_criterion(out['veg_type'], batch['veg_label'])
            loss_deg  = deg_criterion(out['degradation'], batch['deg_label'])
            loss_risk = risk_criterion(
                out['risk_score'].squeeze(), batch['risk_label'].float()
            )
            loss = loss_veg + 0.5 * loss_deg + 0.3 * loss_risk
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        # Validation metrics
        metrics = evaluate_model(model, val_loader, device)
        scheduler.step(metrics['veg_f1'])

        mlflow.log_metrics({
            'full_train_loss': train_loss / len(train_loader),
            'veg_OA': metrics['veg_oa'],
            'veg_F1': metrics['veg_f1'],
            'deg_F1': metrics['deg_f1'],
            'kappa':  metrics['kappa'],
        }, step=epoch)

        logger.info(
            f"Epoch {epoch+1} | Veg F1: {metrics['veg_f1']:.3f} | "
            f"Deg F1: {metrics['deg_f1']:.3f} | κ: {metrics['kappa']:.3f}"
        )

        if metrics['veg_f1'] > best_val_f1:
            best_val_f1 = metrics['veg_f1']
            patience_counter = 0
            ckpt_path = Path(config.get('checkpoint_dir', 'checkpoints')) / 'vegformer_best.pt'
            torch.save(model.state_dict(), ckpt_path)
            mlflow.pytorch.log_model(model, 'best_model')
        else:
            patience_counter += 1
            if patience_counter >= 10:
                logger.info("Early stopping triggered")
                break

    return model


def evaluate_model(model, loader, device) -> dict:
    """Compute OA, macro F1, and Cohen's kappa on validation set."""
    from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

    model.eval()
    all_veg_preds, all_veg_true = [], []
    all_deg_preds, all_deg_true = [], []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch)
            all_veg_preds.extend(out['veg_type'].argmax(-1).cpu().tolist())
            all_veg_true.extend(batch['veg_label'].cpu().tolist())
            all_deg_preds.extend(out['degradation'].argmax(-1).cpu().tolist())
            all_deg_true.extend(batch['deg_label'].cpu().tolist())

    return {
        'veg_oa':  accuracy_score(all_veg_true, all_veg_preds),
        'veg_f1':  f1_score(all_veg_true, all_veg_preds, average='macro'),
        'deg_f1':  f1_score(all_deg_true, all_deg_preds, average='macro'),
        'kappa':   cohen_kappa_score(all_veg_true, all_veg_preds),
    }


# ---------------------------------------------------------------------------
# Main Prefect Flow
# ---------------------------------------------------------------------------

@flow(name="VegFormer Training Pipeline", log_prints=True)
def vegformer_pipeline(config_path: str = 'configs/pretrain_config.yaml'):
    """
    Full 3-stage training pipeline orchestrated by Prefect.
    Tracks all experiments with MLflow.
    """
    config = load_config(config_path)
    run_id = setup_mlflow(config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_vegformer(config.get('model_config', {})).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"VegFormer loaded: {n_params:,} parameters | device: {device}")
    mlflow.log_param('n_parameters', n_params)

    # Note: Replace these with your actual dataset classes
    # from data.datasets.vegformer_dataset import VegFormerDataset
    # train_ds = VegFormerDataset(config['data_dir'], split='train')
    # In practice, use your GEE-exported patches here

    print("Pipeline complete. MLflow run:", run_id)
    mlflow.end_run()
    return run_id


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/pretrain_config.yaml')
    args = parser.parse_args()
    vegformer_pipeline(config_path=args.config)
