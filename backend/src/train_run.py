import gc
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import psutil
from torch.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from dataset import CuraDataset, DISEASES
from transforms import train_transform, val_transform
from model import build_model
from trainer import train_one_epoch, validate_one_epoch, force_memory_release
from utils import load_pos_weights
from safetensors.torch import save_file

class AsymmetricLoss(nn.Module):
    def __init__(
        self,
        gamma_neg=4,
        gamma_pos=1,
        clip=0.05,
        eps=1e-8
    ):
        super().__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits, targets):

        probs = torch.sigmoid(logits)

        xs_pos = probs
        xs_neg = 1.0 - probs

        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        loss_pos = targets * torch.log(
            xs_pos.clamp(min=self.eps)
        )

        loss_neg = (1 - targets) * torch.log(
            xs_neg.clamp(min=self.eps)
        )

        pt_pos = xs_pos * targets
        pt_neg = xs_neg * (1 - targets)

        focal_weight = (
            (1 - pt_pos - pt_neg)
            ** (
                self.gamma_pos * targets
                + self.gamma_neg * (1 - targets)
            )
        )

        loss = (loss_pos + loss_neg) * focal_weight

        return -loss.mean()


class PerClassAsymmetricLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

        gamma_pos_arr = [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        gamma_neg_arr = [4.0, 4.0, 4.0, 3.0, 4.0, 3.0, 3.0, 4.0, 4.0, 3.0, 4.0, 5.0, 5.0, 5.0]
        clip_arr      = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.10, 0.10, 0.10]

        self.register_buffer('gamma_pos', torch.tensor(gamma_pos_arr, dtype=torch.float32))
        self.register_buffer('gamma_neg', torch.tensor(gamma_neg_arr, dtype=torch.float32))
        self.register_buffer('clip', torch.tensor(clip_arr, dtype=torch.float32))

    def forward(self, logits, targets):
        device = logits.device
        
        # Cast data and configurations to the exact same device and dtype
        targets = targets.to(device=device, dtype=logits.dtype)
        g_pos = self.gamma_pos.to(device=device)
        g_neg = self.gamma_neg.to(device=device)
        clip_val = self.clip.to(device=device)

        probs = torch.sigmoid(logits)

        xs_pos = probs
        xs_neg = 1.0 - probs

        if clip_val is not None:
            xs_neg = (xs_neg + clip_val).clamp(max=1)

        loss_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        loss_neg = (1 - targets) * torch.log(xs_neg.clamp(min=self.eps))

        pt_pos = xs_pos * targets
        pt_neg = xs_neg * (1 - targets)

        exponent = g_pos * targets + g_neg * (1 - targets)
        focal_weight = (1 - pt_pos - pt_neg) ** exponent

        loss = (loss_pos + loss_neg) * focal_weight

        return -loss.mean()


def train_model(
    architecture: str,
    data_dir: str,
    image_dir: str,
    output_dir: str,
    epochs: int = 20,
    lr: float = 5e-5,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    dropout_p: float = 0.2,
    internal_dropout_p: float = 0,
    early_stop_patience: int = 5,
    subset_size: int = None,
):
    print(f"\n{'='*50}")
    print(f"Training {architecture}")
    print(f"{'='*50}")

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Initial CPU RAM available: {psutil.virtual_memory().available / 1e9:.1f} GB")

    train_df = pd.read_csv(f'{data_dir}/train.csv')
    val_df   = pd.read_csv(f'{data_dir}/val.csv')

    if subset_size:
        train_df = train_df.sample(subset_size, random_state=42).reset_index(drop=True)

    print(f"Training on {len(train_df)} images")
    print(f"Validating on {len(val_df)} images")
    print(f"Batch size: {batch_size}")

    train_loader = DataLoader(
        CuraDataset(train_df, image_dir, train_transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    val_loader = DataLoader(
        CuraDataset(val_df, image_dir, val_transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )

    model = build_model(
        architecture,
        pretrained=True,
        dropout_p=dropout_p,
        internal_dropout_p=internal_dropout_p
    ).to(device)

    print(f"Model: {architecture} | Dropout: {dropout_p}")

    # Uniform AdamW
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay= weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-6
    )    

    # criterion = AsymmetricLoss(
    #     gamma_neg=4,
    #     gamma_pos=1,
    #     clip=0.05
    # )

    criterion = PerClassAsymmetricLoss()

    # AMP scaler
    scaler = GradScaler(device='cuda')

    os.makedirs(f'{output_dir}/checkpoints', exist_ok=True)
    best_auc         = 0.0
    patience_counter = 0

    for epoch in range(epochs):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch+1}/{epochs} | LR: {current_lr:.2e}")
        print(f"CPU RAM available: {psutil.virtual_memory().available / 1e9:.1f} GB")

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )

        val_loss, mean_auc, per_class_aucs = validate_one_epoch(
            model, val_loader, criterion, device
        )

        scheduler.step()

        print(f"\nEpoch {epoch+1:02d} | "
              f"train_loss: {train_loss:.4f} | "
              f"val_loss: {val_loss:.4f} | "
              f"mean_auc: {mean_auc:.4f}")

        print("\nPer-class AUC:")
        for i, disease in enumerate(DISEASES):
            if i < len(per_class_aucs):
                print(f"  {disease:25} {per_class_aucs[i]:.4f}")

        if mean_auc > best_auc:
            best_auc   = mean_auc
            state_dict = model.state_dict()
            save_path  = f'{output_dir}/checkpoints/{architecture}_best.safetensors'
            save_file(state_dict, save_path)
            del state_dict
            print(f"\n  → New best AUC: {best_auc:.4f} — saved")
        else:
            patience_counter += 1
            print(f"\n  No improvement. Patience: {patience_counter}/{early_stop_patience}")
            if patience_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        del train_loss, val_loss, mean_auc, per_class_aucs
        force_memory_release()
        print(f"CPU RAM after cleanup: {psutil.virtual_memory().available / 1e9:.1f} GB")

    print(f"\nTraining complete. Best AUC: {best_auc:.4f}")

    del model, optimizer, scheduler, criterion, scaler
    del train_loader, val_loader
    force_memory_release()

    return best_auc