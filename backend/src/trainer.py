import gc
import ctypes
import torch
import torch.nn as nn
import numpy as np
import psutil
from torch.amp import autocast
from sklearn.metrics import roc_auc_score

from dataset import DISEASES

def force_memory_release():
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    running_loss  = 0.0
    total_batches = len(loader)

    for i, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        with autocast(device_type='cuda'):
            logits = model(images)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item()

        if (i + 1) % 100 == 0 or (i + 1) == total_batches:
            ram = psutil.virtual_memory()
            print(f"  Batch {i+1}/{total_batches} | "
                  f"loss: {loss.item():.4f} | "
                  f"CPU RAM: {ram.percent:.1f}% | "
                  f"scale: {scaler.get_scale():.0f}")

        del images, labels, logits, loss

    return running_loss / total_batches


def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_labels   = []
    all_probs    = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            # Validation also benefits from autocast
            with autocast(device_type='cuda'):
                logits = model(images)
                loss   = criterion(logits, labels)
                probs  = torch.sigmoid(logits)

            running_loss += loss.item()

            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    all_labels = np.concatenate(all_labels, axis=0)
    all_probs  = np.concatenate(all_probs,  axis=0)

    aucs = []
    for i in range(all_labels.shape[1]):
        if all_labels[:, i].sum() > 0:
            auc = roc_auc_score(all_labels[:, i], all_probs[:, i])
            aucs.append(auc)
        else:
            aucs.append(0.5)

    mean_auc       = np.mean(aucs)
    per_class_aucs = np.array(aucs)

    del all_labels, all_probs
    force_memory_release()

    return running_loss / len(loader), mean_auc, per_class_aucs