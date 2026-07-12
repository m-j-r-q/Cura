import torch
import numpy as np
import base64
import io
import sys
import os
import json
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.dataset import DISEASES, NUM_CLASSES
from src.transforms import val_transform
from src.quality import assess_quality
from api.gradcam import GradCAM
from api.gradcam_targets import get_target_layer
from api.segmentation import get_affected_region
from src.uncertainty import mc_dropout_ensemble

QUALITY_LABEL_THRESHOLDS = {'Good': 150, 'Acceptable': 80}


def image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def get_quality_label(blur_score: float) -> str:
    if blur_score > QUALITY_LABEL_THRESHOLDS['Good']:
        return 'Good'
    if blur_score > QUALITY_LABEL_THRESHOLDS['Acceptable']:
        return 'Acceptable'
    return 'Poor'


def run_pipeline(
    image: Image.Image,
    models: list,
    architectures: list,
    thresholds: dict,
    device
) -> dict:

    # Stage 1 — Quality assessment
    quality_result = assess_quality(image)
    if not quality_result['passed']:
        return {
            'passed_quality':   False,
            'rejection_reason': quality_result['rejection_reason'],
            'quality_metrics':  quality_result['metrics'],
            'image_quality':    'Rejected',
            'diagnoses':        [],
        }

    # Stage 2 — Preprocess
    input_tensor = val_transform(
        image.convert('RGB')
    ).unsqueeze(0).to(device)

    # Stage 3 — MC Dropout ensemble uncertainty
    uncertainty_result = mc_dropout_ensemble(
        models, input_tensor, device, n_passes=10
    )
    mean_probs    = uncertainty_result['mean_probs']
    uncertainties = uncertainty_result['uncertainty']
    per_disease   = uncertainty_result['per_disease']

    # Stage 4 — Grad-CAM on primary model (first in ensemble)
    primary_model = models[0]
    primary_arch  = architectures[0]

    target_layer = get_target_layer(primary_model, primary_arch)
    gradcam      = GradCAM(primary_model, target_layer)

    diagnoses = []
    for i, disease in enumerate(DISEASES):
        prob      = float(mean_probs[i])
        threshold = thresholds.get(disease, 0.5)

        if prob >= threshold:
            cam      = gradcam.generate(input_tensor, class_idx=i)
            region   = get_affected_region(cam, image)
            overlaid = gradcam.overlay(cam, image)

            diagnoses.append({
                'disease':         disease,
                'confidence':      round(prob, 4),
                'uncertainty':     round(float(uncertainties[i]), 4),
                'confidence_level': per_disease[disease]['confidence'],
                'threshold_used':  round(threshold, 2),
                'affected_region': region,
                'heatmap_base64':  image_to_base64(overlaid),
            })

    gradcam.remove_hooks()

    return {
        'passed_quality':   True,
        'rejection_reason': None,
        'image_quality':    get_quality_label(quality_result['metrics']['blur_score']),
        'quality_metrics':  quality_result['metrics'],
        'diagnoses':        diagnoses,
        'models':           architectures,
        'n_mc_passes':      uncertainty_result['n_passes'],
    }