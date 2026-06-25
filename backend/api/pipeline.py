import torch
import numpy as np
import base64
import io
import sys
import os
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.dataset import DISEASES, NUM_CLASSES
from src.transforms import val_transform
from src.quality import assess_quality
from api.gradcam import GradCAM
from api.gradcam_targets import get_target_layer
from api.segmentation import get_affected_region

import json

# Confidence threshold — diseases above this are reported
THRESHOLDS_PATH = os.path.join(os.path.dirname(__file__), '..', 'optimal_thresholds.json')
with open(THRESHOLDS_PATH) as f:
    OPTIMAL_THRESHOLDS = json.load(f)


def image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def run_pipeline(image: Image.Image, model, architecture: str, device) -> dict:

    # Stage 1 Quality assessment
    quality_result = assess_quality(image)

    if not quality_result['passed']:
        return {
            'passed_quality':   False,
            'rejection_reason': quality_result['rejection_reason'],
            'quality_metrics':  quality_result['metrics'],
            'image_quality':    'Rejected',
            'diagnoses':        [],
        }

    # Stage 2 Preprocess
    input_tensor = val_transform(image.convert('RGB')).unsqueeze(0).to(device)

    # Stage 3 Single forward pass
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
        probs  = torch.sigmoid(logits).cpu().numpy().squeeze()

    # Stage 4 Grad-CAM for diseases above threshold
    target_layer = get_target_layer(model, architecture)
    gradcam      = GradCAM(model, target_layer)

    diagnoses = []

    for i, disease in enumerate(DISEASES):
        prob      = float(probs[i])
        threshold = OPTIMAL_THRESHOLDS.get(disease, 0.5)

        if prob >= threshold:
            cam      = gradcam.generate(input_tensor, class_idx=i)
            region   = get_affected_region(cam, image)
            overlaid = gradcam.overlay(cam, image)

            diagnoses.append({
                'disease':         disease,
                'confidence':      round(prob, 4),
                'threshold_used':  round(threshold, 2),
                'affected_region': region,
                'heatmap_base64':  image_to_base64(overlaid),
            })

    gradcam.remove_hooks()

    # Stage 5 Quality label
    blur = quality_result['metrics']['blur_score']
    if blur > 150:
        quality_label = 'Good'
    elif blur > 80:
        quality_label = 'Acceptable'
    else:
        quality_label = 'Poor'

    return {
        'passed_quality':  True,
        'rejection_reason': None,
        'image_quality':   quality_label,
        'quality_metrics': quality_result['metrics'],
        'model':           architecture,
        'diagnoses':       diagnoses,
    }