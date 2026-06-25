# Cura: An AI-Powered Application for Chest X-Ray Analysis

Cura is an end-to-end medical imaging pipeline that performs automated analysis of chest X-rays across 14 pathologies. It combines image quality assessment, multi-label disease classification, anatomically grounded explainability, and a full-stack web interface.

---

## Demo

> Upload a chest X-ray to receive a structured diagnostic report with disease classifications, confidence scores, affected anatomical regions, and Grad-CAM activation maps.

- **Frontend:** https://cura-xray.vercel.app
- **API:** https://mjrq-cura.hf.space/docs

---

## Features

- **Image Quality Assessment**: Rejects low-quality images before inference using blur detection (Laplacian variance), contrast assessment, and exposure analysis. Thresholds calibrated from 500 NIH dataset samples.
- **Multi-Label Classification**: Classifies 14 chest pathologies simultaneously using fine-tuned CNNs. Each image can have multiple simultaneous findings.
- **CNN Benchmark**: Trains and evaluates DenseNet121, ResNet50, and EfficientNet-B0 on the NIH ChestX-ray14 dataset.
- **Explainability**: Grad-CAM heatmaps grounded in TorchXRayVision anatomical segmentation, producing region labels such as "Right lower lobe" rather than raw pixel coordinates.
- **Calibrated Thresholds**: Per-class confidence thresholds derived by maximizing F1 scores, replacing a naive fixed cutoff.
- **REST API**: FastAPI backend serving structured JSON diagnostic reports.
- **React Frontend**: Drag-and-drop interface with real-time results, confidence bars, and heatmap overlays.

---

## Architecture

1. Chest X-Ray Image
2. Image Quality Assessment (Open CV)
3. DenseNet121 Inference (PyTorch)
4. Grad-CAM and TorchXRayVision Segmentation
5. Structured JSON Report
6. React Frontend

---

## Results

### CNN Benchmark — NIH ChestX-ray14 (Mean AUC)

| Architecture    | Mean AUC |
|-----------------|----------|
| DenseNet121     | 0.8289   |
| ResNet50        | 0.8284   |
| EfficientNet-B0 | 0.8219   |

### Per-Class AUC (DenseNet121)

| Disease            | AUC    |
|--------------------|--------|
| Atelectasis        | 0.7978 |
| Cardiomegaly       | 0.9140 |
| Effusion           | 0.8796 |
| Infiltration       | 0.7092 |
| Mass               | 0.8357 |
| Nodule             | 0.7355 |
| Pneumonia          | 0.7665 |
| Pleural Thickening | 0.7926 |
| Pneumothorax       | 0.8587 |
| Consolidation      | 0.7987 |
| Edema              | 0.8887 |
| Emphysema          | 0.9263 |
| Fibrosis           | 0.8200 |
| Hernia             | 0.8808 |

*Results comparable to the original CheXNet paper (Rajpurkar et al., 2017) which reported mean AUC of 0.841.*

---

## Dataset

**NIH ChestX-ray14** — 83,691 frontal chest X-rays (224×224) across 14 disease labels.

- Source: [Kaggle — NIH Chest X-ray 14 (224x224 resized)](https://www.kaggle.com/datasets/khanfashee/nih-chest-x-ray-14-224x224-resized)
- Split by patient ID to prevent data leakage (16,428 train / 2,054 val / 2,054 test patients)
- Class imbalance handled via BCEWithLogitsLoss pos_weight (up to 439x for Hernia)
- Dataset-specific normalization: mean [0.509, 0.509, 0.509], std [0.251, 0.251, 0.251]

---

## Setup

### Backend

```bash
# Create and activate environment
conda create -n cura python=3.11
conda activate cura

# Install dependencies
pip install -r backend/requirements.txt

# Place model checkpoint in backend/checkpoints/
# densenet121_best.pt

# Start API server
uvicorn backend.api.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`

---

## API

### `POST /analyze`

Upload a chest X-ray image and receive a full diagnostic report.

**Request:** `multipart/form-data` with `file` field (PNG or JPEG)

**Response Example:**
```json
{
  "passed_quality": true,
  "rejection_reason": null,
  "image_quality": "Good",
  "quality_metrics": {
    "blur_score": 299.58,
    "contrast_score": 63.41,
    "mean_brightness": 171.06
  },
  "model": "densenet121",
  "diagnoses": [
    {
      "disease": "Atelectasis",
      "confidence": 0.6794,
      "threshold_used": 0.65,
      "affected_region": "Right middle lobe",
      "heatmap_base64": "..."
    }
  ],
}
```

### `GET /health`

Returns server and model status.

---

## Technical Decisions

**Why DenseNet121?** 

Achieves the highest AUC with 3x fewer parameters than ResNet50 with second highest AUC.

**Why patient-level splits?** 

The NIH dataset contains multiple scans per patient. Image-level splits would allow patient anatomy to appear in both train and test sets, inflating metrics. All images from one patient are assigned to exactly one split.

**Why per-class thresholds?** 

A single threshold treats all 14 diseases equally. Hernia (0.5% prevalence, 439x class weight) requires 90% confidence before being reported. Infiltration (most common disease) only requires 50%. Thresholds derived by maximizing F1 Score.

**Why TorchXRayVision for region detection?** 

Raw Grad-CAM coordinates map poorly to anatomy. Multiplying the Grad-CAM heatmap by TorchXRayVision's anatomical segmentation masks constrains activation to clinically meaningful regions and produces human-readable labels.

**Why dataset-specific normalization?** 

X-rays have a fundamentally different pixel distribution from natural images (mean 0.509 vs ImageNet 0.485). Computed from 5,000 random samples using the computational variance identity.

---

## Known Limitations

- Rotation detection disabled: Hough line transform unreliable on chest X-rays due to absence of strong dominant lines
- Uncertainty estimation not yet implemented: requires MC Dropout layer injection into pretrained weights
- Trained on 30k image subset per architecture for 7 epochs and than around 90k images for the last 3 epochs. This was due to computation constraints.
- LLM clinical report generation from JSON output planned but not yet implemented.

---

## Future Work

- Monte Carlo Dropout uncertainty estimation
- LLM-generated clinical narrative reports
- MLflow experiment tracking integration

---

## References

- Rajpurkar et al. (2017). *CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning*
- Wang et al. (2017). *ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks*
- Selvaraju et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*
- Gal & Ghahramani (2016). *Dropout as a Bayesian Approximation*
- Cohen et al. (2022). *TorchXRayVision: A library of chest X-ray datasets and models*

---

## License

MIT