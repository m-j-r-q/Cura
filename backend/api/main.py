import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from PIL import Image
import io
import json

from src.model import build_model
from api.pipeline import run_pipeline

app = FastAPI(
    title="Cura API",
    description="An AI-powered application for chest X-ray analysis.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:8000",
        "https://cura-xray.vercel.app",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

REPO_ID = "mjrq/cura-chest-xray"
ENSEMBLE_ARCHITECTURES = [
    'densenet121',
    'densenet169',
    'convnext_tiny',
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Loading ensemble on {device}...")

ensemble_models = []
for arch in ENSEMBLE_ARCHITECTURES:
    print(f"  Loading {arch}...")
    model = build_model(arch, pretrained=False, dropout_p=0.2)

    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=f"{arch}_best.safetensors"
    )
    state_dict = load_file(path, device=str(device))
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    ensemble_models.append(model)

thresholds_path = hf_hub_download(
    repo_id=REPO_ID,
    filename="ensemble_thresholds.json"
)
with open(thresholds_path) as f:
    ENSEMBLE_THRESHOLDS = json.load(f)

print(f"Ensemble loaded — {len(ensemble_models)} models.")

@app.get("/")
def root():
    return {"status": "Cura API running", "ensemble_size": len(ensemble_models)}

@app.get("/health")
def health():
    return {
        "status":     "healthy",
        "models":     ENSEMBLE_ARCHITECTURES,
        "device":     str(device),
    }

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail=f"Invalid file type: {file.content_type}")

    try:
        contents = await file.read()
        image    = Image.open(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {str(e)}")

    try:
        report = run_pipeline(
            image,
            ensemble_models,
            ENSEMBLE_ARCHITECTURES,
            ENSEMBLE_THRESHOLDS,
            device
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    return report