import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

from src.model import build_model
from api.pipeline import run_pipeline

from safetensors.torch import load_file

app = FastAPI(
    title="Cura API",
    description="Chest X-ray Analysis Pipeline",
    version="1.0.0"
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



ARCHITECTURE = "densenet121"
CHECKPOINT   = os.path.join(
    os.path.dirname(__file__), '..', 'checkpoints', 'densenet121_best.safetensors'
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Loading {ARCHITECTURE} on {device}...")

model = build_model(ARCHITECTURE, pretrained=False)
state_dict = load_file(CHECKPOINT, device=str(device))
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

print("Model loaded.")


@app.get("/")
def root():
    return {"status": "Cura API is running"}


@app.get("/health")
def health():
    return {
        "status":       "healthy",
        "model":        ARCHITECTURE,
        "device":       str(device),
    }


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Must be JPEG or PNG."
        )

    try:
        contents = await file.read()
        image    = Image.open(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {str(e)}")

    try:
        report = run_pipeline(image, model, ARCHITECTURE, device)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    return report