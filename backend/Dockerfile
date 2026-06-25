FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=300

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --timeout 300 \
    torch==2.3.0+cpu \
    torchvision==0.18.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 300 -r requirements.txt

# Pre-download TorchXRayVision segmentation weights
RUN python -c "import torchxrayvision as xrv; xrv.baseline_models.chestx_det.PSPNet()"

COPY src/ ./src/
COPY api/ ./api/
COPY checkpoints/ ./checkpoints/
COPY optimal_thresholds.json .

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]