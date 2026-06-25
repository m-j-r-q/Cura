import torch
from safetensors.torch import load_file

def load_pos_weights(path, device):
    return torch.load(path, map_location=device, weights_only=True)

def load_model_weights(path, device):
    """Load model weights from safetensors format."""
    return load_file(path, device=str(device))