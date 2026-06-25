import torch

def load_pos_weights(path, device):
    return torch.load(path, map_location=device)