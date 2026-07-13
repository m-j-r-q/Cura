import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image


class GradCAM:

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model        = model
        self.target_layer = target_layer
        self.activations  = None
        self.gradients    = None

        self._forward_hook  = target_layer.register_forward_hook(self._save_activations)
        self._backward_hook = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach().float()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach().float()

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)  # [1, 14]

        # Zero existing gradients
        self.model.zero_grad()

        # Backward pass on the target class score only
        target = output[0, class_idx]
        target.backward()

        # Global average pool the gradients, one weight per channel
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # [1, C, 1, 1]

        # Weighted sum of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [1, 1, H, W]

        # ReLU, we only care about features that increase the class score
        cam = torch.relu(cam)

        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()

        # Upsample to input image size
        cam = cv2.resize(cam, (224, 224))

        return cam

    def overlay(self, cam: np.ndarray, original_image: Image.Image) -> Image.Image:

        # Convert heatmap to color
        heatmap = np.uint8(255 * cam)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Resize original image to 224x224
        original = np.array(original_image.convert('RGB').resize((224, 224)))

        # Blend
        overlaid = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

        return Image.fromarray(overlaid)

    def remove_hooks(self):
        self._forward_hook.remove()
        self._backward_hook.remove()