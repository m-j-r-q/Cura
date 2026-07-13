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

        output = self.model(input_tensor)
        self.model.zero_grad()

        target = output[0, class_idx]
        target.backward()

        weights = self.gradients.float().mean(dim=[2, 3], keepdim=True)
        cam     = (weights * self.activations.float()).sum(dim=1, keepdim=True)

        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        if cam.max() > 0:
            cam = cam / cam.max()

        # If region too broad increase threshold.
        threshold = np.percentile(cam, 95)
        cam = np.where(cam >= threshold, cam, 0.0)

        if cam.max() > 0:
            cam = cam / cam.max()

        # If peak too bright reduce power.
        cam = np.power(cam, 10)

        cam = cv2.resize(cam, (224, 224))

        return cam


    def overlay(self, cam: np.ndarray, original_image: Image.Image) -> Image.Image:
        heatmap = np.uint8(255 * cam)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        original = np.array(original_image.convert('RGB').resize((224, 224)))

        alpha = cam[:, :, np.newaxis]
        overlaid = (
            original * (1 - alpha * 0.3) + heatmap * (alpha * 0.3)
        ).astype(np.uint8)

        return Image.fromarray(overlaid)

    def remove_hooks(self):
        self._forward_hook.remove()
        self._backward_hook.remove()