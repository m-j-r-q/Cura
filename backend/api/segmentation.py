import torch
import numpy as np
import torchxrayvision as xrv
from PIL import Image

_seg_model = None

def get_seg_model():
    global _seg_model
    if _seg_model is None:
        _seg_model = xrv.baseline_models.chestx_det.PSPNet()
        _seg_model.eval()
    return _seg_model


def get_segmentation_mask(image: Image.Image) -> dict:
    model = get_seg_model()

    image_512 = image.convert('L').resize((512, 512))

    img_array = np.array(image_512)
    img_array = img_array / 255.0 * 2048 - 1024
    img_tensor = torch.from_numpy(img_array).float()
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)

    output = output.squeeze().numpy()

    targets = model.targets
    
    def get_mask(name):
        if name in targets:
            idx = targets.index(name)
            mask = output[idx]
            import cv2
            mask = cv2.resize(mask, (224, 224))
            return (mask > 0.5).astype(np.float32)
        return np.zeros((224, 224), dtype=np.float32)

    return {
        'left_lung':  get_mask('Left Lung'),
        'right_lung': get_mask('Right Lung'),
        'cardiac':    get_mask('Cardiac Silhouette'),
    }


def get_affected_region(cam: np.ndarray, image: Image.Image) -> str:

    masks = get_segmentation_mask(image)

    left_activation  = (cam * masks['left_lung']).sum()
    right_activation = (cam * masks['right_lung']).sum()
    cardiac_activation = (cam * masks['cardiac']).sum()

    activations = {
        'left_lung':  left_activation,
        'right_lung': right_activation,
        'cardiac':    cardiac_activation,
    }

    primary_region = max(activations, key=activations.get)

    if activations[primary_region] < 0.1:
        return "Diffuse"

    if primary_region == 'cardiac':
        return "Cardiac region"

    lung_mask = masks[primary_region]
    side      = "Left" if primary_region == 'left_lung' else "Right"

    rows = np.where(lung_mask.sum(axis=1) > 0)[0]

    if len(rows) == 0:
        return f"{side} lung"

    top    = rows.min()
    bottom = rows.max()
    height = bottom - top

    if height == 0:
        return f"{side} lung"

    masked_cam = cam * lung_mask
    peak_y, _  = np.unravel_index(masked_cam.argmax(), masked_cam.shape)

    relative_pos = (peak_y - top) / height

    if relative_pos < 0.33:
        lobe = "upper lobe"
    elif relative_pos < 0.66:
        lobe = "middle lobe"
    else:
        lobe = "lower lobe"

    return f"{side} {lobe}"