import cv2
import numpy as np
from PIL import Image
from api.segmentation import get_segmentation_mask

# Thresholds — tuned for chest X-rays
BLUR_THRESHOLD        = 39.57   # Laplacian variance below this = blurry
CONTRAST_THRESHOLD    = 43.26   # Pixel std below this = low contrast
BRIGHTNESS_LOW        = 88.34   # Mean pixel below this = underexposed
BRIGHTNESS_HIGH       = 184.42  # Mean pixel above this = overexposed
ROTATION_THRESHOLD     = 20

def detect_blur(gray: np.ndarray) -> tuple[float, bool]:

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    score = laplacian.var()
    return score, score < BLUR_THRESHOLD

def detect_contrast_and_exposure(gray: np.ndarray) -> tuple[float, float, bool, bool, bool]:

    gray_8bit = (gray / gray.max() * 255).astype(np.uint8) if gray.max() > 0 else gray.astype(np.uint8)
    
    contrast_score  = float(gray_8bit.std())
    mean_brightness = float(gray_8bit.mean())
    
    low_contrast  = contrast_score  < CONTRAST_THRESHOLD
    underexposed  = mean_brightness < BRIGHTNESS_LOW
    overexposed   = mean_brightness > BRIGHTNESS_HIGH
    
    return contrast_score, mean_brightness, low_contrast, underexposed, overexposed

def detect_rotation_from_segmentation(image: Image.Image) -> tuple[float, bool]:
    """
    Use TorchXRayVision segmentation to detect rotation.
    The line connecting the centroids of left and right lung
    should be approximately horizontal in a properly oriented X-ray.
    """
    masks = get_segmentation_mask(image)
    
    left_mask  = masks['left_lung']
    right_mask = masks['right_lung']
    
    # If either lung isn't detected, skip rotation check
    if left_mask.sum() < 100 or right_mask.sum() < 100:
        return 0.0, False
    
    # Find centroid of each lung mask
    left_coords  = np.argwhere(left_mask > 0)
    right_coords = np.argwhere(right_mask > 0)
    
    left_centroid  = left_coords.mean(axis=0)   # [y, x]
    right_centroid = right_coords.mean(axis=0)  # [y, x]
    
    # Angle of line connecting centroids from horizontal
    dy = right_centroid[0] - left_centroid[0]
    dx = right_centroid[1] - left_centroid[1]
    angle = abs(np.degrees(np.arctan2(dy, dx)))
    
    # Should be close to 0 or 180 (horizontal line)
    deviation = min(angle, abs(180 - angle))
    
    return deviation, deviation > ROTATION_THRESHOLD

def assess_quality(image: Image.Image) -> dict:
    gray = np.array(image.convert('L'))

    blur_score,  is_blurry   = detect_blur(gray)
    contrast_score, mean_brightness, low_contrast, underexposed, overexposed = \
        detect_contrast_and_exposure(gray)
    rotation, is_rotated = detect_rotation_from_segmentation(image)

    rejection_reason = None
    if is_blurry:
        rejection_reason = f"Image too blurry (score: {blur_score:.1f}, threshold: {BLUR_THRESHOLD})"
    elif low_contrast:
        rejection_reason = f"Insufficient contrast (score: {contrast_score:.1f}, threshold: {CONTRAST_THRESHOLD})"
    elif underexposed:
        rejection_reason = f"Image underexposed (brightness: {mean_brightness:.1f})"
    elif overexposed:
        rejection_reason = f"Image overexposed (brightness: {mean_brightness:.1f})"
    elif is_rotated:
        rejection_reason = f"Image rotated (rotation: {rotation:.1f})"

    passed = rejection_reason is None

    return {
        "passed":            passed,
        "rejection_reason":  rejection_reason,
        "metrics": {
            "blur_score":      round(blur_score, 2),
            "contrast_score":  round(contrast_score, 2),
            "mean_brightness": round(mean_brightness, 2),
            "rotation": round(rotation, 2),
        },
        "flags": {
            "is_blurry":     is_blurry,
            "low_contrast":  low_contrast,
            "underexposed":  underexposed,
            "overexposed":   overexposed,
            "rotated": is_rotated,
        }
    }