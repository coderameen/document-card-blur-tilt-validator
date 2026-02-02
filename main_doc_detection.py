"""
Combined document detection script.
Checks for tilt and blur, returns validate_doc_flag.
validate_doc_flag = False if image is tilted OR blurred (or both)
validate_doc_flag = True if image is NOT tilted AND NOT blurred
"""

import cv2
import numpy as np
from pathlib import Path
from docaligner import DocAligner, ModelType
import capybara as cb


# ========== Tilt Detection Functions (from main.py) ==========

def calculate_corner_angle(p1: np.ndarray, vertex: np.ndarray, p2: np.ndarray) -> float:
    """Calculate the angle at a vertex formed by three points in degrees."""
    v1 = p1 - vertex
    v2 = p2 - vertex
    
    # Calculate dot product and magnitudes
    dot_product = np.dot(v1, v2)
    mag1 = np.linalg.norm(v1)
    mag2 = np.linalg.norm(v2)
    
    if mag1 == 0 or mag2 == 0:
        return 0.0
    
    # Calculate angle in radians, then convert to degrees
    cos_angle = np.clip(dot_product / (mag1 * mag2), -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_angle))
    
    return angle


def is_tilted(polygon: np.ndarray, threshold: float = 10.0) -> bool:
    """
    Check if the document has perspective distortion (not a rectangle).
    Focuses on perspective distortion rather than rotation.
    
    Args:
        polygon: Array of 4 corner points [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        threshold: Angle threshold in degrees. If any corner angle deviates more than this
                   from 90 degrees, the document is considered to have perspective distortion.
                   Default is 10.0 degrees to allow slight perspective.
    
    Returns:
        True if perspective distortion detected, False otherwise
    """
    if polygon is None or len(polygon) != 4:
        return False
    
    # Calculate angles at each corner
    # Corner 0: angle between edge (3->0) and edge (0->1)
    angle0 = calculate_corner_angle(polygon[3], polygon[0], polygon[1])
    # Corner 1: angle between edge (0->1) and edge (1->2)
    angle1 = calculate_corner_angle(polygon[0], polygon[1], polygon[2])
    # Corner 2: angle between edge (1->2) and edge (2->3)
    angle2 = calculate_corner_angle(polygon[1], polygon[2], polygon[3])
    # Corner 3: angle between edge (2->3) and edge (3->0)
    angle3 = calculate_corner_angle(polygon[2], polygon[3], polygon[0])
    
    # For a rectangle without perspective distortion, all angles should be close to 90 degrees
    # Check if any corner deviates significantly from 90 degrees
    corner_angles = [angle0, angle1, angle2, angle3]
    
    for angle in corner_angles:
        deviation = abs(angle - 90.0)
        if deviation > threshold:
            return True  # Perspective distortion detected
    
    return False  # No significant perspective distortion


def detect_cards_in_regions(img: np.ndarray, model: DocAligner) -> list:
    """
    Detect cards by splitting image into regions and processing each.
    Tries both horizontal (top/bottom) and vertical (left/right) splits.
    
    Returns:
        List of detected polygons, each with region info
    """
    h, w = img.shape[:2]
    detected_cards = []
    
    # Try horizontal split (top and bottom halves) - most common for front/back cards
    top_half = img[0:h//2, :]
    bottom_half = img[h//2:, :]
    
    # Detect in top half
    try:
        top_polygon = model(top_half, do_center_crop=False)
        if top_polygon is not None and len(top_polygon) == 4:
            # Check if polygon is valid (all coordinates within bounds)
            if np.all(top_polygon >= 0) and np.all(top_polygon[:, 0] < w) and np.all(top_polygon[:, 1] < h//2):
                print(f"Detected card in top region - Coordinates: {top_polygon}")
                detected_cards.append({
                    'polygon': top_polygon.copy(),
                    'region': 'top',
                    'valid': True
                })
    except Exception as e:
        pass
    
    # Detect in bottom half
    try:
        bottom_polygon = model(bottom_half, do_center_crop=False)
        if bottom_polygon is not None and len(bottom_polygon) == 4:
            # Adjust coordinates to full image space (add h//2 to y coordinates)
            bottom_polygon_adjusted = bottom_polygon.copy()
            bottom_polygon_adjusted[:, 1] += h // 2
            # Check if adjusted polygon is valid
            if np.all(bottom_polygon_adjusted >= 0) and np.all(bottom_polygon_adjusted[:, 0] < w) and np.all(bottom_polygon_adjusted[:, 1] < h):
                print(f"Detected card in bottom region - Coordinates: {bottom_polygon_adjusted}")
                detected_cards.append({
                    'polygon': bottom_polygon_adjusted,
                    'region': 'bottom',
                    'valid': True
                })
    except Exception as e:
        pass
    
    # If we found 2 cards with horizontal split, return them
    if len(detected_cards) == 2:
        return detected_cards
    
    # If we didn't find two cards with horizontal split, try vertical split
    detected_cards = []
    left_half = img[:, 0:w//2]
    right_half = img[:, w//2:]
    
    # Detect in left half
    try:
        left_polygon = model(left_half, do_center_crop=False)
        if left_polygon is not None and len(left_polygon) == 4:
            # Check if polygon is valid
            if np.all(left_polygon >= 0) and np.all(left_polygon[:, 0] < w//2) and np.all(left_polygon[:, 1] < h):
                print(f"Detected card in left region - Coordinates: {left_polygon}")
                detected_cards.append({
                    'polygon': left_polygon.copy(),
                    'region': 'left',
                    'valid': True
                })
    except Exception as e:
        pass
    
    # Detect in right half
    try:
        right_polygon = model(right_half, do_center_crop=False)
        if right_polygon is not None and len(right_polygon) == 4:
            # Adjust coordinates to full image space (add w//2 to x coordinates)
            right_polygon_adjusted = right_polygon.copy()
            right_polygon_adjusted[:, 0] += w // 2
            # Check if adjusted polygon is valid
            if np.all(right_polygon_adjusted >= 0) and np.all(right_polygon_adjusted[:, 0] < w) and np.all(right_polygon_adjusted[:, 1] < h):
                print(f"Detected card in right region - Coordinates: {right_polygon_adjusted}")
                detected_cards.append({
                    'polygon': right_polygon_adjusted,
                    'region': 'right',
                    'valid': True
                })
    except Exception as e:
        pass
    
    return detected_cards


def check_tilt(img: np.ndarray, model: DocAligner) -> bool:
    """
    Check if image has perspective distortion. Handles both single and two-card scenarios.
    Returns True if perspective distortion detected, False otherwise.
    """
    # Try to detect two cards first (split image approach)
    detected_cards = detect_cards_in_regions(img, model)
    
    if len(detected_cards) == 2:
        # We have exactly two cards - check if ANY card has perspective distortion
        for card_info in detected_cards:
            polygon = card_info['polygon']
            if is_tilted(polygon):
                return True
        return False
    else:
        # Single card - try full image detection
        try:
            polygon = model(img, do_center_crop=False)
            if polygon is None or len(polygon) != 4:
                # If we can't detect corners, assume not tilted (can't determine)
                return False
            print(f"Detected single card - Coordinates: {polygon}")
            return is_tilted(polygon)
        except Exception as e:
            # If detection fails, assume not tilted (can't determine)
            return False


# ========== Blur Detection Functions (from work2.py) ==========

def load_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert BGR image to grayscale numpy array."""
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def focus_score(img: np.ndarray) -> float:
    """Variance of Laplacian; lower values typically indicate blur."""
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def focus_score_small(img: np.ndarray, target: int = 256) -> float:
    """
    Laplacian variance on a downscaled copy; downscale reduces noise-driven scores
    and penalizes global blur better.
    """
    h, w = img.shape
    scale = target / max(h, w)
    if scale < 1.0:
        resized = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        resized = img
    return float(cv2.Laplacian(resized, cv2.CV_64F).var())


def tenengrad_score(img: np.ndarray) -> float:
    """Tenengrad (Sobel-based sharpness): mean squared gradient magnitude."""
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag_sq = gx * gx + gy * gy
    return float(grad_mag_sq.mean())


def edge_density(img: np.ndarray) -> float:
    """Percent of pixels detected as edges via Canny; low density can mean blur or blank."""
    v = float(np.median(img))
    lower = int(max(0, 0.66 * v))
    upper = int(min(255, 1.33 * v))
    edges = cv2.Canny(img, lower, upper)
    return float((edges > 0).mean() * 100.0)


def brightness(img: np.ndarray) -> float:
    """Mean pixel value; extremely dark or bright images are likely unusable."""
    return float(img.mean())


def contrast_score(img: np.ndarray) -> float:
    """Use standard deviation as a simple contrast measure."""
    return float(img.std())


def normalize_score(value: float, min_val: float, max_val: float) -> float:
    """Normalize a value to 0-1 range, clamping at boundaries."""
    if max_val <= min_val:
        return 0.0
    normalized = (value - min_val) / (max_val - min_val)
    return max(0.0, min(1.0, normalized))


def classify_quality(
    focus: float,
    focus_small: float,
    tenengrad: float,
    contrast: float,
    edge_pct: float,
    brightness_val: float,
    quality_threshold: float = 0.35,
) -> tuple[str, float]:
    """
    Weighted scoring system for image quality assessment.
    Returns (verdict, quality_score) where quality_score is 0-1.
    """
    # Normalize each metric to 0-1 scale based on observed ranges
    # More strict focus normalization to catch blurry images
    # More lenient lower bound to help lower-focus but acceptable images
    focus_score_norm = normalize_score(focus, 30.0, 1000.0)
    focus_small_score = normalize_score(focus_small, 50.0, 1500.0)
    tenengrad_score_norm = normalize_score(tenengrad, 1500.0, 20000.0)
    contrast_score_norm = normalize_score(contrast, 18.0, 50.0)
    edge_score = normalize_score(edge_pct, 0.2, 10.0)
    
    
    # Brightness: penalize only extreme values (too dark < 10 or too bright > 248)
    if brightness_val < 10.0:
        brightness_score = 0.0
    elif brightness_val > 248.0:
        brightness_score = 0.0
    elif brightness_val < 25.0:
        brightness_score = normalize_score(brightness_val, 10.0, 25.0)
    elif brightness_val > 245.0:
        brightness_score = normalize_score(brightness_val, 245.0, 248.0)
    else:
        brightness_score = 1.0
    
    # Weighted combination
    weights = {
        'focus': 0.22,
        'focus_small': 0.22,
        'tenengrad': 0.22,
        'contrast': 0.18,
        'edge': 0.12,
        'brightness': 0.04,
    }
    
    quality_score = (
        weights['focus'] * focus_score_norm +
        weights['focus_small'] * focus_small_score +
        weights['tenengrad'] * tenengrad_score_norm +
        weights['contrast'] * contrast_score_norm +
        weights['edge'] * edge_score +
        weights['brightness'] * brightness_score
    )
    
    # Additional penalty for moderate focus (500-600 range): apply to overall score
    # This catches images that are slightly blurry but might have good other metrics
    # Reduced penalty range and severity to avoid false positives on good quality images
    if 500.0 <= focus < 600.0:
        focus_penalty = (600.0 - focus) / 100.0  # Penalty factor 0-1 (for focus 500-600)
        focus_penalty = min(1.0, focus_penalty)  # Cap at 1.0
        # Reduce overall score by up to 40% for moderate-low focus (slightly blurry)
        quality_score = quality_score * (1.0 - focus_penalty * 0.40)
    
    # Threshold: score >= quality_threshold is GOOD
    verdict = "GOOD" if quality_score >= quality_threshold else "BAD"
    
    return verdict, quality_score


def check_blur(img: np.ndarray, quality_threshold: float = 0.35) -> bool:
    """
    Check if image is blurred.
    Returns True if blurred (BAD quality), False if not blurred (GOOD quality).
    Lower threshold (0.25) allows slightly blurry but readable images to pass.
    """
    img_gray = load_grayscale(img)
    
    focus_val = focus_score(img_gray)
    focus_small_val = focus_score_small(img_gray)
    tenengrad_val = tenengrad_score(img_gray)
    contrast_val = contrast_score(img_gray)
    edge_pct = edge_density(img_gray)
    bright_val = brightness(img_gray)
    
    verdict, quality_score = classify_quality(
        focus_val,
        focus_small_val,
        tenengrad_val,
        contrast_val,
        edge_pct,
        bright_val,
        quality_threshold,
    )
    
    # Debug output
    print(f"Blur Detection Metrics - Focus: {focus_val:.2f}, Focus_small: {focus_small_val:.2f}, Tenengrad: {tenengrad_val:.2f}, Contrast: {contrast_val:.2f}, Edge%: {edge_pct:.2f}, Brightness: {bright_val:.2f}")
    print(f"Quality Score: {quality_score:.4f}, Verdict: {verdict}, Threshold: {quality_threshold}")
    
    # Return True if BAD (blurred), False if GOOD (not blurred)
    return verdict == "BAD"


# ========== Main Function ==========

def main():
    # Path to the test image
    # image_path = Path(__file__).parent / "Check/RepTraffic_FileStore_Docls-SA_FN_MAIN/content/cont_1764563163708.jpg"
    image_path = Path(__file__).parent / "blurimg.png" #Test image path
    
    
    if not image_path.exists():
        print(f"Error: Image not found at {image_path}")
        return
    
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    # Initialize DocAligner model
    model = DocAligner(
        model_type=ModelType.heatmap,
        model_cfg='fastvit_sa24',
        backend=cb.Backend.cpu
    )
    
    # Check for tilt
    is_tilted_flag = check_tilt(img, model)
    
    # Check for blur
    is_blur_flag = check_blur(img)
    
    # Determine validate_doc_flag
    # False if tilted OR blurred (or both), True if neither
    validate_doc_flag = not (is_tilted_flag or is_blur_flag)
    
    # Output all flags
    print(f"is_tilted_flag: {is_tilted_flag}")
    print(f"is_blur_flag: {is_blur_flag}")
    print(f"validate_doc_flag: {validate_doc_flag}")


if __name__ == "__main__":
    main()
