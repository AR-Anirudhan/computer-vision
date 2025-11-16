import cv2
import numpy as np

def overlay_transparent(background, overlay, x, y):
    """
    Overlay transparent PNG on background image
    
    Args:
        background: Background image (BGR)
        overlay: Overlay image with alpha channel (BGRA)
        x, y: Top-left position for overlay
    
    Returns:
        Modified background image
    """
    bg_h, bg_w = background.shape[:2]
    
    # Ensure overlay has alpha channel
    if overlay.shape[2] == 3:
        # Convert BGR to BGRA
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2BGRA)
    
    overlay_h, overlay_w = overlay.shape[:2]
    
    # Handle out of bounds completely
    if x >= bg_w or y >= bg_h or x + overlay_w <= 0 or y + overlay_h <= 0:
        return background
    
    # Calculate crop boundaries
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(bg_w, x + overlay_w)
    y2 = min(bg_h, y + overlay_h)
    
    # Calculate overlay crop
    overlay_x1 = max(0, -x)
    overlay_y1 = max(0, -y)
    overlay_x2 = overlay_x1 + (x2 - x1)
    overlay_y2 = overlay_y1 + (y2 - y1)
    
    # Extract regions
    overlay_crop = overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2]
    background_crop = background[y1:y2, x1:x2]
    
    # Check if regions are valid
    if overlay_crop.shape[0] == 0 or overlay_crop.shape[1] == 0:
        return background
    
    # Separate alpha channel and normalize
    overlay_rgb = overlay_crop[:, :, :3]
    alpha = overlay_crop[:, :, 3:] / 255.0
    
    # Blend images
    blended = (alpha * overlay_rgb + (1 - alpha) * background_crop).astype(np.uint8)
    
    # Copy back to background
    background[y1:y2, x1:x2] = blended
    
    return background


def resize_with_aspect_ratio(image, width=None, height=None):
    """
    Resize image maintaining aspect ratio
    
    Args:
        image: Input image
        width: Target width (optional)
        height: Target height (optional)
    
    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    
    if width is None and height is None:
        return image
    
    if width is None:
        ratio = height / h
        width = int(w * ratio)
    else:
        ratio = width / w
        height = int(h * ratio)
    
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return np.linalg.norm(np.array(point1) - np.array(point2))


def calculate_angle(point1, point2):
    """Calculate angle between two points in degrees"""
    return np.degrees(np.arctan2(point2[1] - point1[1], 
                                 point2[0] - point1[0]))
