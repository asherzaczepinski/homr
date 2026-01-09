"""
Image deskewing utilities for sheet music.
"""

import cv2
import numpy as np


def deskew_image(img):
    """
    Detect and correct skew in the image using Hough line detection.

    Args:
        img: Grayscale or color image (numpy array)

    Returns:
        Deskewed image
    """
    # Create binary image for line detection
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Apply threshold
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Detect lines using Hough transform
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

    if lines is None or len(lines) == 0:
        return img

    # Calculate angles of detected lines (focus on near-horizontal lines for staff lines)
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 != 0:  # Avoid division by zero
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            # Only consider near-horizontal lines (within 10 degrees of horizontal)
            if abs(angle) < 10:
                angles.append(angle)

    if not angles:
        return img

    # Use median angle to avoid outliers
    median_angle = np.median(angles)

    # Only deskew if angle is significant (> 0.1 degrees)
    if abs(median_angle) < 0.1:
        return img

    print(f"Deskewing image by {median_angle:.2f} degrees")

    # Rotate image to correct skew
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)

    # Calculate new image size to avoid clipping
    cos_angle = abs(rotation_matrix[0, 0])
    sin_angle = abs(rotation_matrix[0, 1])
    new_w = int(h * sin_angle + w * cos_angle)
    new_h = int(h * cos_angle + w * sin_angle)

    # Adjust rotation matrix for new size
    rotation_matrix[0, 2] += (new_w - w) / 2
    rotation_matrix[1, 2] += (new_h - h) / 2

    # Apply rotation with white background
    deskewed = cv2.warpAffine(img, rotation_matrix, (new_w, new_h),
                               borderMode=cv2.BORDER_CONSTANT, borderValue=255)

    return deskewed
