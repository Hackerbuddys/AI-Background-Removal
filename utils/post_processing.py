import numpy as np
import cv2
from PIL import Image

def refine_edges(pil_image, smooth_level=3):
    np_img = np.array(pil_image)

    if np_img.shape[2] < 4:
        return pil_image

    alpha = np_img[:, :, 3]

    # Detect fine edges from RGB (not only alpha)
    gray = cv2.cvtColor(np_img[:, :, :3], cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 30, 120)

    # Expand detected edges slightly
    edge_kernel = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, edge_kernel, iterations=1)

    # Merge edges into alpha mask
    alpha = cv2.bitwise_or(alpha, edges)

    # Light smoothing (avoid aggressive blur)
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)

    # Soft threshold (important)
    _, alpha = cv2.threshold(alpha, 2, 255, cv2.THRESH_BINARY)

    np_img[:, :, 3] = alpha

    return Image.fromarray(np_img)
