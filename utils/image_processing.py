from PIL import Image
import numpy as np
import cv2


def remove_white_background(img, threshold=240):
    """
    Removes white or near-white background only.
    Perfect for logos and text images.
    """

    img = img.convert("RGBA")
    np_img = np.array(img)

    # Convert to grayscale
    gray = cv2.cvtColor(np_img[:, :, :3], cv2.COLOR_RGB2GRAY)

    # Detect white pixels
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # Invert mask
    mask = cv2.bitwise_not(mask)

    # Apply mask as alpha channel
    np_img[:, :, 3] = mask

    return Image.fromarray(np_img)
