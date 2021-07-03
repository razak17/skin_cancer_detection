import cv2
import numpy as np
from skimage.filters import unsharp_mask


# Dilate and erode the input image to get rid of hair and other artifacts
def closing_operation(image):
    kernel = np.ones((5, 5), np.uint8)
    dil = cv2.dilate(image, kernel, iterations=1)
    erode = cv2.erode(dil, kernel, iterations=1)
    # closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return erode


# Apply the unsharped filter to remove blurring effect from image
# Subtract nn
def unsharp_mask(image, blurred_image):
    unsharped_image = unsharp_mask(blurred_image, radius=3, amount=2)
    sharped_image = image - unsharped_image
    return sharped_image
