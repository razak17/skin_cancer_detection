import cv2
import numpy as np
from skimage.filters import unsharp_mask


# Dilate and erode the input image to get rid of hair and other artifacts
def closing_operation(image):
    kernel = np.ones((7, 7), np.uint8)
    dil = cv2.dilate(image, kernel, iterations=1)
    erode = cv2.erode(dil, kernel, iterations=1)
    # closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return erode


# Apply the unsharped filter to remove blurring effect from image
# Subtract image from blurred image to get sharped image
def sharpen_image(image):
    sharped_image = unsharp_mask(image, radius=0, amount=2)
    return sharped_image
