import cv2
import numpy as np
from skimage.filters import unsharp_mask
from skimage import exposure


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


# Enhance image contrast
def enhance_contrast(image):
    enhanced_img = exposure.adjust_log(image, 1)
    return enhanced_img


def hsv_equalizer(image, new_channel):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h,s,v =  cv2.split(hsv)
    merged_hsv = cv2.merge((h, s, new_channel))
    bgr_img = cv2.cvtColor(merged_hsv, cv2.COLOR_HSV2BGR)
    return bgr_img


def hsv_equalized_img(image, clahe_applied_perceived_channel):
    return hsv_equalizer(image, clahe_applied_perceived_channel)
