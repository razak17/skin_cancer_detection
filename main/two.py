import cv2
import numpy as np
from matplotlib import pyplot as plt
from preprocessing.skin_refinement import closing_operation, enhance_contrast, hsv_equalized_img


def get_perceive_brightness(image):
    float_img = np.float64(image)  # uint8 will make overflow
    b, g, r = cv2.split(float_img)
    float_brightness = np.sqrt(
        (0.241 * (r ** 2)) + (0.691 * (g ** 2)) + (0.068 * (b ** 2)))
    brightness_channel = np.uint8(np.absolute(float_brightness))
    return brightness_channel


clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))


def refine(train_data_keras):
    x, _ = next(train_data_keras)
    # View each image
    for i in range (0, 7):
        image = x[i]
        float_img = np.uint8(image)  # uint8 will make overflow
        clahe_applied_perceived_channel = clahe.apply(get_perceive_brightness(float_img))
        morph_image = closing_operation(float_img)
        contrast_enhanced_image = enhance_contrast(morph_image)
        illuminated_image = hsv_equalized_img(contrast_enhanced_image, clahe_applied_perceived_channel)
        plt.imshow(contrast_enhanced_image)
        plt.show()
    return illuminated_image

