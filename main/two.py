import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import io
from utils.skin_refinement import closing_operation, enhance_contrast, hsv_equalized_img


def get_perceive_brightness(image):
    float_img = np.float64(image)  # uint8 will make overflow
    b, g, r = cv2.split(float_img)
    float_brightness = np.sqrt((0.241 * (r**2)) + (0.691 * (g**2)) + (0.068 *
                                                                      (b**2)))
    brightness_channel = np.uint8(np.absolute(float_brightness))
    return brightness_channel


clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))


def refine(train_data_keras, dest_dir):
    x, _ = next(train_data_keras)
    # View each image
    for i in range(0, 5):
        # image = x[i]
        image = x[i].astype(int)
        float_img = np.uint8(image)  # uint8 will make overflow
        clahe_applied_perceived_channel = clahe.apply(
            get_perceive_brightness(float_img))
        # morph_image = closing_operation(float_img)
        contrast_enhanced_image = enhance_contrast(float_img)
        # illuminated_image = hsv_equalized_img(contrast_enhanced_image,
        #                                       clahe_applied_perceived_channel)
        new_file_path = '%s/refined_image_%s.jpg' % (dest_dir, i)
        plt.imshow(contrast_enhanced_image)
        plt.show()

        # write image to the disk
        # io.imsave(new_file_path, contrast_enhanced_image)
