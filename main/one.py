import numpy as np
from matplotlib import pyplot as plt
from preprocessing.skin_refinement import closing_operation, sharpen_image


def refine(train_data_keras):
    """
    Refine images to remove artifacts and correct illumination
    """
    x, _ = next(train_data_keras)
    # View each image
    for i in range (0, 7):
        image = x[i]
        float_img = np.uint8(image)  # uint8 will make overflow
        morph_image = closing_operation(float_img)
        sharped_image = sharpen_image(morph_image)
        plt.imshow(sharped_image)
        plt.show()
    return sharped_image
