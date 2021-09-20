import numpy as np
from matplotlib import pyplot as plt
from utils.skin_refinement import closing_operation, sharpen_image
from skimage import io


def refine(train_data_keras, dest_dir):
    """
    Refine images to remove artifacts and correct illumination
    """
    x, _ = next(train_data_keras)

    for i in range(0, 5):
        if i > 2:
            break
        image = x[i]
        float_img = np.uint8(image)  # uint8 will make overflow
        morph_image = closing_operation(float_img)
        sharped_image = sharpen_image(morph_image)
        new_file_path = '%s/refined_image_%s.jpg' % (dest_dir, i)
        plt.imshow(sharped_image)
        plt.show()

        # write image to the disk
        io.imsave(new_file_path, sharped_image)
