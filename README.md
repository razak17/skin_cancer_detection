## Skin cancer detection

These implementations are based on the following research papers:

1. Melanoma lesion detection and segmentation using deep region based convolutional neural network and fuzzy C-means clustering.

2. A comparative study of deep learning architectures on melanoma detection.


## 1. Melanoma lesion detection and segmentation using deep region based convolutional neural network and fuzzy C-means clustering.

The first step was to preprocess / refine images to remove hair and other artifacts from then input image.
We did this by running to morphological closing operations (dilation, then erosion) on the input image.