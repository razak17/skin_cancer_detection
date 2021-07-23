# Skin cancer detection

These implementations are based on the following research papers:

+ [Melanoma lesion detection and segmentation using deep region based convolutional neural network and fuzzy C-means clustering](https://www.sciencedirect.com/science/article/pii/S1386505618307470)

+ [A comparative study of deep learning architectures on melanoma detection](https://www.sciencedirect.com/science/article/pii/S0040816619300904)

### Repo structure
+ [`dataset`](/data/): Scripts and instructions on how to handle images

### Requirements
* [Python 3.9+](https://www.python.org)
* [Jupyter notebook](https://jupyter.readthedocs.io/en/latest/install.html)
* [Anaconda or Miniconda](https://anaconda.org/) (optional but recommended)

### Install required libraries
`pip3 install -r requirements.txt`

### 1. Melanoma lesion detection and segmentation using deep region based convolutional neural network and fuzzy C-means clustering.
This implementation of skin cancer detection has three main steps namely:
1. Skin Enhancement Phase.
2. Lesion localization Phase.
3. Lesion segmentation Phase.

### Skin Enhancement Phase
The first step was to preprocess / refine images to remove hair and other artifacts from then input image.
We did this by running two morphological closing operations (dilation, then erosion) on the input image.
