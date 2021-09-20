import os
import numpy as np
import shutil
import pandas as pd
"""
Loading the dataset for skin cancer detection.
Dataset link: https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000
Data description: https://arxiv.org/ftp/arxiv/papers/1803/1803.10417.pdf

The 7 classes of skin cancer lesions included in this dataset are:
Melanocytic nevi (nv) - benign
Melanoma (mel) - malignant
Benign keratosis-like lesions (bkl) - benign
Basal cell carcinoma (bcc) - benign
Actinic keratoses (akiec) - benign
Vascular lesions (vas) - benign
Dermatofibroma (df) - benign
"""


def split_into_classes(label, skin_df, label_images, images_dir, dest_dir):
    sample_images = os.listdir(images_dir)
    new_sample_images = []

    os.mkdir(dest_dir)
    for n in sample_images:
        new_sample_images.append(n.split('.')[0])

    for i in label:
        if not os.path.isdir(dest_dir + str(i) + "/"):
            os.mkdir(dest_dir + str(i) + "/")
        sample = skin_df[skin_df['dx'] == i]['image_id']
        label_images.extend(sample)
        for id in label_images:
            if id in new_sample_images:
                shutil.copyfile((images_dir + id + ".jpg"),
                                (dest_dir + i + "/" + id + ".jpg"))
            label_images = []


def split_into_train_test_val(input_folder, train, test, val):
    src = input_folder  # Folder to copy images from

    allFileNames = os.listdir(src)
    np.random.shuffle(allFileNames)

    # 70:15:15
    train_FileNames, val_FileNames, test_FileNames = np.split(
        np.array(allFileNames),
        [int(len(allFileNames) * 0.7),
         int(len(allFileNames) * 0.85)])

    train_FileNames = [src + '/' + name for name in train_FileNames.tolist()]
    val_FileNames = [src + '/' + name for name in val_FileNames.tolist()]
    test_FileNames = [src + '/' + name for name in test_FileNames.tolist()]

    print('Total images: ', len(allFileNames))
    print('Training: ', len(train_FileNames))
    print('Validation: ', len(val_FileNames))
    print('Testing: ', len(test_FileNames))

    # Copy-pasting images
    for name in train_FileNames:
        shutil.copy(name, "data/images/train")

    for name in val_FileNames:
        shutil.copy(name, "data/images/val")

    for name in test_FileNames:
        shutil.copy(name, "data/images/test")
