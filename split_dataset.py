import os
import numpy as np
import shutil


def split_into_train_test_val(input_folder):
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


input_folder = 'data/HAM10000/all_images/'

split_into_train_test_val(input_folder)
