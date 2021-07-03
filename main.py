"""
Loading the dataset for skin cancer detection.
Dataset link: https://challenge2020.isic-archive.com/
"""

import os
import shutil
import pandas as pd
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from preprocessing import skin_refinement


# Dump all images into a folder and specify the path:
data_dir = os.getcwd() + "/data/ISIC-images/2020_challenge/"

# Path to destination directory where for want subfolders
dest_dir = os.getcwd() + "/data/ISIC-images/reorganized/"

# Read the csv file containing image names and corresponding labels
skin_df = pd.read_csv('data/ISIC-images/metadata.csv')
print(skin_df['meta.clinical.benign_malignant'].value_counts())

# Extract labels into a list
label=skin_df['meta.clinical.benign_malignant'].unique().tolist()
benign_malignant = []
# print(label)

# Copy images to new dest_dir
def copy_to_dest(label, data_dir, dest_dir):
    for i in label:
        os.mkdir(dest_dir + str(i) + "/")
        sample = skin_df[skin_df['meta.clinical.benign_malignant'] == i]['name']
        benign_malignant.extend(sample)
        print(benign_malignant)
        for id in benign_malignant:
            shutil.copyfile((data_dir + "/" + id +".jpg"), (dest_dir + i + "/" + id + ".jpg"))
        benign_malignant = []


# copy_to_dest()

# Flow_from_directory Method with keras
# Useful when the images are sorted and placed in there respective class/label folders
# Identifies classes automatically from the folder name.

# Define datagen. Here we can define any transformations we want to apply to images
datagen = ImageDataGenerator()

# define training directory that contains subfolders
train_dir = os.getcwd() + "/data/ISIC-images/reorganized/"

# Use flow_from_directory
train_data_keras = datagen.flow_from_directory(directory=train_dir,
                                         class_mode='categorical',
                                         batch_size=16)  #16 images at a time
                                         # target_size=(32,32))  #Resize images

# We can check images for a single batch.
x, y = next(train_data_keras)

# View each image
for i in range (0, 7):
    image = x[i]
    float_img = np.uint8(image)  # uint8 will make overflow
    morph_image = skin_refinement.closing_operation(float_img)
    sharped_image = skin_refinement.unsharp_mask(image, morph_image)
    plt.imshow(sharped_image)
    plt.show()

#Now you can train via model.fit_generator
