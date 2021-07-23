"""
Loading the dataset for skin cancer detection.
Dataset link: https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000
Data description: https://arxiv.org/ftp/arxiv/papers/1803/1803.10417.pdf

The 7 classes of skin cancer lesions included in this dataset are:
Melanocytic nevi (nv)
Melanoma (mel)
Benign keratosis-like lesions (bkl)
Basal cell carcinoma (bcc)
Actinic keratoses (akiec)
Vascular lesions (vas)
Dermatofibroma (df)
"""
import os
import shutil
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
# from sklearn.model_selection import train_test_split

from main import one, two

# Dump all images into a folder and specify the path:
data_dir = os.getcwd() + "/data/HAM10000/all_images/"

# Path to destination directory where for want subfolders
dest_dir = os.getcwd() + "/data/HAM10000/reorganized/"

# Read the csv file containing image names and corresponding labels
skin_df = pd.read_csv('data/HAM10000/metadata/HAM10000_metadata.csv')
# print(skin_df['dx'].value_counts())

# Extract labels into a list
label = skin_df['dx'].unique().tolist()  #Extract labels into a list

# Initialize array to store image classes
label_images = []


def copy_to_dest(label, label_images, data_dir, dest_dir):
    for i in label:
        os.mkdir(dest_dir + str(i) + "/")
        sample = skin_df[skin_df['dx'] == i]['image_id']
        label_images.extend(sample)
        for id in label_images:
            shutil.copyfile((data_dir + "/" + id + ".jpg"),
                            (dest_dir + i + "/" + id + ".jpg"))
        label_images = []


# Flow_from_directory Method with keras
# Useful when the images are sorted and placed in there respective class/label folders
# Identifies classes automatically from the folder name.
def refine(train_dir):
    # Define datagen. Here we can define any transformations we want to apply to images
    datagen = ImageDataGenerator()

    # Use flow_from_directory
    train_data_keras = datagen.flow_from_directory(
        directory=train_dir,
        class_mode='categorical',
        batch_size=16,  #16 images at a time
        target_size=(32, 32))  # images

    one.refine(train_data_keras)
    # two.refine(train_data_keras)


train_dir = os.getcwd() + "/data/images/train/"

refine(train_dir)
