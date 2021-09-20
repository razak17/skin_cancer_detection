import os
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from utils import split_dataset
from main import one, two

if not os.path.isdir("data/HAM10000"):
    os.mkdir("data/HAM10000/")
    os.mkdir("data/HAM10000/all_images/")
    os.mkdir("data/HAM10000/reorganized/")
    os.mkdir("data/HAM10000/metadata/")

# Dump all images into a folder and specify the path
images_dir = "data/HAM10000/all_images/"

# Path to destination directory where for want subfolders
dest_dir = "data/HAM10000/reorganized/"

# Read the csv file containing image names and corresponding labels
skin_df = pd.read_csv('data/HAM10000/metadata/HAM10000_metadata.csv')
print(skin_df['dx'].value_counts())

# Extract labels into a list
label = skin_df['dx'].unique().tolist()  # Extract labels into a list

# Initialize array to store image classes
label_images = []


def split():
    if not os.path.isdir(dest_dir):
        # split dataset into various classes (run this once)
        split_dataset.split_into_classes(label, skin_df, label_images,
                                         images_dir, dest_dir)
    else:
        print("dest_dir already exists.\nRemove it then run the split again.")

    # split dataset into train, test and validation dirs (run this once)
    train = 'data/dataset/train'
    test = 'data/dataset/test'
    val = 'data/dataset/val'
    for i in label:
        if not os.path.isdir('data/dataset/' + i):
            split_dataset.split_into_train_test_val('data/dataset/all_images')


# specify training images directory
train_dir = "data/images/train"

# Path for refined images
refined_yolo = 'data/images/refined'
refined_resnet = 'data/images/refined_resnet'

if __name__ == '__main__':
    # split dataset (run once)
    split()
