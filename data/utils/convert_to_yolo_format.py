import os
import sys
import pandas as pd
import argparse


def get_parent_dir(n=1):
    """returns the n-th parent directory of the current working directory"""
    current_path = os.path.dirname(os.path.abspath(__file__))
    for _ in range(n):
        current_path = os.path.dirname(current_path)
    return current_path


sys.path.append(os.path.join(get_parent_dir(1), "utils"))
from utils.convert_format import convert_vott_csv_to_yolo

data_folder = os.path.join(get_parent_dir(), "data")
vott_folder = os.path.join(data_folder, "images", "vott-csv-export")
vott_csv = os.path.join(vott_folder, "Annotations-export.csv")
yolo_filename = os.path.join(vott_folder, "data_train.txt")

model_folder = os.path.join(data_folder, "weights")
classes_filename = os.path.join(model_folder, "data_classes.txt")

if __name__ == "__main__":
    # surpress any inhereted default values
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    """
    Command line options
    """
    parser.add_argument(
        "--vott_folder",
        type=str,
        default=vott_folder,
        help=
        "Absolute path to the exported files from the image tagging step with VoTT. Default is "
        + vott_folder,
    )

    parser.add_argument(
        "--VoTT_csv",
        type=str,
        default=vott_csv,
        help="Absolute path to the *.csv file exported from VoTT. Default is "
        + vott_csv,
    )
    parser.add_argument(
        "--YOLO_filename",
        type=str,
        default=yolo_filename,
        help=
        "Absolute path to the file where the annotations in YOLO format should be saved. Default is "
        + yolo_filename,
    )

    FLAGS = parser.parse_args()

    # Prepare the dataset for YOLO
    multi_df = pd.read_csv(FLAGS.VoTT_csv)
    labels = multi_df["label"].unique()
    labeldict = dict(zip(labels, range(len(labels))))
    multi_df.drop_duplicates(subset=None, keep="first", inplace=True)
    train_path = FLAGS.vott_folder
    convert_vott_csv_to_yolo(multi_df,
                             labeldict,
                             path=train_path,
                             target_name=FLAGS.yolo_filename)

    # Make classes file
    file = open(classes_filename, "w")

    # Sort Dict by Values
    SortedLabelDict = sorted(labeldict.items(), key=lambda x: x[1])
    for elem in SortedLabelDict:
        file.write(elem[0] + "\n")
    file.close()
