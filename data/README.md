# Dataset and Image Annotation

### Dataset
We trained our model using the [`HAM10000 dataset`](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000). For training YOLOv5, we require both RGB dermoscopic images along with corresponding lesion region as a training pairs. Therefore we need to generate ground truth masks for the images. The images along with their ground truths are then passed to YOLOv5 for model learning in the lesion localization  phase.

Copy your images to [`data/HAM10000/all_images/`](/data/HAM10000/all_images/).
Copy metadata (csv files from the archive) to [`data/HAM10000/metadata/`](/data/HAM10000/metadata/).
Run from the project root to split the dataset into training, test and validation images.
  ```
  python split_dataset.py
  ```

The split images would be stored at [`/data/images/`](/data/images) in their respective folders.


### Image Annotation
We need to annotate the training images for model learning. We use the 70:15:15 ratio for training, testing and validation images respectively. We use Microsoft's Visual Object Tagging Tool [`vOtt`](https://github.com/Microsoft/VoTT/releases) to label our images.

Copy the labels for train and validation data to [`data/labels/train/`](/data/labels/train/) and [`data/labels/val`](/data/labels/val) respectively.

### Convert to YOLO Format
As a final step, convert the VoTT csv format to the YOLOv5 format. To do so, run the conversion script from within the [`data/`](/data/) folder:
```
python convert_to_yolo_format.py
```
The script generates two output files: [`data_train.txt`](/data/vott-csv-export/data_train.txt) located in the [`/data/vott-csv-export`](/data/vott-csv-export/) folder and [`data_classes.txt`](/data/weights/data_classes.txt) located in the [`data/weights/`](/data/weights/) folder. To list available command line options run `python convert_to_yolo_format.py -h`.

### That's all for annotation!
Next, go to [`/training`](/training) to train your YOLOv5 detector.
