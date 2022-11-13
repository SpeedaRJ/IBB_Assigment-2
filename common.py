# System imports
import os
import numpy as np
import cv2
import torch

# Custom imports
from yolo import run_yolo_detector


# Function for converting YOLO normalized format to Pascal Voc standard Pixal values
def yolo_to_pascal_voc(x_center, y_center, w, h,  image_w, image_h):
    w = w * image_w
    h = h * image_h
    x1 = ((2 * x_center * image_w) - w)/2
    y1 = ((2 * y_center * image_h) - h)/2
    return torch.tensor([[x1, y1, x1 + w, y1 + h]], dtype=torch.float64)


# Function for converting Coco format (returned by .detectMultiScale(...)) to Pascal Voc
def coco_to_pascal_voc(x1, y1, w, h):
    return torch.tensor([[x1, y1, x1 + w, y1 + h]], dtype=torch.float64)


# Loading up *.xml cascade classifiers using cv2
LEFT_EAR_CASCADE = cv2.CascadeClassifier(
    'support_files/haarcascade_mcs_leftear.xml')
RIGHT_EAR_CASCADE = cv2.CascadeClassifier(
    'support_files/haarcascade_mcs_rightear.xml')

# Some global variables
PATH = "./support_files/ear_data/test"
IMAGES = []
BOXES = []

bad_vj_detections = []
good_vj_detections = []

# Going through the array of pictures
for file in os.listdir(PATH):
    file_path = os.path.join(PATH, file)  # Creating file path
    # Check if file is valid and is image
    if os.path.isfile(file_path) and "png" in file_path:
        # Load the image as a tensor
        img = cv2.imread(file_path)
        # Add it to the array
        IMAGES.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # Else check if file is valid and is text file
    elif os.path.isfile(file_path) and "txt" in file_path:
        with open(file_path, "r") as f:
            content = np.array(f.readlines()[0].split(" ")).astype(
                np.float64)  # Splitting the file content into values
            # Add Pascal Voc values to array of ground truths
            BOXES.append(yolo_to_pascal_voc(
                content[1], content[2], content[3], content[4], IMAGES[-1].shape[1], IMAGES[-1].shape[0]))


run_yolo_detector(IMAGES, BOXES)
