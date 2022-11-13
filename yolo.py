# System imports
import numpy as np
import torchvision as tv
import torch
from scipy import integrate


# Function for running the entire YOLO detection process
def run_yolo_detector(IMAGES, BOXES):
    # Load provided model from provided files
    model = torch.hub.load('./yolov5', "custom",
                           path="./support_files/yolo5s.pt", source="local")

    # Run model on RGB images
    results = model(IMAGES)

    # Convert results to pandas format
    results = results.pandas().xyxy

    # Create a tensor representing the bounding box
    yolo_detected_boxes = [
        torch.tensor([box["xmin"], box["ymin"], box["xmax"], box["ymax"]]).T
        for box in results
    ]

    # Compute IoU values based on the saved ground truth bounding boxes
    yolo_ious = [tv.ops.boxes.box_iou(BOXES[index], box)
                 for index, box in enumerate(yolo_detected_boxes)]

    # Get the number of images where more than one ear was detected (assuming there is only one ground truth)
    #   and save them as False Positive hits to keep in line with VJ detector
    fp_ = len([yolo_iou for yolo_iou in yolo_ious if yolo_iou.nelement() > 1])

    # Get max IoU values from all detections that happened
    yolo_ious = [yolo_iou.amax()
                 for yolo_iou in yolo_ious if yolo_iou.nelement() > 0]

    # Save difference of images where no ear was detected as False Negative
    fn = len(BOXES) - len(yolo_ious)

    # Define empty array for storing (precision, recall) pairs
    precision_recalls_yolo = []

    # Transform array to numpy for easier comparisons
    iou = np.array(yolo_ious, dtype="object")

    # Go over [0, 1] interval with step 0.01 for threshold values
    for tau in np.arange(0, 1.01, 0.01):
        # Count True Positive hits, those being IoU over (or equal to) the threshold
        tp = np.sum(iou >= tau)
        # Count False Positive hits, those being IoU under threshold
        fp = np.sum(iou < tau)
        precison = tp / (tp + fp + fp_)  # Calculate precision
        recall = tp / (tp + fn)  # Calculate recall
        # Store value pairs as numpy array for easier processing
        precision_recalls_yolo.append(
            np.array([tau, np.mean(iou), precison, recall]))

    # Convert to numpy array for easier computations
    precision_recalls_yolo = np.array(precision_recalls_yolo)

    # Compute mAP or AP as a integral approximation when using Pascal Voc coordinate format
    mean_ap_yolo = np.abs(integrate.simpson(
        precision_recalls_yolo[:, 3], precision_recalls_yolo[:, 2]))

    print(
        f'The (mean) average precision (AP) of the YOLO5 detector is: {mean_ap_yolo}')
