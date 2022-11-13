# Custom imports
from common import *


# System imports
from scipy import integrate
import torchvision as tv


def get_detection_scores_for_s_and_n(s=1.1, n=3):
    iou_values = []  # Initialize empty array for iou values
    fn = 0  # Value tracker for False Negative hits
    fp_ = 0  # Value tracker for False Positive hits during initial detection
    # Going through the loaded images
    for index, image in enumerate(IMAGES):
        img = np.array(image)  # Change image from tensor to numpy
        # Convert image to greyscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Running right ear detection
        right_ears = RIGHT_EAR_CASCADE.detectMultiScale(gray,
                                                        scaleFactor=s,
                                                        minNeighbors=n)
        # Running left ear detection
        left_ears = LEFT_EAR_CASCADE.detectMultiScale(gray,
                                                      scaleFactor=s,
                                                      minNeighbors=n)

        # If both don't return a result we have a False Negative hit
        if not len(left_ears) and not len(right_ears):
            fn = fn + 1  # Track the hit
        else:
            all_hits = [*right_ears, *left_ears]
            # If not first loop through left ear hits
            ious_for_hits = [
                tv.ops.boxes.box_iou(
                    BOXES[index], coco_to_pascal_voc(*ear))[0].numpy()
                for ear in all_hits
            ]
            selected_iou = np.amax(ious_for_hits)
            iou_values.append(selected_iou)
            fp_ = fp_ + len(ious_for_hits) - 1
            # This part was used for finding good and bad detections
            '''
            if selected_iou < 0.1:
                bad_vj_detections.append(
                    np.array([selected_iou, BOXES[index], s, n,
                              coco_to_pascal_voc(*all_hits[ious_for_hits.index(selected_iou)])],
                             dtype="object"))
            elif selected_iou >= 0.9:
                good_vj_detections.append(
                    np.array([selected_iou, BOXES[index], s, n,
                              coco_to_pascal_voc(*all_hits[ious_for_hits.index(selected_iou)])],
                             dtype="object"))
            '''

    # Define empty array for storing (precision, recall) pairs
    precision_recall = []
    # Transform array to numpy for easier comparisons
    iou = np.array(iou_values)
    # Go over [0, 1] interval with step 0.01 for threshold values
    for tau in np.arange(0, 1.01, 0.01):
        # Count True Positive hits, those being IoU over (or equal to) the treshold
        tp = np.sum(iou >= tau)
        # Count False Positive hits, those being IoU under threshold
        fp = np.sum(iou < tau)
        precison = tp / (tp + fp + fp_)  # Calculate precision
        recall = tp / (tp + fn)  # Calculate recall
        # Store value pairs as numpy array for easier processing
        precision_recall.append(
            np.array([tau, np.mean(iou), precison, recall]))

    # Convert results array to numpy
    return np.array(precision_recall)


# Go through all combinations of chosen parameters and call precision recall calculations
precision_recalls = []
for s in [1.01, 1.1, 1.5, 1.7, 2]:
    for n in [1, 2, 3, 4, 5]:
        precision_recalls.append(get_detection_scores_for_s_and_n(s, n))

# Calculate AP for above calculations using Simpson integration approximation method
mean_aps = [np.abs(integrate.simpson(pr_data[:, 3], pr_data[:, 2]))
            for pr_data in precision_recalls]

"""
# Write results to file
with open("VJ_results.txt", "w") as f:
    f.write(json.dumps(mean_aps))
"""
