import os
import csv
import cv2
from ultralytics import YOLO
import yaml
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score

# Load YOLO model
def load_yolo_model(weights_path="yolo11x.pt"):
    return YOLO(weights_path)

# Perform object detection
def detect_objects(image, model, conf_threshold=0.3, nms_threshold=0.9):
    results = model(image, imgsz=1280)
    boxes, confidences, class_ids = [], [], []
    
    for result in results:
        for box in result.boxes:
            xywh = box.xywh[0].cpu().numpy()
            confidence = box.conf[0].item()
            class_id = int(box.cls[0].item())

            if confidence > conf_threshold:
                x, y, w, h = xywh
                x, y = int(x - w / 2), int(y - h / 2)
                boxes.append([x, y, int(w), int(h)])
                confidences.append(confidence)
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    if len(indices) > 0:
        indices = indices.flatten()
        boxes = [boxes[i] for i in indices]
        confidences = [confidences[i] for i in indices]
        class_ids = [class_ids[i] for i in indices]

    return boxes, confidences, class_ids

# Load class names from YAML
def load_classes_from_yaml(yaml_path):
    with open(yaml_path, "r", encoding='utf-8') as f:
        classes = yaml.safe_load(f)['names']
    return [class_name for class_name in classes.values()]

# Load ground truth bounding boxes and class names from CSV
def load_ground_truth(csv_path, classes):
    gt_boxes = []
    with open(csv_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            class_name = row[3]  # Assuming class name is in the 4th column
            if class_name not in classes:
                print(f"Class '{class_name}' not found in the class list.")
                continue
            class_id = classes.index(class_name)
            xmin, ymin, xmax, ymax = map(int, row[4:8])  # Assuming bbox is in columns 4 to 7
            gt_boxes.append([xmin, ymin, xmax - xmin, ymax - ymin, class_name])  # Include class_name
            
    return gt_boxes

def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Calculate intersection area
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    # Calculate intersection area
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    # Calculate union area
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    # Calculate IoU
    iou = inter_area / union_area if union_area > 0 else 0
    return iou

# Evaluate detections against ground truth
def evaluate_detections(detected_boxes, ground_truth_boxes, iou_threshold=0.5):
    tp, fp, fn, tn = 0, 0, 0, 0

    # Filter valid detected boxes
    detected_ids = []
    valid_detected_boxes = []
    for box in detected_boxes:
        if len(box) == 5:  # Ensure the box has (xmin, ymin, xmax, ymax, class_id)
            valid_detected_boxes.append((box[0], box[1], box[2], box[3]))
            detected_ids.append(box[4])
        else:
            print(f"Warning: Detected box does not have expected structure: {box}")

    gt_boxes = [(box[0], box[1], box[2], box[3]) for box in ground_truth_boxes]
    gt_ids = [box[4] for box in ground_truth_boxes]

<<<<<<< HEAD
    # Check for True Positives (TP), False Positives (FP), and False Negatives (FN)
    matched_gt = set()
    for idx, det_box in enumerate(valid_detected_boxes):
        for gt_idx, gt_box in enumerate(gt_boxes):
            if idx < len(detected_ids) and gt_idx < len(gt_ids):
                if gt_ids[gt_idx] == detected_ids[idx]:  # Class should match
                    iou = calculate_iou(det_box, gt_box)
                    if iou >= iou_threshold:
                        tp += 1
                        matched_gt.add(gt_idx)
                        break
        else:
            fp += 1  # If no match found, it's a false positive

    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn, tn
=======
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        
        
    return thresholds, tpr_list, fpr_list
>>>>>>> 2bc15d4 (added coco.yaml)


# Evaluate TPR, FNR at different confidence thresholds
def evaluate_at_confidence_thresholds(detections, confidences, class_ids, ground_truths, iou_threshold=0.5, conf_thresholds=np.arange(0.1, 1.0, 0.1)):
    tpr_values = []
    fpr_values = []
    
    for conf_threshold in conf_thresholds:
        detections_filtered = [d for i, d in enumerate(detections) if confidences[i] >= conf_threshold]
        class_ids_filtered = [class_ids[i] for i, d in enumerate(detections) if confidences[i] >= conf_threshold]

        tp, fp, fn = evaluate_detections(detections_filtered, ground_truths)
        
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        tpr_values.append(tpr)
        fpr_values.append(fpr)
    
    return conf_thresholds, tpr_values, fpr_values

# Compare TPR and FPR between synthetic and real datasets
def compare_datasets(real_detections, real_confidences, real_class_ids, real_ground_truths,
                     synthetic_detections, synthetic_confidences, synthetic_class_ids, synthetic_ground_truths,
                     classes, iou_threshold=0.5):
    conf_thresholds = np.arange(0.1, 1.0, 0.1)

    # Evaluate for real data
    real_conf_thresholds, real_tpr, real_fpr = evaluate_at_confidence_thresholds(real_detections, real_confidences, real_class_ids, real_ground_truths, iou_threshold)

    # Evaluate for synthetic data
    synthetic_conf_thresholds, synthetic_tpr, synthetic_fpr = evaluate_at_confidence_thresholds(synthetic_detections, synthetic_confidences, synthetic_class_ids, synthetic_ground_truths, iou_threshold)
    
    return real_conf_thresholds, real_tpr, real_fpr, synthetic_conf_thresholds, synthetic_tpr, synthetic_fpr

# Plot comparison between real and synthetic data
def plot_comparison(real_conf_thresholds, real_tpr, real_fpr, synthetic_conf_thresholds, synthetic_tpr, synthetic_fpr):
    plt.plot(real_conf_thresholds, real_tpr, label="Real TPR", color="g")
    plt.plot(real_conf_thresholds, real_fpr, label="Real FPR", color="r")
    plt.plot(synthetic_conf_thresholds, synthetic_tpr, label="Synthetic TPR", color="b")
    plt.plot(synthetic_conf_thresholds, synthetic_fpr, label="Synthetic FPR", color="orange")

    plt.xlabel("Confidence Threshold")
    plt.ylabel("Rate")
    plt.title("TPR and FPR vs Confidence Threshold Comparison (Real vs Synthetic Data)")
    plt.legend()
    plt.grid(True)
    plt.show()

# Main function to execute detection, saving results, and evaluation
def main():
    yolo_model = load_yolo_model()  # Load YOLO model
    classes = load_classes_from_yaml("coco.yaml")  # Load class names from YAML

    # Paths for real and synthetic images and ground truth CSVs
    real_image_path = r"Images\real_frames\frame_7.png"
    synthetic_image_path = r"Images\sd_frames\15.png"
    real_ground_truth_csv = r"Annotations\annotated_real\frame7.csv"
    synthetic_ground_truth_csv = r"Annotations\annotated_syn\frame15.csv"
    
    # Load ground truth for real and synthetic datasets
    real_ground_truths = load_ground_truth(real_ground_truth_csv, classes)
    synthetic_ground_truths = load_ground_truth(synthetic_ground_truth_csv, classes)

    # Perform object detection on real and synthetic images
    real_detections, real_confidences, real_class_ids = detect_objects(real_image_path, yolo_model)
    synthetic_detections, synthetic_confidences, synthetic_class_ids = detect_objects(synthetic_image_path, yolo_model)

    # Compare performance between real and synthetic datasets
    real_conf_thresholds, real_tpr, real_fpr, synthetic_conf_thresholds, synthetic_tpr, synthetic_fpr = compare_datasets(
        real_detections, real_confidences, real_class_ids, real_ground_truths,
        synthetic_detections, synthetic_confidences, synthetic_class_ids, synthetic_ground_truths,
        classes)

    # Plot the comparison
    plot_comparison(real_conf_thresholds, real_tpr, real_fpr, synthetic_conf_thresholds, synthetic_tpr, synthetic_fpr)

# Run main function
if __name__ == "__main__":
    main()
