import os
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import csv
import yaml
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve

# Paths to your images, model weights, and COCO YAML file
real_image_path = r"Images\fog_real\RD_frame_1008.png"
synthetic_image_path = r"Images\fog_syn\SD_Frame_1008.png"
weights_path = "yolo11x.pt"  # Specify the correct path to your YOLO model weights
output_folder = "detection_results"  # Folder to save the results
yaml_path = "coco.yaml"  # Path to the COCO YAML file

# Load the YOLO model
def load_yolo_model(weights_path):
    try:
        model = YOLO(weights_path)
        return model
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return None

# Load COCO class names from YAML
def load_classes_from_yaml(yaml_path):
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data['names']  # List of class names
    except Exception as e:
        print(f"Error loading class names from YAML: {e}")
        return []

# Threshold values
thresholds = [round(0.3 + i * 0.1, 2) for i in range(8)]  # 0.3 to 1.0 in steps of 0.1

# Function to perform detection
def run_detection(image_path, model, thresholds):
    results_dict = {}
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        for threshold in thresholds:
            results = model(image_rgb, conf=threshold)  # Perform detection with the threshold
            annotated_image = results[0].plot()  # Annotated image with bounding boxes
            results_dict[threshold] = {"results": results[0], "image": annotated_image}
        
        return results_dict
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return {}

# Save detections into CSV files organized by threshold with confidence values as floats
def save_detections_to_csv_by_threshold(results, ground_truth_csv, base_output_folder, image_name, class_names):
    try:
        os.makedirs(base_output_folder, exist_ok=True)
        
        for threshold, data in results.items():
            threshold_folder = os.path.join(base_output_folder, f"threshold_{threshold}")
            os.makedirs(threshold_folder, exist_ok=True)
            
            output_csv = os.path.join(threshold_folder, f"{image_name}_detections.csv")
            with open(output_csv, mode='w', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(["Image Name", "Threshold", "Class Name", "Confidence", "x1", "y1", "x2", "y2"])
                
                for detection in data["results"].boxes:
                    bbox = detection.xyxy[0].tolist()  # [x1, y1, x2, y2]
                    class_id = int(detection.cls.item())  # Convert class ID to an integer
                    class_name = class_names[class_id] if class_id < len(class_names) else "Unknown"
                    conf = float(detection.conf.item())  # Convert confidence to float
                    
                    writer.writerow([image_name, threshold, class_name, conf, bbox[0], bbox[1], bbox[2], bbox[3]])
            
            print(f"Results saved to {output_csv}")
            
            # Load ground truth boxes
            ground_truth_boxes = load_ground_truth_csv(ground_truth_csv)
            ground_truth_boxes = ground_truth_boxes[['xmin', 'ymin', 'xmax', 'ymax']].values.tolist()
            
            # Extract boxes from the results at this threshold
            detected_boxes = [[*box.xyxy[0].tolist()] for box in data["results"].boxes if box.conf.item() >= threshold]
            
            # Calculate TPR and FPR for the threshold
            tpr, fpr, tp, fp, fn = calculate_tpr_fpr(detected_boxes, ground_truth_boxes)
            
            # Save metrics
            save_metrics(image_name, threshold, tpr, fpr, tp, fp, fn)

    except Exception as e:
        print(f"Error saving results to CSV: {e}")

# Calculate IoU between two bounding boxes
def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union = box1_area + box2_area - intersection

    return intersection / union if union > 0 else 0

# Calculate TPR, FPR, TP, FP, FN
def calculate_tpr_fpr(detections, ground_truths, iou_threshold=0.5):
    matched = set()
    tp = 0
    fp = 0
    fn = len(ground_truths)

    for det in detections:
        for gt_idx, gt in enumerate(ground_truths):
            if gt_idx in matched:
                continue

            iou = calculate_iou(det, gt)
            if iou >= iou_threshold:
                tp += 1
                matched.add(gt_idx)
                break
        else:
            fp += 1

    fn -= tp
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tp) if (fp + tp) > 0 else 0

    return tpr, fpr, tp, fp, fn

# Save metrics to a CSV file
def save_metrics(image_name, threshold, tpr, fpr, tp, fp, fn, metrics_output_dir="metrics"):
    os.makedirs(metrics_output_dir, exist_ok=True)
    metrics_file = os.path.join(metrics_output_dir, "detection_metrics.csv")
    
    file_exists = os.path.isfile(metrics_file)
    with open(metrics_file, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        if not file_exists:
            writer.writerow(['Image', 'Threshold', 'TPR', 'FPR', 'TP', 'FP', 'FN'])
        writer.writerow([image_name, threshold, tpr, fpr, tp, fp, fn])

# Load ground truth CSV
def load_ground_truth_csv(ground_truth_csv_path):
    try:
        ground_truth = pd.read_csv(ground_truth_csv_path)
        ground_truth['xmin'] = ground_truth['xmin'].astype(int)
        ground_truth['ymin'] = ground_truth['ymin'].astype(int)
        ground_truth['xmax'] = ground_truth['xmax'].astype(int)
        ground_truth['ymax'] = ground_truth['ymax'].astype(int)
        return ground_truth
    except Exception as e:
        print(f"Error loading ground truth CSV: {e}")
        return pd.DataFrame()

# Main logic
if __name__ == "__main__":
    # Load the class names from the YAML file
    class_names = load_classes_from_yaml(yaml_path)

    ground_truth_real_csv = r"Annotations\Scenario_fog\Real_annotated_fog\frame1008.csv"
    ground_truth_synthetic_csv = r"Annotations\Scenario_fog\Synth_annotated_fog\frame1008.csv"

    # Load the YOLO model
    model = load_yolo_model(weights_path)
    if model and class_names:
        # Run detection on both images
        real_results = run_detection(real_image_path, model, thresholds)
        synthetic_results = run_detection(synthetic_image_path, model, thresholds)

        # Save results to CSV organized by threshold
        if real_results:
            save_detections_to_csv_by_threshold(real_results, ground_truth_real_csv, output_folder, "Real_Image", class_names)
        if synthetic_results:
            save_detections_to_csv_by_threshold(synthetic_results, ground_truth_synthetic_csv, output_folder, "Synthetic_Image", class_names)
