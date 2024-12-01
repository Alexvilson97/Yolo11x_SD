import os
import csv
import cv2
from ultralytics import YOLO
import numpy as np
import yaml
import re
import chardet

# Load YOLO model
def load_yolo_model(weights_path="yolo11x.pt"):
    model = YOLO(weights_path)
    return model

# Function to perform object detection
def detect_objects(image, model, conf_threshold=0.3, nms_threshold=0.9):
    results = model(image, imgsz=1280)
    
    boxes = []
    confidences = []
    class_ids = []

    for result in results:
        for box in result.boxes:
            xywh = box.xywh[0].cpu().numpy()
            confidence = box.conf[0].item()
            class_id = int(box.cls[0].item())

            if confidence > conf_threshold:
                x, y, w, h = xywh
                x = int(x - w / 2)
                y = int(y - h / 2)
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

# Load class names from yaml
def load_classes_from_yaml(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data['names']

def calculate_iou(box1, box2):
    # Calculate the Intersection over Union (IoU) between two bounding boxes
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union = box1_area + box2_area - intersection

    return intersection / union if union > 0 else 0

def calculate_tpr_fpr(detections, ground_truths, iou_threshold=0.5):
    # Match detections to ground truths using IoU
    matched = set()
    tp = 0
    fp = 0
    fn = len(ground_truths)

    for det_idx, det in enumerate(detections):
        for gt_idx, gt in enumerate(ground_truths):
            if gt_idx in matched:
                continue

            iou = calculate_iou(det, gt)
            if iou >= iou_threshold:
                tp += 1
                matched.add(gt_idx)
                break
        else:
            # Detection does not match any ground truth
            fp += 1

    fn -= tp  # Remaining unmatched ground truths
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tp) if (fp + tp) > 0 else 0

    return tpr, fpr, tp, fp, fn

def save_metrics(image_path, tpr, fpr, tp, fp, fn, metrics_output_dir="metrics"):
    os.makedirs(metrics_output_dir, exist_ok=True)
    metrics_file = os.path.join(metrics_output_dir, "fog_syn_detection_metrics.csv")
    
    # Append metrics to CSV
    file_exists = os.path.isfile(metrics_file)
    with open(metrics_file, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        if not file_exists:
            writer.writerow(['Image', 'TPR', 'FPR', 'TP', 'FP', 'FN'])
        writer.writerow([os.path.basename(image_path), tpr, fpr, tp, fp, fn])

def detect_file_encoding(file_path):
    """
    Detect the encoding of a file using chardet.
    """
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result.get('encoding', 'utf-8')

def save_detections_and_ground_truth(
    image_path, 
    ground_truth_csv, 
    boxes, 
    confidences, 
    class_ids, 
    classes, 
    image_output_dir='fog_syn_Detected_Images', 
    csv_output_dir='fog_syn_Detected_CSVs'
):
    # Create output directories if they don't exist
    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(csv_output_dir, exist_ok=True)

    # Load the original image
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Error: Could not load image {image_path}")
        return

    # Create a copy for YOLO detections
    yolo_image = original_image.copy()

    # Save YOLO detections to CSV
    yolo_csv_filename = os.path.join(csv_output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_detections.csv")
    with open(yolo_csv_filename, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['filename', 'class', 'xmin', 'ymin', 'xmax', 'ymax', 'confidence'])

        for i in range(len(boxes)):
            x, y, w, h = boxes[i]
            xmax = x + w
            ymax = y + h
            writer.writerow([os.path.basename(image_path), classes[class_ids[i]], x, y, xmax, ymax, confidences[i]])

            # Draw YOLO detections (green) on YOLO image
            label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
            cv2.rectangle(yolo_image, (x, y), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(yolo_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    print(f"YOLO detections CSV saved to: {yolo_csv_filename}")

    # Save YOLO-only detected image
    detected_image_path = os.path.join(image_output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_detections.jpg")
    cv2.imwrite(detected_image_path, yolo_image)
    print(f"YOLO detections image saved to: {detected_image_path}")

    # Overlay ground truth on the combined image
    combined_image = yolo_image.copy()
    with open(ground_truth_csv, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            xmin = int(row['xmin'])
            ymin = int(row['ymin'])
            xmax = int(row['xmax'])
            ymax = int(row['ymax'])
            label = row['class']

            # Draw ground truth boxes (blue) on combined image
            cv2.rectangle(combined_image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
            cv2.putText(combined_image, f"GT: {label}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Save combined image
    combined_image_path = os.path.join(image_output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_combined.jpg")
    cv2.imwrite(combined_image_path, combined_image)
    print(f"Combined image saved to: {combined_image_path}")
    
    ground_truth_boxes = []
    with open(ground_truth_csv, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            xmin = int(row['xmin'])
            ymin = int(row['ymin'])
            xmax = int(row['xmax'])
            ymax = int(row['ymax'])
            ground_truth_boxes.append([xmin, ymin, xmax - xmin, ymax - ymin])

    # Calculate TPR and FPR
    tpr, fpr, tp, fp, fn = calculate_tpr_fpr(boxes, ground_truth_boxes)

    # Save metrics
    save_metrics(image_path, tpr, fpr, tp, fp, fn)
    print(f"Metrics for {image_path} -> TPR: {tpr:.2f}, FPR: {fpr:.2f}, TP: {tp}, FP: {fp}, FN: {fn}")


def process_folder(folder_path, model, classes, ground_truth_csv_folder, image_output_dir='fog_syn_detected_Images', csv_output_dir='fog_syn_detected_CSVs'):
    for image_filename in os.listdir(folder_path):
        if image_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, image_filename)
            
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not load image: {image_path}")
                continue

            # Perform YOLO detection
            boxes, confidences, class_ids = detect_objects(image, model)

            # Find corresponding ground truth CSV file
            frame_number = extract_frame_number(image_filename)
            ground_truth_csv = os.path.join(ground_truth_csv_folder, f"frame{frame_number}.csv")

            if os.path.exists(ground_truth_csv):
                save_detections_and_ground_truth(
                    image_path, ground_truth_csv, boxes, confidences, class_ids, classes, 
                    image_output_dir, csv_output_dir
                )
            else:
                print(f"Ground truth CSV not found for {image_filename}")

# Main function
def main(folder_path, ground_truth_csv_folder):
    print("Starting batch YOLOv8 object detection...")
    
    yaml_path = "coco.yaml"
    classes = load_classes_from_yaml(yaml_path)
    
    model = load_yolo_model(weights_path="yolo11x.pt")
    
    # Process folder for YOLO detection and save results
    process_folder(folder_path, model, classes, ground_truth_csv_folder)



def extract_frame_number(filename):
    # Extract the numeric part of the filename (e.g., 19 from "19.png")
    match = re.search(r'\d+', filename)
    if match:
        return match.group(0)  # Return the numeric part as string
    return None

# Example usage
if __name__ == "__main__":
    folder_path = r"Images\fog_syn"  # Folder containing multiple images_
    ground_truth_csv_folder = r"Annotations\Scenario_fog\Synth_annotated_fog"  # Folder containing ground truth CSVs
    main(folder_path, ground_truth_csv_folder)
