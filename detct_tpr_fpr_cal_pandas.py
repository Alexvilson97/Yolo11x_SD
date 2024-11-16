import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Helper Function: Calculate IoU
def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area_box1 + area_box2 - intersection

    return intersection / union if union > 0 else 0

# Helper Function: Match Detections to Ground Truths
def evaluate_detections(detections, ground_truths, iou_threshold=0.5):
    tp, fp, fn = 0, 0, 0
    matched_gt = set()

    for det in detections:
        best_iou = 0
        best_gt = -1
        for i, gt in enumerate(ground_truths):
            if i in matched_gt or gt[4] != det[4]:  # Check class match
                continue
            iou = calculate_iou(det[:4], gt[:4])
            if iou > best_iou:
                best_iou = iou
                best_gt = i

        if best_iou >= iou_threshold:
            tp += 1
            matched_gt.add(best_gt)
        else:
            fp += 1

    fn = len(ground_truths) - len(matched_gt)
    return tp, fp, fn

# Function to calculate metrics (this is assumed from your context)
def calculate_metrics(detections, ground_truths, confidence_thresholds, iou_threshold):
    precisions = []
    recalls = []
    tpr = []
    fpr = []
    # Simulate precision, recall, TPR, FPR
    for threshold in confidence_thresholds:
        tp = fp = fn = tn = 0  # Initialize counts
        for det in detections:
            if det[5] >= threshold:  # If the detection confidence is above the threshold
                # Evaluate detection
                tp_val, fp_val, fn_val = evaluate_detections([det], ground_truths, iou_threshold)
                tp += tp_val
                fp += fp_val
                fn += fn_val

        tn = len(ground_truths) - tp - fp  # Assuming that the rest are True Negatives
        tpr_value = tp / (tp + fn) if (tp + fn) != 0 else 0
        fpr_value = fp / (fp + tn) if (fp + tn) != 0 else 0  # Fix FPR calculation

        precisions.append(tp / (tp + fp) if (tp + fp) != 0 else 0)
        recalls.append(tp / (tp + fn) if (tp + fn) != 0 else 0)
        tpr.append(tpr_value)
        fpr.append(fpr_value)

    return precisions, recalls, tpr, fpr

# Function to normalize values
def normalize(values):
    max_val = max(values)
    min_val = min(values)
    return [(val - min_val) / (max_val - min_val) if max_val > min_val else 0 for val in values]

# Updated Plotting Functions
def plot_normalized_tpr(real_tpr, synthetic_tpr, confidence_thresholds):
    plt.figure(figsize=(8, 6))
    plt.plot(confidence_thresholds, real_tpr, marker='o', label='Normalized Real Image TPR')
    plt.plot(confidence_thresholds, synthetic_tpr, marker='x', label='Normalized Synthetic Image TPR')
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Normalized TPR')
    plt.title('Normalized TPR vs Confidence')
    plt.grid()
    plt.legend()
    plt.show()

def plot_normalized_fpr(real_fpr, synthetic_fpr, confidence_thresholds):
    plt.figure(figsize=(8, 6))
    plt.plot(confidence_thresholds, real_fpr, marker='o', label='Normalized Real Image FPR', color='red')
    plt.plot(confidence_thresholds, synthetic_fpr, marker='x', label='Normalized Synthetic Image FPR', color='orange')
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Normalized FPR')
    plt.title('Normalized FPR vs Confidence')
    plt.grid()
    plt.legend()
    plt.show()

# Main Program
if __name__ == "__main__":
    # Load CSVs for real image
    detection_file_real = r'New_real_detected_csv\frame_7.csv'  # Replace with your real detection CSV path
    ground_truth_file_real = r'Annotations\annotated_real\frame7.csv'  # Replace with your real ground truth CSV path
    
    # Load CSVs for synthetic image
    detection_file_synthetic = r'New_synth_detected_csv\15.csv'  # Replace with your synthetic detection CSV path
    ground_truth_file_synthetic = r'Annotations\annotated_syn\frame15.csv'  # Replace with your synthetic ground truth CSV path

    # Read real image detections and ground truths
    detections_real = pd.read_csv(detection_file_real)[['xmin', 'ymin', 'xmax', 'ymax', 'class', 'confidence']].values.tolist()
    ground_truths_real = pd.read_csv(ground_truth_file_real)[['xmin', 'ymin', 'xmax', 'ymax', 'class']].values.tolist()

    # Read synthetic image detections and ground truths
    detections_synthetic = pd.read_csv(detection_file_synthetic)[['xmin', 'ymin', 'xmax', 'ymax', 'class', 'confidence']].values.tolist()
    ground_truths_synthetic = pd.read_csv(ground_truth_file_synthetic)[['xmin', 'ymin', 'xmax', 'ymax', 'class']].values.tolist()

    # Define confidence thresholds and IoU threshold
    confidence_thresholds = np.linspace(0.1, 0.9, 9)
    iou_threshold = 0.5

    # Calculate metrics for real image
    precisions_real, recalls_real, tpr_real, fpr_real = calculate_metrics(
        detections_real, ground_truths_real, confidence_thresholds, iou_threshold
    )

    # Calculate metrics for synthetic image
    precisions_synthetic, recalls_synthetic, tpr_synthetic, fpr_synthetic = calculate_metrics(
        detections_synthetic, ground_truths_synthetic, confidence_thresholds, iou_threshold
    )

    # Normalize TPR and FPR values
    tpr_real_normalized = normalize(tpr_real)
    tpr_synthetic_normalized = normalize(tpr_synthetic)
    fpr_real_normalized = normalize(fpr_real)
    fpr_synthetic_normalized = normalize(fpr_synthetic)

    # Plot normalized TPR and FPR
    plot_normalized_tpr(tpr_real_normalized, tpr_synthetic_normalized, confidence_thresholds)
    plot_normalized_fpr(fpr_real_normalized, fpr_synthetic_normalized, confidence_thresholds)