import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score

# Perform object detection
def read_detections_from_csv(detections_csv):
    df = pd.read_csv(detections_csv)
    detections = df[['filename', 'xmin', 'ymin', 'xmax', 'ymax', 'class']].values.tolist()
    confidences = df['confidence'].values.tolist()
    class_ids = df['class'].values.tolist()
    return detections, confidences, class_ids

def load_ground_truth(ground_truth_csv):
    df = pd.read_csv(ground_truth_csv)
    ground_truths = df[['filename', 'xmin', 'ymin', 'xmax', 'ymax', 'class']].values.tolist()
    return ground_truths

def evaluate_detections(detections, ground_truths, iou_threshold=0.5):
    tp, fp, fn = 0, 0, 0
    matched_ground_truths = set()

    for det in detections:
        filename_det, det_box = det[0], det[1:5]
        det_class = det[5]
        max_iou = 0
        matched_gt = None

        for gt in ground_truths:
            filename_gt, gt_box = gt[0], gt[1:5]
            gt_class = gt[5]

            if filename_det != filename_gt or det_class != gt_class:
                continue

            iou = calculate_iou(det_box, gt_box)

            if iou > max_iou:
                max_iou = iou
                matched_gt = gt

        if max_iou >= iou_threshold and matched_gt:
            tp += 1
            matched_ground_truths.add(tuple(matched_gt))
        else:
            fp += 1

    fn = len(ground_truths) - len(matched_ground_truths)
    return tp, fp, fn

def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area if union_area > 0 else 0
    return iou

def evaluate_at_confidence_thresholds(detections, confidences, class_ids, ground_truths, iou_threshold):
    thresholds = np.arange(0.1, 1.0, 0.1)
    tpr_list, fpr_list = [], []

    for threshold in thresholds:
        filtered_detections = [det for det, conf in zip(detections, confidences) if conf >= threshold]
        tp, fp, fn = evaluate_detections(filtered_detections, ground_truths, iou_threshold)

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + len(ground_truths)) if (fp + len(ground_truths)) > 0 else 0

        tpr_list.append(tpr)
        fpr_list.append(fpr)
        
        
    return thresholds, tpr_list, fpr_list

def plot_images_with_metrics(real_image_path, synthetic_image_path, thresholds, real_tpr, real_fpr, synthetic_tpr, synthetic_fpr):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Plot the real image
    real_image = cv2.imread(real_image_path)
    if real_image is None:
        print(f"Warning: Real image not found at {real_image_path}")
        axes[0].text(0.5, 0.5, 'Real Image Not Found', horizontalalignment='center', verticalalignment='center', fontsize=12, color='red')
    else:
        real_image = cv2.cvtColor(real_image, cv2.COLOR_BGR2RGB)
        axes[0].imshow(real_image)
    
    axes[0].set_title("Real Image")
    axes[0].axis('off')

    # Plot the synthetic image
    synthetic_image = cv2.imread(synthetic_image_path)
    if synthetic_image is None:
        print(f"Warning: Synthetic image not found at {synthetic_image_path}")
        axes[1].text(0.5, 0.5, 'Synthetic Image Not Found', horizontalalignment='center', verticalalignment='center', fontsize=12, color='red')
    else:
        synthetic_image = cv2.cvtColor(synthetic_image, cv2.COLOR_BGR2RGB)
        axes[1].imshow(synthetic_image)
    
    axes[1].set_title("Synthetic Image")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

    # Plot TPR vs FPR
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, real_tpr, label="Real TPR", color='green', marker='o')
    ax.plot(thresholds, synthetic_tpr, label="Synthetic TPR", color='blue', marker='x')
    ax.plot(thresholds, real_fpr, label="Real FPR", color='red', linestyle='--', marker='o')
    ax.plot(thresholds, synthetic_fpr, label="Synthetic FPR", color='orange', linestyle='--', marker='x')
    
    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel("Rate")
    ax.set_title("TPR and FPR vs Confidence Threshold")
    ax.legend()
    ax.grid(True)
    plt.show()


def evaluate_and_plot_metrics(detections, confidences, class_ids, ground_truths, iou_threshold, real_image_path, synthetic_image_path):
    thresholds, real_tpr, real_fpr = evaluate_at_confidence_thresholds(detections, confidences, class_ids, ground_truths, iou_threshold)

    # Plot images and metrics together
    plot_images_with_metrics(real_image_path, synthetic_image_path, thresholds, real_tpr, real_fpr, real_tpr, real_fpr)

def main():
    real_detections_csv = r"real_detected_CSVs\frame_7_detections.csv"
    real_ground_truth_csv = "Annotations/annotated_real/frame7.csv"
    synthetic_detections_csv = r"syn_detected_CSVs\15_detections.csv"
    synthetic_ground_truth_csv = "Annotations/annotated_syn/frame15.csv"
    
    iou_threshold = 0.5

    real_detections, real_confidences, real_class_ids = read_detections_from_csv(real_detections_csv)
    real_ground_truths = load_ground_truth(real_ground_truth_csv)
    
    synthetic_detections, synthetic_confidences, synthetic_class_ids = read_detections_from_csv(synthetic_detections_csv)
    synthetic_ground_truths = load_ground_truth(synthetic_ground_truth_csv)

    real_image_path = r"Images\real_frames\frame_7.png"  # Adjust the image path
    synthetic_image_path = r"Images\sd_frames\15.png"  # Adjust the image path

    evaluate_and_plot_metrics(real_detections, real_confidences, real_class_ids, real_ground_truths, iou_threshold, real_image_path, synthetic_image_path)
    evaluate_and_plot_metrics(synthetic_detections, synthetic_confidences, synthetic_class_ids, synthetic_ground_truths, iou_threshold, real_image_path, synthetic_image_path)

if __name__ == "__main__":
    main()
