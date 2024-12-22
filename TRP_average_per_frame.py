import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


def calculate_iou(box1, box2):
    # Calculate intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate union
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection

    # Calculate IoU
    iou = intersection / union if union != 0 else 0
    return iou

def process_frame(frame_number, ground_truth_boxes, predicted_boxes, iou_thresholds):
    TP = 0
    FP = 0
    FN = 0
    matched_gt_boxes = set()

    num_gt_boxes = len(ground_truth_boxes)
    num_pred_boxes = len(predicted_boxes)

    # Calculate IoU and match ground truth with predictions
    for pred_idx, pred_box in enumerate(predicted_boxes):
        if len(pred_box) < 5:
            continue
        for gt_idx, gt_box in enumerate(ground_truth_boxes):
            if len(gt_box) < 4:
                continue
            iou = calculate_iou(pred_box[:4], gt_box)

            if iou >= min(iou_thresholds):
                matched_gt_boxes.add(gt_idx)

    # Determine TP, FP, FN
    TP = len(matched_gt_boxes)
    FP = num_pred_boxes - TP
    FN = num_gt_boxes - len(matched_gt_boxes)

    # Ensure no negative values
    FP = max(FP, 0)
    FN = max(FN, 0)

    return TP, FP, FN

def calculate_frame_metrics(TP, FP, FN):
    """Calculate TPR, Precision, and Recall for a single frame."""
    TPR = TP / max(TP + FN, 1)  # Sensitivity
    Precision = TP / max(TP + FP, 1)
    Recall = TPR  # By definition, TPR = Recall
    return TPR, Precision, Recall

def process_csv(csv_path, confidence_levels, iou_thresholds):
    """Process a single CSV file to calculate metrics."""
    combined_df = pd.read_csv(csv_path)
    frame_numbers = combined_df['Frame Number'].unique()

    all_ground_truth_boxes = []
    all_predicted_boxes = []

    for frame_number in frame_numbers:
        frame_data = combined_df[combined_df['Frame Number'] == frame_number]
        gt_boxes = frame_data[['Ground Truth X-min', 'Ground Truth Y-min', 'Ground Truth X-max',
                               'Ground Truth Y-max']].dropna().values.tolist()
        pred_boxes = frame_data[['Predicted X-min', 'Predicted Y-min', 'Predicted X-max', 'Predicted Y-max',
                                 'Confidence']].dropna().values.tolist()
        all_ground_truth_boxes.append(gt_boxes)
        all_predicted_boxes.append(pred_boxes)

    metrics_results = []

    for confidence in confidence_levels:
        frame_tprs = []
        frame_precisions = []
        frame_recalls = []
        total_TP, total_FP, total_FN = 0, 0, 0

        for frame_number, (gt_boxes, pred_boxes) in enumerate(zip(all_ground_truth_boxes, all_predicted_boxes)):
            pred_boxes_filtered = [box for box in pred_boxes if box[4] >= confidence]
            TP, FP, FN = process_frame(frame_number, gt_boxes, pred_boxes_filtered, iou_thresholds)

            # Update cumulative metrics
            total_TP += TP
            total_FP += FP
            total_FN += FN

            # Calculate metrics for the frame
            frame_tpr, frame_precision, frame_recall = calculate_frame_metrics(TP, FP, FN)
            frame_tprs.append(frame_tpr)
            frame_precisions.append(frame_precision)
            frame_recalls.append(frame_recall)

        # Calculate average metrics across frames for this confidence level
        avg_tpr = np.mean(frame_tprs)
        avg_precision = np.mean(frame_precisions)
        avg_recall = np.mean(frame_recalls)

        metrics_results.append({
            'confidence': confidence,
            'average_tpr': avg_tpr,
            'average_precision': avg_precision,
            'average_recall': avg_recall,
            'total_TP': total_TP,
            'total_FP': total_FP,
            'total_FN': total_FN
        })

    return metrics_results

# Process real and synthetic CSV files as before
all_metrics_results = []
all_metrics_results.extend(process_frame(real_csv_files, "Real"))
all_metrics_results.extend(process_frame(synthetic_csv_files, "Synthetic"))

# Create a DataFrame for all metrics
df_metrics = pd.DataFrame(all_metrics_results)

# Save metrics to a CSV file
output_csv_path = "detailed_metrics.csv"
df_metrics.to_csv(output_csv_path, index=False)

print(f"Metrics saved to {output_csv_path}")

# Plot the graphs
plt.figure(figsize=(12, 8))

# Plot metrics for Real and Synthetic data
metrics_to_plot = ['average_tpr', 'average_precision', 'average_recall']
for metric in metrics_to_plot:
    for (scenario, data_type), group in df_metrics.groupby(['scenario', 'data_type']):
        label = f"{scenario} ({data_type}) - {metric.replace('average_', '').capitalize()}"
        plt.plot(group['confidence'], group[metric], marker='o', label=label)
        for i, value in enumerate(group[metric]):
            plt.text(group['confidence'].values[i], value, f"{value:.2f}", fontsize=9)

plt.title("Metrics vs Confidence Levels", fontsize=16)
plt.xlabel("Confidence Levels", fontsize=14)
plt.ylabel("Metrics (TPR, Precision, Recall)", fontsize=14)
plt.legend()
plt.grid(True)

# Format y-axis as a percentage
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
plt.show()
