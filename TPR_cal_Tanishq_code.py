import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


def calculate_metrics(TP, FP, FN, total_objects):
    """Calculate precision, recall, FPR, FNR, and TPR."""
    precision = TP / max(TP + FP, 1)
    recall = TP / max(TP + FN, 1)
#    FPR = FP / max(total_objects - (TP + FN), 1)
#    FNR = FN / max(TP + FN, 1)
    TPR = recall

    return {
        'precision': precision,
        'recall': recall,
        'TPR': TPR,
        'TP': TP,
        'FP': FP,
        'FN': FN,
    }

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

    total_objects = sum([len(gt_boxes) for gt_boxes in all_ground_truth_boxes])

    for confidence in confidence_levels:
        TP_total, FP_total, FN_total = 0, 0, 0
        for frame_number, (gt_boxes, pred_boxes) in enumerate(zip(all_ground_truth_boxes, all_predicted_boxes)):
            pred_boxes_filtered = [box for box in pred_boxes if box[4] >= confidence]
            TP, FP, FN = process_frame(frame_number, gt_boxes, pred_boxes_filtered, iou_thresholds)
            TP_total += TP
            FP_total += FP
            FN_total += FN

        metrics = calculate_metrics(TP_total, FP_total, FN_total, total_objects)
        metrics['confidence'] = confidence
        metrics_results.append(metrics)

    return metrics_results

# Define CSV files for real and synthetic data
real_csv_files = [r"Annotations\frame15_RD.csv"]
synthetic_csv_files = [r"Annotations\frame15_SD.csv"]

confidence_levels = np.arange(0.3, 1.0, 0.1)
iou_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def process_files(csv_files, data_type):
    """Process a list of CSV files and add a 'data_type' column."""
    metrics_results = []
    for csv_path in csv_files:
        scenario_metrics = process_csv(csv_path, confidence_levels, iou_thresholds)

        for result in scenario_metrics:
            result['scenario'] = os.path.splitext(os.path.basename(csv_path))[0]
            result['data_type'] = data_type  # Add real/synthetic label

        metrics_results.extend(scenario_metrics)
    return metrics_results


# Process real and synthetic CSV files
all_metrics_results = []
all_metrics_results.extend(process_files(real_csv_files, "Real"))
all_metrics_results.extend(process_files(synthetic_csv_files, "Synthetic"))

# Create a single DataFrame from all results
df_metrics = pd.DataFrame(all_metrics_results)

# Save the combined DataFrame to a single CSV file
output_csv_path = "combined_tpr_metrics.csv"
df_metrics.to_csv(output_csv_path, index=False)

print(f"Metrics saved to {output_csv_path}")

# Plot the graphs
plt.figure(figsize=(12, 8))

# Plot TPR vs Confidence for Real and Synthetic data
for (scenario, data_type), group in df_metrics.groupby(['scenario', 'data_type']):
    label = f"{scenario} ({data_type})"
    plt.plot(group['confidence'], group['TPR'], marker='o', label=label)
    for i, tpr in enumerate(group['TPR']):
        plt.text(group['confidence'].values[i], tpr, f"{tpr:.2f}", fontsize=10)

plt.title("True Positive Rate vs Confidence Levels", fontsize=16)
plt.xlabel("Confidence Levels", fontsize=14)
plt.ylabel("True Positive Rate (TPR)", fontsize=14)
plt.legend()
plt.grid(True)
plt.show()