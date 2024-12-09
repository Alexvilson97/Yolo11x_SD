import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Calculate intersection area
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection

    return intersection / union if union > 0 else 0

def process_frame(ground_truth_boxes, predicted_boxes, iou_threshold=0.5):
    """Process a single frame to calculate TP, FP, and FN."""
    tp_count = 0  # True positives
    fp_count = 0  # False positives
    fn_count = len(ground_truth_boxes)  # All ground truth are unmatched initially
    matched_gt = set()  # To track matched ground truth indices

    for pred_box in predicted_boxes:
        pred_xmin, pred_ymin, pred_xmax, pred_ymax, confidence, pred_class = pred_box
        matched = False
        for gt_idx, gt_box in enumerate(ground_truth_boxes):
            gt_xmin, gt_ymin, gt_xmax, gt_ymax, gt_class = gt_box
            if pred_class == gt_class:  # Check if class names match
                iou = calculate_iou([pred_xmin, pred_ymin, pred_xmax, pred_ymax], [gt_xmin, gt_ymin, gt_xmax, gt_ymax])
                if iou >= iou_threshold and gt_idx not in matched_gt:
                    tp_count += 1
                    matched_gt.add(gt_idx)
                    fn_count -= 1
                    matched = True
                    break
        if not matched:
            fp_count += 1  # Unmatched prediction is a false positive

    # Debug output for each predicted box
    print(f"TP={tp_count}, FP={fp_count}, FN={fn_count}, Matched GT indices: {matched_gt}")

    return tp_count, fp_count, fn_count

def calculate_fpr(fp, tn):
    """Calculate False Positive Rate (FPR)."""
    return fp / (fp + tn) if (fp + tn) > 0 else 0

def plot_fpr_vs_confidence(synth_csv_files, real_csv_files, confidence_levels, output_csv):
    """Calculate and plot FPR at different confidence levels for synthetic and real images."""
    results = []  # Store results for CSV
    plt.figure(figsize=(12, 8))

    def process_files(csv_files, label_prefix):
        fpr_values = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            frame_numbers = df['Frame Number'].unique()

            all_ground_truth_boxes = []
            all_predicted_boxes = []

            for frame_number in frame_numbers:
                frame_data = df[df['Frame Number'] == frame_number]
                gt_boxes = frame_data[['Ground Truth X-min', 'Ground Truth Y-min', 'Ground Truth X-max', 'Ground Truth Y-max', 'Ground Truth Class Name']].dropna().values.tolist()
                pred_boxes = frame_data[['Predicted X-min', 'Predicted Y-min', 'Predicted X-max', 'Predicted Y-max', 'Confidence', 'Predicted Class Name']].dropna().values.tolist()
                all_ground_truth_boxes.append(gt_boxes)
                all_predicted_boxes.append(pred_boxes)

            total_objects = sum(len(gt_boxes) for gt_boxes in all_ground_truth_boxes)
            print(f"Debug: {label_prefix} Total Objects={total_objects}")

            file_fpr_values = []
            for confidence in confidence_levels:
                tp_total, fp_total, fn_total = 0, 0, 0
                for gt_boxes, pred_boxes in zip(all_ground_truth_boxes, all_predicted_boxes):
                    # Filter predictions by confidence level
                    pred_boxes_filtered = [box for box in pred_boxes if box[4] >= confidence]
                    tp, fp, fn = process_frame(gt_boxes, pred_boxes_filtered)
                    tp_total += tp
                    fp_total += fp
                    fn_total += fn

                # Calculate total negatives (TN)
                tn_total = len([gt for gt in all_ground_truth_boxes for _ in gt]) - tp_total
                print(f"Debug: {label_prefix} Confidence={confidence:.2f} | TP: {tp_total}, FP: {fp_total}, FN: {fn_total}, TN: {tn_total}")

                # Calculate FPR
                fpr = calculate_fpr(fp_total, tn_total)
                print(f"Debug: {label_prefix} Confidence={confidence:.2f} | FPR: {fpr:.4f}")

                file_fpr_values.append(fpr)
                results.append({
                    'Confidence': confidence,
                    'FPR': fpr,
                    'TP': tp_total,
                    'FP': fp_total,
                    'FN': fn_total,
                    'Scenario': label_prefix + os.path.splitext(os.path.basename(csv_file))[0]
                })

            # Plot FPR for this file
            plt.plot(confidence_levels, file_fpr_values, marker='o', label=f"{label_prefix} {os.path.splitext(os.path.basename(csv_file))[0]}")
            for confidence, fpr in zip(confidence_levels, file_fpr_values):
                plt.text(confidence, fpr, f"{fpr:.2f}", fontsize=8, ha='center')

    # Process synthetic images
    process_files(synth_csv_files, "Synthetic")

    # Process real images
    process_files(real_csv_files, "Real")

    # Finalize plot
    plt.title("False Positive Rate vs Confidence Levels (Synthetic vs Real)")
    plt.xlabel("Confidence Levels")
    plt.ylabel("False Positive Rate (FPR)")
    plt.xticks(confidence_levels)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.show()

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

if __name__ == "__main__":
    synth_csv_files = ["Annotations/Scenario_fog/Synth_annotated_fog/frame1008.csv"]  # Synthetic CSV files
    real_csv_files = ["Annotations/Scenario_real/Real_annotated/frame1008.csv"]  # Real CSV files
    confidence_levels = np.arange(0.3, 1.0, 0.1)
    output_csv = "fpr_results_combined.csv"

    plot_fpr_vs_confidence(synth_csv_files, real_csv_files, confidence_levels, output_csv)


if __name__ == "__main__":
    csv_files = [r"Annotations\frame15_real_combined.csv"]
    confidence_levels = np.arange(0.3, 1.0, 0.1)
    output_csv = "fpr_results.csv"

    plot_fpr_vs_confidence(csv_files, confidence_levels, output_csv)
