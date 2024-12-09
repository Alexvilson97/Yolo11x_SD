import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection

    iou = intersection / union if union != 0 else 0
    return iou

def process_frame(ground_truth_boxes, predicted_boxes):
    TP = 0
    FP = 0
    FN = 0
    matched_gt_boxes = set()

    num_gt_boxes = len(ground_truth_boxes)
    num_pred_boxes = len(predicted_boxes)

    for pred_box in predicted_boxes:
        if len(pred_box) < 5:
            continue
        for gt_idx, gt_box in enumerate(ground_truth_boxes):
            if len(gt_box) < 4:
                continue
            iou = calculate_iou(pred_box[:4], gt_box)

            if iou >= 0.5:  # Threshold for IoU
                matched_gt_boxes.add(gt_idx)

    TP = len(matched_gt_boxes)
    FP = num_pred_boxes - TP
    FN = num_gt_boxes - len(matched_gt_boxes)

    return TP, FP, FN

def calculate_fpr(TP, FP, FN, total_negatives):
    TN = total_negatives
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0    
    print(f"Debug: TP={TP}, FP={FP}, FN={FN}, Total Negatives={total_negatives}, FPR={FPR}")  # Debugging line
    return FPR

def plot_fpr_vs_confidence(csv_files, confidence_levels, output_csv):
    results = []  # To store results for saving to CSV
    plt.figure(figsize=(12, 8))

    for csv_file in csv_files:
        combined_df = pd.read_csv(csv_file)
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

        total_objects = sum([len(gt_boxes) for gt_boxes in all_ground_truth_boxes])
        
        print(f"Debug: Total Objects={total_objects}")


        fpr_values = []  # Collect FPR values for this dataset
        for confidence in confidence_levels:
            TP_total, FP_total, FN_total = 0, 0, 0
            for gt_boxes, pred_boxes in zip(all_ground_truth_boxes, all_predicted_boxes):
                pred_boxes_filtered = [box for box in pred_boxes if box[4] >= confidence]
                TP, FP, FN = process_frame(gt_boxes, pred_boxes_filtered)
                TP_total += TP
                FP_total += FP
                FN_total += FN

                # Calculate Total Negatives and Debug
            total_negatives = total_objects - TP_total
            print(f"Debug: TP={TP_total}, FP={FP_total}, FN={FN_total}, Total Negatives={total_negatives}")

            # Calculate FPR and Debug
            FPR = calculate_fpr(TP_total, FP_total, FN_total, total_negatives)
            print(f"Debug: Confidence={confidence}, FPR={FPR}")

            fpr_values.append(FPR)
            results.append({
                'Confidence': confidence,
                'FPR': FPR,
                'TP': TP_total,
                'FP': FP_total,
                'FN': FN_total,
                'Scenario': os.path.splitext(os.path.basename(csv_file))[0]
            })

        # Plot the line for FPR
        plt.plot(confidence_levels, fpr_values, marker='o', label=os.path.splitext(os.path.basename(csv_file))[0])
        for confidence, fpr in zip(confidence_levels, fpr_values):
            plt.text(confidence, fpr, f"{fpr:.2f}", fontsize=10, ha='center')

    plt.title("False Positive Rate vs Confidence Levels")
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
    csv_files = [r"Annotations\Scenario_fog\Synth_annotated_fog\frame1008.csv"]
    confidence_levels = np.arange(0.3, 1.0, 0.1)
    output_csv = "fpr_results.csv"  # Output CSV file name

    plot_fpr_vs_confidence(csv_files, confidence_levels, output_csv)
