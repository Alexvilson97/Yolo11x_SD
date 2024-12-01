import torch
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Function to load YOLOv11x model
def load_yolo_model(weights_path="yolo11x.pt"):
    model = YOLO(weights_path)
    return model

# Function to preprocess the image (resize, normalize, etc.)
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image = cv2.resize(image, (640, 640))  # Resize image to 640x640 (common input size for YOLO)
    image = np.transpose(image, (2, 0, 1))  # Change to (C, H, W)
    image = image / 255.0  # Normalize to [0, 1]
    image = torch.tensor(image).float()  # Convert to tensor
    image = image.unsqueeze(0)  # Add batch dimension
    return image

# Function to get detections and plot TPR vs Confidence
# Function to get detections and plot TPR vs Confidence
# Function to get detections and plot TPR vs Confidence
def get_detections(model, image, threshold=0.4):
    with torch.no_grad():
        pred = model(image)  # Perform prediction

    # Extract the tensor for the first image (batch size = 1)
    detections = pred[0].boxes  # Use the 'boxes' attribute to get bounding boxes

    # Convert detections to a numpy array for easier manipulation
    detections = detections.data.cpu().numpy()

    # The model's output should have columns: [x1, y1, x2, y2, confidence, class]
    # If the output format contains confidence and class info (6 values), filter by confidence
    if detections.shape[1] == 6:
        detections = detections[detections[:, 4] > threshold]  # Filter out low-confidence detections
    elif detections.shape[1] == 4:
        # If there are no confidence values, just return the bounding boxes
        pass
    else:
        raise ValueError(f"Unexpected output shape: {detections.shape[1]}")

    return detections

# Function to calculate TPR and FPR
def calculate_metrics(detections, ground_truth, thresholds):
    tpr = []
    fpr = []
    for threshold in thresholds:
        # Get detections above threshold
        filtered_detections = detections[detections[:, 4] > threshold]
        true_positive = 0
        false_positive = 0
        false_negative = len(ground_truth)
        
        # Compare detections with ground truth
        for detection in filtered_detections:
            # Assuming ground_truth is a list of [x1, y1, x2, y2]
            for gt in ground_truth:
                # Calculate Intersection over Union (IoU)
                iou = compute_iou(detection[:4], gt)
                if iou > 0.5:  # If IoU > 0.5, consider it a true positive
                    true_positive += 1
                    false_negative -= 1
                    break
            else:
                false_positive += 1
        
        tpr.append(true_positive / (true_positive + false_negative))
        fpr.append(false_positive / len(ground_truth))
    
    return tpr, fpr

# Compute Intersection over Union (IoU)
def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_gt, y1_gt, x2_gt, y2_gt = box2
    # Compute area of intersection
    inter_x1 = max(x1, x1_gt)
    inter_y1 = max(y1, y1_gt)
    inter_x2 = min(x2, x2_gt)
    inter_y2 = min(y2, y2_gt)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    
    # Compute areas of both boxes
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2_gt - x1_gt) * (y2_gt - y1_gt)
    
    # Compute IoU
    union_area = area1 + area2 - inter_area
    iou = inter_area / union_area if union_area > 0 else 0
    return iou

# Function to visualize the plot of TPR vs Confidence
def plot_tpr_fpr(tpr, fpr, thresholds):
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, tpr, label="TPR")
    plt.plot(thresholds, fpr, label="FPR")
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Rate')
    plt.title('TPR and FPR vs Confidence Threshold')
    plt.legend()
    plt.grid(True)
    plt.show()

# Main function
def main(image1_path, image2_path, ground_truth_csv1, ground_truth_csv2, thresholds):
    # Load YOLOv11x model
    model = load_yolo_model(weights_path="yolo11x.pt")
    
    # Preprocess the images
    image1 = preprocess_image(image1_path)
    image2 = preprocess_image(image2_path)
    
    # Get detections for both images
    detections1 = get_detections(model, image1)
    detections2 = get_detections(model, image2)
    
    # Load ground truth from CSV files (assuming the CSV format is [x1, y1, x2, y2])
    ground_truth1 = pd.read_csv(ground_truth_csv1).values
    ground_truth2 = pd.read_csv(ground_truth_csv2).values
    
    # Calculate TPR and FPR for both images
    tpr1, fpr1 = calculate_metrics(detections1, ground_truth1, thresholds)
    tpr2, fpr2 = calculate_metrics(detections2, ground_truth2, thresholds)
    
    # Plot TPR vs Confidence Threshold for both images
    plt.plot(thresholds, tpr1, label='Image 1 TPR')
    plt.plot(thresholds, tpr2, label='Image 2 TPR')
    plt.xlabel('Confidence Threshold')
    plt.ylabel('TPR')
    plt.title('TPR vs Confidence Threshold Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # You can also plot FPR comparison if needed
    plt.plot(thresholds, fpr1, label='Image 1 FPR')
    plt.plot(thresholds, fpr2, label='Image 2 FPR')
    plt.xlabel('Confidence Threshold')
    plt.ylabel('FPR')
    plt.title('FPR vs Confidence Threshold Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage:
image1_path = r'Images\fog_real\RD_frame_1008.png'  # Path to the first image
image2_path = r'Images\fog_syn\Frame_1008_FOG_100m.png'  # Path to the second image
ground_truth_csv1 = r'Annotations\Scenario_fog\Real_annotated_fog\frame1008.csv'  # Path to the CSV for image 1
ground_truth_csv2 = r'Annotations\Scenario_fog\Synth_annotated_fog\frame_1008_fog_100m.csv'  # Path to the CSV for image 2
#weights_path = 'yolo11x.pt'  # Path to YOLOv11x model weights
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # List of thresholds to evaluate

main(image1_path, image2_path, ground_truth_csv1, ground_truth_csv2, thresholds)
