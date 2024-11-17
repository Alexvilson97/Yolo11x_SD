import os
import csv
import cv2
from ultralytics import YOLO
import yaml

# Load YOLO model
def load_yolo_model(weights_path="yolo11x.pt"):
    model = YOLO(weights_path)
    return model

# Function to perform object detection
def detect_objects(image, model, conf_threshold=0.1, nms_threshold=0.9):
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

# Save detections to CSV and save image with detections in a different folder
def save_detections_to_csv(boxes, confidences, class_ids, classes, image_path, csv_output_dir='New_real_detected_csv', image_output_dir='real_Detected_Images'):
    image_filename = os.path.basename(image_path)
    image_name, _ = os.path.splitext(image_filename)
    
    # Prepare CSV file path
    csv_filename = f"{image_name}.csv"
    csv_path = os.path.join(csv_output_dir, csv_filename)
    
    # Prepare image output folder
    os.makedirs(csv_output_dir, exist_ok=True)
    os.makedirs(image_output_dir, exist_ok=True)

    # Load and annotate the image
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    # Save detections to CSV
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax', 'confidence'])
        
        for i in range(len(boxes)):
            x, y, w, h = boxes[i]
            xmax = x + w
            ymax = y + h
            
            writer.writerow([
                image_filename,
                width,
                height,
                classes[class_ids[i]],
                x,
                y,
                xmax,
                ymax,
                confidences[i]
            ])
            
            # Draw bounding box and label on the image
            label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
            cv2.rectangle(image, (x, y), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the annotated image to the specified output folder
    detected_image_path = os.path.join(image_output_dir, f"{image_name}_detected.jpg")
    cv2.imwrite(detected_image_path, image)
    
    print(f"Detections saved to: {csv_path}")
    print(f"Detected image saved to: {detected_image_path}")

# Process all images in the folder
def process_folder(folder_path, model, classes, csv_output_dir='New_real_detected_csv', image_output_dir='real_Detected_Images'):
    for image_filename in os.listdir(folder_path):
        if image_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, image_filename)
            
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not load image: {image_path}")
                continue

            boxes, confidences, class_ids = detect_objects(image, model)
            save_detections_to_csv(boxes, confidences, class_ids, classes, image_path, csv_output_dir, image_output_dir)

# Main function
def main(folder_path):
    print("Starting batch YOLOv8 object detection...")
    
    yaml_path = "coco.yaml"
    classes = load_classes_from_yaml(yaml_path)
    
    model = load_yolo_model(weights_path="yolo11x.pt")
    process_folder(folder_path, model, classes)

# Example usage
if __name__ == "__main__":
    folder_path = "Images/real_frames"  # Folder containing multiple images
    main(folder_path)
