import os
import cv2
import pandas as pd

# Helper function to draw bounding boxes on an image
def draw_bounding_boxes(image, boxes, color, label=""):
    """
    Draws bounding boxes on an image.
    :param image: The image to draw on
    :param boxes: List of bounding boxes [[xmin, ymin, xmax, ymax, class, confidence (optional)]]
    :param color: Color of the bounding boxes
    :param label: Label to add for bounding boxes (e.g., 'GT' or 'Detected')
    """
    for box in boxes:
        xmin, ymin, xmax, ymax = map(int, box[:4])
        cls = box[4]
        confidence = box[5] if len(box) > 5 else None
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
        label_text = f"{label} {cls}"
        if confidence is not None:
            label_text += f" ({confidence:.2f})"
        cv2.putText(image, label_text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Main function
def overlay_annotations_and_detections(image_folder, annotation_folder, detection_csv, output_folder):
    """
    Matches images with annotations and detected bounding boxes, overlays them, and saves the results.
    :param image_folder: Path to folder containing input images
    :param annotation_folder: Path to folder containing annotated CSV files
    :param detection_csv: Path to the detection CSV
    :param output_folder: Path to save images with bounding boxes
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # Load detections
    detections = pd.read_csv(detection_csv)
    
    # Process each image
    for image_file in os.listdir(image_folder):
        # Ensure the file is an image
        if not image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            continue
        
        # Load the image
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image: {image_path}")
            continue
        
        # Load ground truth annotations for the corresponding image
        base_name = os.path.splitext(image_file)[0]
        annotation_file = os.path.join(annotation_folder, f"{base_name}.csv")
        if not os.path.exists(annotation_file):
            print(f"No annotation file found for {image_file}, skipping.")
            continue
        
        ground_truths = pd.read_csv(annotation_file)
        ground_truth_boxes = ground_truths[['xmin', 'ymin', 'xmax', 'ymax', 'class']].values.tolist()
        
        # Filter detections for this specific image
        detection_boxes = detections[detections['image_name'] == image_file][
            ['xmin', 'ymin', 'xmax', 'ymax', 'class', 'confidence']
        ].values.tolist()
        
        # Overlay ground truth boxes in green
        draw_bounding_boxes(image, ground_truth_boxes, color=(0, 255, 0), label="GT")
        
        # Overlay detected boxes in red
        draw_bounding_boxes(image, detection_boxes, color=(0, 0, 255), label="Detected")
        
        # Save the image with bounding boxes
        output_path = os.path.join(output_folder, image_file)
        cv2.imwrite(output_path, image)
        print(f"Saved annotated image to {output_path}")

# Example usage
if __name__ == "__main__":
    image_folder = r"Yolo11x_SD\Images\real_frames"  # Replace with the folder containing images
    annotation_folder = r"Yolo11x_SD\Annotations\annotated_real"  # Replace with the folder containing annotated CSVs
    detection_csv = r"Yolo11x_SD\New_real_detected_csv"  # Replace with the detection CSV file
    output_folder = r"Path_to_output_images"  # Replace with the folder to save output images
    
    overlay_annotations_and_detections(image_folder, annotation_folder, detection_csv, output_folder)
