import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw
import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Load COCO YAML File for Class Labels
def load_coco_labels(yaml_path):
    # Open the YAML file with UTF-8 encoding
    with open(yaml_path, 'r', encoding='utf-8') as file:
        coco_data = yaml.safe_load(file)
    
    # Extract categories from the 'names' key
    if 'names' in coco_data:
        categories = {int(key): value for key, value in coco_data['names'].items()}
    else:
        raise KeyError("'names' key is missing in the YAML file")
    
    return categories

# Preprocess Image: Convert image to a tensor
def preprocess_image(image_path):
    image = Image.open(image_path)
    image_tensor = F.to_tensor(image).unsqueeze(0)  # Convert image to tensor and add batch dimension
    return image, image_tensor

# Load Pretrained Faster R-CNN Model and Modify it for Custom Classes
def load_model(num_classes):
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()  # Set model to evaluation mode

    # Modify the model to match the number of classes (including background class)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    
    return model

# Perform Object Detection
def detect_objects(model, image_tensor):
    with torch.no_grad():
        prediction = model(image_tensor)
    
    # Extract bounding boxes, labels, and scores
    boxes = prediction[0]['boxes']
    labels = prediction[0]['labels']
    scores = prediction[0]['scores']
    
    return boxes, labels, scores

# Draw Detected Objects on Image
def draw_boxes(image, boxes, labels, scores, categories, threshold=0.5):
    draw = ImageDraw.Draw(image)
    
    for i in range(len(boxes)):
        score = scores[i].item()
        if score >= threshold:  # Only draw boxes with confidence above the threshold
            class_id = labels[i].item()
            label = categories.get(class_id, 'Unknown')  # Map class ID to class name
            x1, y1, x2, y2 = boxes[i].tolist()  # Convert box coordinates to list
            
            # Draw the bounding box
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, y1), f'{label} {score:.2f}', fill="red")
    
    return image

# Display the Image with Bounding Boxes
def display_image_with_boxes(image):
    # Ensure the image is in the right format for display
    image.show()  # This opens the default image viewer

    # Also using matplotlib to show the image inline (for Jupyter or script environments)
    plt.imshow(image)
    plt.axis('off')  # Hide axes
    plt.show()


def display_results(image_path, boxes, labels, scores, categories):
    # Use matplotlib to display results
    image = Image.open(image_path)
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)

    for i in range(len(boxes)):
        class_id = labels[i].item()
        label = categories.get(class_id, 'Unknown')  # Map class ID to class name
        score = scores[i].item()
        
        # Create a Rectangle patch
        xmin, ymin, xmax, ymax = boxes[i].tolist()
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                 linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        # Add label and score text
        ax.text(xmin, ymin, f'{label} ({score:.2f})', color='r', fontsize=12)

    plt.show()


# Main Function
def main(image_path, yaml_path):
    # Load COCO class labels from YAML
    categories = load_coco_labels(yaml_path)
    
    # Load the pre-trained Faster R-CNN model
    num_classes = len(categories) + 1  # Adding 1 for background class
    model = load_model(num_classes)
    
    # Preprocess the image
    image, image_tensor = preprocess_image(image_path)
    
    # Perform object detection
    boxes, labels, scores = detect_objects(model, image_tensor)
    
    # Draw bounding boxes on the image
    image_with_boxes = draw_boxes(image, boxes, labels, scores, categories)
    
    # Display the image with bounding boxes
    display_image_with_boxes(image_with_boxes)

    # Display results using matplotlib for better visualization
    display_results(image_path, boxes, labels, scores, categories)


# Example Usage
if __name__ == "__main__":
    image_path = r'Images\fog_real\RD_frame_0936.png'  # Replace with the path to your image
    yaml_path = r'coco.yaml'  # Replace with the path to your COCO YAML file
    main(image_path, yaml_path)
