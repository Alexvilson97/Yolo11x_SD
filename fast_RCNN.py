import torch
from torchvision import models, transforms
import cv2
import numpy as np
import json

# COCO class labels (for pre-trained models)
with open('coco_labels.json', 'r') as f:
    COCO_LABELS = json.load(f)

# Load pre-trained Fast R-CNN model (example with Torchvision)
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()  # Set the model to evaluation mode

# Image loading and preprocessing
image = cv2.imread("Images/fog_Highway/Frame_900_V100.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
transform = transforms.Compose([transforms.ToTensor()])
image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# Perform inference
with torch.no_grad():
    prediction = model(image_tensor)

# Post-process: extract bounding boxes, labels, and scores
boxes = prediction[0]['boxes'].cpu().numpy()
labels = prediction[0]['labels'].cpu().numpy()
scores = prediction[0]['scores'].cpu().numpy()

# Visualize the results
for box, label, score in zip(boxes, labels, scores):
    if score > 0.5:  # Filter by score threshold
        label_name = COCO_LABELS.get(str(label), "Unknown") # Get label name from dictionary
        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        cv2.putText(image, f'{label_name} ({score:.2f})', (int(box[0]), int(box[1])-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Convert image back to BGR for OpenCV and display
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
cv2.imshow("Detection Results", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
