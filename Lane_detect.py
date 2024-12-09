import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_lanes(image_path):
    # Step 1: Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Step 2: Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Step 3: Apply edge detection using Canny
    edges = cv2.Canny(blur, 50, 150)
    
    # Step 4: Define region of interest (ROI) for lane detection
    height, width = edges.shape
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (int(0.1 * width), height),  # Bottom-left
        (int(0.9 * width), height),  # Bottom-right
        (int(0.55 * width), int(0.6 * height)),  # Top-right
        (int(0.45 * width), int(0.6 * height))   # Top-left
    ]], dtype=np.int32)
    cv2.fillPoly(mask, polygon, 255)
    cropped_edges = cv2.bitwise_and(edges, mask)
    
    # Step 5: Detect lines using Hough Transform
    lines = cv2.HoughLinesP(cropped_edges, rho=1, theta=np.pi/180, threshold=50, 
                            minLineLength=40, maxLineGap=5)
    
    # Step 6: Draw detected lines on the original image
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
    
    # Combine the line image with the original image
    result = cv2.addWeighted(image, 0.8, line_image, 1, 1)
    
    # Display the result
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title("Lane Detection Result")
    plt.axis("off")
    plt.show()

# Replace 'lane_image.png' with the path to your PNG image
image_path = r'Images\fog_real\RD_frame_0936.png'
detect_lanes(image_path)
