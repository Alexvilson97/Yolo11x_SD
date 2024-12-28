import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = "Images/fog_Highway/Frame_900_V100.png"
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur to reduce noise
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Use Canny edge detection
edges = cv2.Canny(blur, 50, 150)

# Define a region of interest (ROI)
def region_of_interest(img):
    height, width = img.shape[:2]
    mask = np.zeros_like(img)
    
    # Define the ROI as a polygon
    polygon = np.array([[
        (0, height),  # Bottom-left
        (width // 2 - 100, height // 2),  # Top-middle-left
        (width // 2 + 100, height // 2),  # Top-middle-right
        (width, height)  # Bottom-right
    ]], np.int32)
    
    cv2.fillPoly(mask, polygon, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

roi_edges = region_of_interest(edges)

# Perform Hough Line Transformation to detect lines
lines = cv2.HoughLinesP(roi_edges, rho=1, theta=np.pi / 180, threshold=100, minLineLength=200, maxLineGap=150)

# Function to calculate line length and confidence score
def calculate_line_confidence(line):
    x1, y1, x2, y2 = line
    length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)  # Length of the line
    confidence = min(length / 400, 1.0)  # Normalize confidence, length over arbitrary threshold (e.g., 400 pixels)
    return length, confidence


# Function to filter lines based on slope and length
def filter_lines(lines, min_slope=0.2, max_slope=2.0, min_length=100):
    filtered_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Calculate the slope of the line (avoid vertical lines with infinite slope)
        if x2 - x1 != 0:
            slope = (y2 - y1) / (x2 - x1)
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            
            # Apply slope and length filters
            if min_slope <= abs(slope) <= max_slope and length >= min_length:
                filtered_lines.append(line)
    return filtered_lines

# Filter the detected lines (for now, without position-based filter)
if lines is not None:
    lines = filter_lines(lines)
# Filter the detected lines
if lines is not None:
    lines = filter_lines(lines)

# Draw the detected lines and show confidence
output = np.copy(image)
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Calculate the line's length and confidence score
        length, confidence = calculate_line_confidence((x1, y1, x2, y2))
        
        # Draw the line
        cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Display the confidence score near the line
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(output, f'{confidence*100:.1f}%', (x1, y1 - 10), font, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

# Save the output image with lane detection and confidence scores
output_image_path = "fog_highway_frame900_sd.jpg"
cv2.imwrite(output_image_path, output)

# Display the results
plt.figure(figsize=(10, 10))
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.subplot(1, 3, 2)
plt.title("Edges")
plt.imshow(edges, cmap="gray")

plt.subplot(1, 3, 3)
plt.title("Lane Detection with Confidence")
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))

plt.show()
