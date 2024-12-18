import cv2
import numpy as np

def region_of_interest(img, vertices):
    """
    Applies an image mask to keep only the region defined by the polygon `vertices`.
    """
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

def draw_lane_lines(img, lines, color=(0, 255, 0), thickness=5):
    """
    Draws lane lines on the image by averaging and extrapolating them.
    """
    left_lines = []
    right_lines = []
    
    if lines is None:
        return
    
    # Separate lines into left and right based on their slope
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 0
            if 0.5 < slope < 2:  # Right lane
                right_lines.append((x1, y1, x2, y2))
            elif -2 < slope < -0.5:  # Left lane
                left_lines.append((x1, y1, x2, y2))
    
    # Average left and right lines
    def average_lines(lines):
        if len(lines) == 0:
            return None
        x_coords = []
        y_coords = []
        for x1, y1, x2, y2 in lines:
            x_coords.extend([x1, x2])
            y_coords.extend([y1, y2])
        poly_fit = np.polyfit(y_coords, x_coords, 1)  # Fit a line: x = my + b
        y1 = img.shape[0]  # Bottom of the image
        y2 = int(y1 * 0.6)  # Slightly above the middle
        x1 = int(poly_fit[0] * y1 + poly_fit[1])
        x2 = int(poly_fit[0] * y2 + poly_fit[1])
        return (x1, y1, x2, y2)

    # Draw the averaged lines
    left_line = average_lines(left_lines)
    right_line = average_lines(right_lines)
    
    if left_line is not None:
        cv2.line(img, (left_line[0], left_line[1]), (left_line[2], left_line[3]), color, thickness)
    if right_line is not None:
        cv2.line(img, (right_line[0], right_line[1]), (right_line[2], right_line[3]), color, thickness)

def detect_lanes(image_path):
    """
    Detect lanes in an image while excluding the car bonnet.
    """
    # Load the image
    img = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian Blur
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
    
    # Use Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Define region of interest (ROI) to exclude the car bonnet
    height, width = edges.shape
    roi_vertices = np.array([[
        (width * 0.1, height),  # Bottom left
        (width * 0.9, height),  # Bottom right
        (width * 0.6, height * 0.6),  # Top right
        (width * 0.4, height * 0.6)   # Top left
    ]], dtype=np.int32)
    masked_edges = region_of_interest(edges, roi_vertices)
    
    # Use Hough Transform to detect lines
    lines = cv2.HoughLinesP(
        masked_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=40,
        maxLineGap=100
    )
    
    # Draw lane lines on a blank image
    line_img = np.zeros_like(img)
    draw_lane_lines(line_img, lines)
    
    # Combine the lane lines with the original image
    result = cv2.addWeighted(img, 0.8, line_img, 1, 0)
    
    # Save the result
    output_path = "lane_detected_refined.jpg"
    cv2.imwrite(output_path, result)
    print(f"Lane detection complete. Saved as '{output_path}'.")

# Run the lane detection
detect_lanes("Images/fog_syn/Frame_0720.V100.png")
