import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import csv
from skimage.metrics import structural_similarity as ssim
import torch
import lpips
import warnings

warnings.filterwarnings("ignore")
# Paths for real and synthetic images
real_folder = "Images/Highway_sunny/RD"
synthetic_folder = "Images/Highway_sunny/SD"
output_folder = "SSIM/SSIM_outputs_Highway_Sunny"
os.makedirs(output_folder, exist_ok=True)

# Helper function to compute multi-scale SSIM (MS-SSIM)
def compute_ms_ssim(image1, image2):
    return ssim(image1, image2, data_range=image2.max() - image2.min())

# Helper function to convert grayscale to RGB
def convert_to_rgb(image):
    return np.stack([image] * 3, axis=-1)

def sharpen_image(image):
    # Create a kernel for sharpening using unsharp mask
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)

def reduce_noise(image):
    # Convert image to 32-bit float for bilateral filter
    image_float32 = np.float32(image)
    
    # Apply bilateral filter
    return cv2.bilateralFilter(image_float32, d=9, sigmaColor=75, sigmaSpace=75)

def enhance_edges(image):
    # Apply Canny edge detection
    edges = cv2.Canny(np.uint8(image * 255), 100, 200)  # Canny requires uint8 images, so multiply by 255
    return edges


def multi_scale_edges(image):
    # Create a Gaussian pyramid to capture edges at multiple scales
    pyramid = [image]
    for i in range(4):  # Create 4 scales (you can adjust this)
        pyramid.append(cv2.pyrDown(pyramid[-1]))  # Downsample to create lower resolutions

    # Perform edge detection at each scale
    edges = [enhance_edges(scale) for scale in pyramid]
    
    # Resize all edge images to the size of the original image
    edges_resized = [cv2.resize(edge, (image.shape[1], image.shape[0])) for edge in edges]
    
    # Now compute the average of the resized edge images
    return np.mean(edges_resized, axis=0)  # Average edges from all scales

def normalize_image(image):
    return np.clip(image, 0, 1)  # Ensure all values are in the [0, 1] range

def equalize_histogram(image):
    return cv2.equalizeHist((image * 255).astype(np.uint8)) / 255.0  # Return normalized image


def enhance_image(image):
    # Apply sharpening
    sharpened_image = sharpen_image(image)
    
    # Apply noise reduction
    denoised_image = reduce_noise(sharpened_image)
    
    # Apply multi-scale edge detection
    enhanced_image = multi_scale_edges(denoised_image)
    
    # Normalize the enhanced image to ensure pixel values are within [0, 1]
    enhanced_image_normalized = normalize_image(enhanced_image)
    
    return enhanced_image_normalized




# Function to compute luminance, contrast, structure, and perceptual loss
def compute_metrics(image1, image2, output_folder, image_name, perceptual_model):
    luminance1 = np.mean(image1)
    contrast1 = np.std(image1)
    luminance2 = np.mean(image2)
    contrast2 = np.std(image2)

    grad_x1 = cv2.Sobel(image1, cv2.CV_64F, 1, 0, ksize=3)
    grad_y1 = cv2.Sobel(image1, cv2.CV_64F, 0, 1, ksize=3)
    structure1 = np.mean(np.sqrt(grad_x1**2 + grad_y1**2))

    grad_x2 = cv2.Sobel(image2, cv2.CV_64F, 1, 0, ksize=3)
    grad_y2 = cv2.Sobel(image2, cv2.CV_64F, 0, 1, ksize=3)
    structure2 = np.mean(np.sqrt(grad_x2**2 + grad_y2**2))

    # Compute SSIM and save SSIM map
    ssim_score, ssim_map = ssim(image1, image2, full=True, data_range=1.0)
    plt.imshow(ssim_map, cmap='gray')
    plt.colorbar()
    plt.title(f"SSIM Map for {image_name}")
    plt.savefig(os.path.join(output_folder, f"SSIM_map_{image_name}.png"))
    plt.close()

    # Prepare images for LPIPS (convert to RGB and tensor)
    image1_tensor = torch.tensor(convert_to_rgb(image1)).permute(2, 0, 1).float().unsqueeze(0)
    image2_tensor = torch.tensor(convert_to_rgb(image2)).permute(2, 0, 1).float().unsqueeze(0)

    # Compute perceptual loss
    perceptual_loss = perceptual_model(image1_tensor, image2_tensor)

    return (luminance1, contrast1, structure1), (luminance2, contrast2, structure2), perceptual_loss.item()

# Initialize LPIPS perceptual loss model
perceptual_model = lpips.LPIPS(net='alex')  # Use AlexNet backbone

# Get list of common files
real_files = {os.path.splitext(f)[0]: f for f in os.listdir(real_folder)}
synthetic_files = {os.path.splitext(f)[0]: f for f in os.listdir(synthetic_folder)}
common_files = sorted(real_files.keys() & synthetic_files.keys())

# Initialize lists for metrics
real_luminance_values = []
real_contrast_values = []
real_structure_values = []

synthetic_luminance_values = []
synthetic_contrast_values = []
synthetic_structure_values = []

# Process each common image pair
# Process each common image pair
for name in common_files:
    real_img = cv2.imread(os.path.join(real_folder, real_files[name]), cv2.IMREAD_GRAYSCALE)
    synthetic_img = cv2.imread(os.path.join(synthetic_folder, synthetic_files[name]), cv2.IMREAD_GRAYSCALE)

    # Resize synthetic image if sizes do not match
    if real_img.shape != synthetic_img.shape:
        synthetic_img = cv2.resize(synthetic_img, (real_img.shape[1], real_img.shape[0]))

    # Normalize images to [0, 1]
    real_img_norm = real_img.astype(np.float64) / 255.0
    synthetic_img_norm = synthetic_img.astype(np.float64) / 255.0

    # **Enhance the synthetic image** by applying multiple techniques
    synthetic_img_enhanced = enhance_image(synthetic_img_norm)

    synthetic_img_equalized = equalize_histogram(synthetic_img_norm)

    # Compute MS-SSIM score
    ms_ssim_score = compute_ms_ssim(real_img_norm, synthetic_img_equalized)

    # Calculate metrics and append to lists
    (real_luminance, real_contrast, real_structure), (synthetic_luminance, synthetic_contrast, synthetic_structure), perceptual_loss = compute_metrics(
        real_img_norm, synthetic_img_equalized, output_folder, name, perceptual_model
    )

    real_luminance_values.append(real_luminance)
    real_contrast_values.append(real_contrast)
    real_structure_values.append(real_structure)

    synthetic_luminance_values.append(synthetic_luminance)
    synthetic_contrast_values.append(synthetic_contrast)
    synthetic_structure_values.append(synthetic_structure)

    print(f"{name}: MS-SSIM = {ms_ssim_score}, Perceptual Loss = {perceptual_loss}")

# Calculate average metrics for real and synthetic images
real_means = [np.mean(real_luminance_values), np.mean(real_contrast_values), np.mean(real_structure_values)]
synthetic_means = [np.mean(synthetic_luminance_values), np.mean(synthetic_contrast_values), np.mean(synthetic_structure_values)]

# Plot comparison bar chart
labels = ["Luminance", "Contrast", "Structure"]
x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(x - width / 2, real_means, width, label="Real", color="blue")
ax.bar(x + width / 2, synthetic_means, width, label="Synthetic", color="orange")
ax.set_ylabel("Average Value")
ax.set_title("Luminance, Contrast, and Structure Comparison")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.tight_layout()

chart_path = os.path.join(output_folder, "SSIM_comparison_chart.png")
plt.savefig(chart_path)
plt.show()

# Save results to CSV
csv_path = os.path.join(output_folder, "SSIM_Values_Real_vs_Synthetic.csv")
with open(csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Component", "Real Average", "Synthetic Average"])
    writer.writerow(["Luminance", real_means[0], synthetic_means[0]])
    writer.writerow(["Contrast", real_means[1], synthetic_means[1]])
    writer.writerow(["Structure", real_means[2], synthetic_means[2]])

print(f"Chart saved to: {chart_path}")
print(f"CSV saved to: {csv_path}")
