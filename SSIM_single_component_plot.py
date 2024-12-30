import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import csv
from skimage.metrics import structural_similarity as ssim

# Paths for real and synthetic images
real_folder = "Images\City_sunny\RD"
synthetic_folder = "Images\City_sunny\SD"
output_folder = "SSIM_outputs_City_sunny"
os.makedirs(output_folder, exist_ok=True)

# Helper function to compute luminance, contrast, and structure (real vs synthetic comparison)
def compute_metrics(image1, image2, output_folder, image_name):
    # Calculate luminance (mean intensity) and contrast (standard deviation) for the real image
    luminance = np.mean(image1)
    contrast = np.std(image1)
    
    # Compute SSIM between real and synthetic images to get structure
    ssim_score, ssim_map = ssim(image1, image2, full=True, data_range=1.0)
    structure = np.mean(ssim_map)  # Structure derived from SSIM map
    
    # Check for invalid structure values (NaN or Inf)
    if np.isnan(structure) or np.isinf(structure):
        print("Warning: structure value is NaN or Inf.")
        structure = 0
    
    print(f"SSIM Structure for {image_name}: {structure}")
    
    # Save SSIM map as image
    ssim_map_path = os.path.join(output_folder, f"SSIM_map_{image_name}.png")
    plt.imshow(ssim_map, cmap='gray')
    plt.colorbar()
    plt.title(f"SSIM Map for {image_name}")
    plt.savefig(ssim_map_path)
    plt.close()  # Close the plot to avoid memory overflow during large loops
    
    return luminance, contrast, structure

# Get common files
real_files = {os.path.splitext(f)[0]: f for f in os.listdir(real_folder)}
synthetic_files = {os.path.splitext(f)[0]: f for f in os.listdir(synthetic_folder)}
common_files = sorted(real_files.keys() & synthetic_files.keys())

# Initialize metric lists for real and synthetic images
real_luminance_values = []
real_contrast_values = []
real_structure_values = []

synthetic_luminance_values = []
synthetic_contrast_values = []
synthetic_structure_values = []

# Process images and calculate metrics
for name in common_files:
    real_img = cv2.imread(os.path.join(real_folder, real_files[name]), cv2.IMREAD_GRAYSCALE)
    synthetic_img = cv2.imread(os.path.join(synthetic_folder, synthetic_files[name]), cv2.IMREAD_GRAYSCALE)

    if real_img.shape != synthetic_img.shape:
        synthetic_img = cv2.resize(synthetic_img, (real_img.shape[1], real_img.shape[0]))

    # Normalize images to [0, 1] for SSIM and ensure float64 data type
    real_img_norm = real_img.astype(np.float64) / 255.0
    synthetic_img_norm = synthetic_img.astype(np.float64) / 255.0

    # Calculate metrics for real vs. synthetic images
    real_luminance, real_contrast, real_structure = compute_metrics(real_img_norm, synthetic_img_norm, output_folder, name)
    synthetic_luminance, synthetic_contrast, synthetic_structure = compute_metrics(synthetic_img_norm, real_img_norm, output_folder, name)
    
    # Append values to respective lists
    real_luminance_values.append(real_luminance)
    real_contrast_values.append(real_contrast)
    real_structure_values.append(real_structure)

    synthetic_luminance_values.append(synthetic_luminance)
    synthetic_contrast_values.append(synthetic_contrast)
    synthetic_structure_values.append(synthetic_structure)

# Compute the means for each component (Luminance, Contrast, Structure) for both real and synthetic
real_means = [np.mean(real_luminance_values), np.mean(real_contrast_values), np.mean(real_structure_values)]
synthetic_means = [np.mean(synthetic_luminance_values), np.mean(synthetic_contrast_values), np.mean(synthetic_structure_values)]

# Plotting: Luminance, Contrast, and Structure comparison between Real and Synthetic images
labels = ["Luminance", "Contrast", "Structure"]
x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 6))
bars1 = ax.bar(x - width / 2, real_means, width, label="Real", color="blue")
bars2 = ax.bar(x + width / 2, synthetic_means, width, label="Synthetic", color="orange")
ax.set_ylabel("Average Value")
ax.set_title("Luminance, Contrast, and Structure")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.tight_layout()

# Save the plot as an image
chart_path = os.path.join(output_folder, "SSIM_comparison_chart.png")
plt.savefig(chart_path)
print(f"SSIM comparison chart saved to: {chart_path}")

# Show the plot
plt.show()

# Save the average values to CSV
csv_path = os.path.join(output_folder, "SSIM_Values_Real_vs_Synthetic.csv")
with open(csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Component", "Real Average", "Synthetic Average"])
    writer.writerow(["Luminance", np.mean(real_luminance_values), np.mean(synthetic_luminance_values)])
    writer.writerow(["Contrast", np.mean(real_contrast_values), np.mean(synthetic_contrast_values)])
    writer.writerow(["Structure", np.mean(real_structure_values), np.mean(synthetic_structure_values)])

# Now you can print the path of the saved chart and CSV
print(f"SSIM comparison chart saved to: {chart_path}")
print(f"SSIM values saved to: {csv_path}")
