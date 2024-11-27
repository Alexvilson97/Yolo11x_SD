import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Paths for real and synthetic images
real_folder = r"Images\real_frames"
synthetic_folder = r"Images\sd_frames"
output_folder = r"SSIM_outputs"

# Create the output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Get image file names
real_images = sorted(os.listdir(real_folder))
synthetic_images = sorted(os.listdir(synthetic_folder))

# Ensure the number of images matches
if len(real_images) != len(synthetic_images):
    raise ValueError("The number of real and synthetic images must be the same!")

# Initialize lists to store component averages
real_luminance_values = []
real_contrast_values = []
real_structure_values = []

synthetic_luminance_values = []
synthetic_contrast_values = []
synthetic_structure_values = []

# SSIM constants
K1 = 0.01
K2 = 0.03
L = 255  # Dynamic range of pixel values (8-bit images)

# Process each pair of images
for idx, (real_file, synthetic_file) in enumerate(zip(real_images, synthetic_images)):
    # Load images in grayscale
    real_image = cv2.imread(os.path.join(real_folder, real_file), cv2.IMREAD_GRAYSCALE)
    synthetic_image = cv2.imread(os.path.join(synthetic_folder, synthetic_file), cv2.IMREAD_GRAYSCALE)

    # Ensure images are loaded correctly
    if real_image is None or synthetic_image is None:
        print(f"Error loading {real_file} or {synthetic_file}. Skipping...")
        continue

    # Resize synthetic image to match real image if necessary
    if real_image.shape != synthetic_image.shape:
        synthetic_image = cv2.resize(synthetic_image, (real_image.shape[1], real_image.shape[0]))

    # Calculate means and variances
    mu_x = cv2.GaussianBlur(real_image, (11, 11), 1.5)
    mu_y = cv2.GaussianBlur(synthetic_image, (11, 11), 1.5)
    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x_sq = cv2.GaussianBlur(real_image ** 2, (11, 11), 1.5) - mu_x_sq
    sigma_y_sq = cv2.GaussianBlur(synthetic_image ** 2, (11, 11), 1.5) - mu_y_sq
    sigma_xy = cv2.GaussianBlur(real_image * synthetic_image, (11, 11), 1.5) - mu_xy

    # SSIM components
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    # Calculate components
    real_luminance = (2 * mu_x + C1) / (mu_x_sq + C1)
    synthetic_luminance = (2 * mu_y + C1) / (mu_y_sq + C1)

    real_contrast = (2 * np.sqrt(sigma_x_sq) + C2) / (sigma_x_sq + C2)
    synthetic_contrast = (2 * np.sqrt(sigma_y_sq) + C2) / (sigma_y_sq + C2)

    real_structure = sigma_xy / (np.sqrt(sigma_x_sq) * np.sqrt(sigma_x_sq) + C2)
    synthetic_structure = sigma_xy / (np.sqrt(sigma_y_sq) * np.sqrt(sigma_y_sq) + C2)

    # Compute average values for each component
    real_luminance_values.append(real_luminance.mean())
    synthetic_luminance_values.append(synthetic_luminance.mean())
    real_contrast_values.append(real_contrast.mean())
    synthetic_contrast_values.append(synthetic_contrast.mean())
    real_structure_values.append(real_structure.mean())
    synthetic_structure_values.append(synthetic_structure.mean())

    print(f"Processed: {real_file} and {synthetic_file}")

    # Plot bar chart for current frame
    labels = ["Luminance", "Contrast", "Structure"]
    real_values = [real_luminance.mean(), real_contrast.mean(), real_structure.mean()]
    synthetic_values = [synthetic_luminance.mean(), synthetic_contrast.mean(), synthetic_structure.mean()]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, real_values, width, label="Real", color="blue")
    bars2 = ax.bar(x + width/2, synthetic_values, width, label="Synthetic", color="orange")

    # Add labels, title, and legend
    ax.set_ylabel("Average Value")
    ax.set_title(f"SSIM Component Comparison: Frame {idx + 1}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Add value labels to the bars
    for bars in [bars1, bars2]:
        ax.bar_label(bars, fmt='%.2f')

    plt.tight_layout()
    frame_chart_path = os.path.join(output_folder, f"SSIM_Component_Comparison_Frame_{idx + 1}.png")
    plt.savefig(frame_chart_path)
    plt.close()

# After processing all frames, plot overall averages
labels = ["Luminance", "Contrast", "Structure"]
real_values = [
    np.mean(real_luminance_values),
    np.mean(real_contrast_values),
    np.mean(real_structure_values),
]
synthetic_values = [
    np.mean(synthetic_luminance_values),
    np.mean(synthetic_contrast_values),
    np.mean(synthetic_structure_values),
]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, real_values, width, label="Real", color="blue")
bars2 = ax.bar(x + width/2, synthetic_values, width, label="Synthetic", color="orange")

# Add labels, title, and legend
ax.set_ylabel("Average Value")
ax.set_title("Overall SSIM Component Comparison")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Add value labels to the bars
for bars in [bars1, bars2]:
    ax.bar_label(bars, fmt='%.2f')

plt.tight_layout()
overall_chart_path = os.path.join(output_folder, "Overall_SSIM_Component_Comparison.png")
plt.savefig(overall_chart_path)
plt.show()

# Compute and print overall averages
overall_real_luminance = np.mean(real_luminance_values)
overall_synthetic_luminance = np.mean(synthetic_luminance_values)
overall_real_contrast = np.mean(real_contrast_values)
overall_synthetic_contrast = np.mean(synthetic_contrast_values)
overall_real_structure = np.mean(real_structure_values)
overall_synthetic_structure = np.mean(synthetic_structure_values)

print("\n=== Average Values ===")
print(f"Real Luminance: {overall_real_luminance:.4f}")
print(f"Synthetic Luminance: {overall_synthetic_luminance:.4f}")
print(f"Real Contrast: {overall_real_contrast:.4f}")
print(f"Synthetic Contrast: {overall_synthetic_contrast:.4f}")
print(f"Real Structure: {overall_real_structure:.4f}")
print(f"Synthetic Structure: {overall_synthetic_structure:.4f}")

# Save the averages to a text file
average_values_path = os.path.join(output_folder, "Average_SSIM_Component_Values.txt")
with open(average_values_path, "w") as f:
    f.write("=== Average SSIM Component Values ===\n")
    f.write(f"Real Luminance: {overall_real_luminance:.4f}\n")
    f.write(f"Synthetic Luminance: {overall_synthetic_luminance:.4f}\n")
    f.write(f"Real Contrast: {overall_real_contrast:.4f}\n")
    f.write(f"Synthetic Contrast: {overall_synthetic_contrast:.4f}\n")
    f.write(f"Real Structure: {overall_real_structure:.4f}\n")
    f.write(f"Synthetic Structure: {overall_synthetic_structure:.4f}\n")

print(f"Averages saved to: {average_values_path}")
print("All charts saved and process completed.")
