import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
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

# Process each pair of images
for real_file, synthetic_file in zip(real_images, synthetic_images):
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

    # Compute SSIM and the SSIM map for the entire image
    ssim_value, ssim_map = ssim(real_image, synthetic_image, full=True)

    # Plot the compared images and SSIM map
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Display the real image
    axs[0].imshow(real_image, cmap='gray')
    axs[0].set_title('Real Image')
    axs[0].axis('off')

    # Display the synthetic image
    axs[1].imshow(synthetic_image, cmap='gray')
    axs[1].set_title('Synthetic Image')
    axs[1].axis('off')

    # Display the SSIM map
    ssim_plot = axs[2].imshow(ssim_map, cmap='viridis')
    axs[2].set_title(f'SSIM Map\nSSIM: {ssim_value:.4f}')
    axs[2].axis('off')

    # Add colorbar to SSIM map
    plt.colorbar(ssim_plot, ax=axs[2], fraction=0.046, pad=0.04)

    # Save the comparison plot
    output_path = os.path.join(output_folder, f"Comparison_{os.path.splitext(real_file)[0]}.png")
    plt.savefig(output_path)
    plt.close()

    # Print progress
    print(f"Processed: {real_file} and {synthetic_file} - SSIM: {ssim_value:.4f}")

print("SSIM processing complete. All results saved to the output folder.")

