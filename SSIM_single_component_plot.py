import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import warnings
from mpl_toolkits.axes_grid1 import make_axes_locatable

warnings.filterwarnings("ignore")

# Paths for real and synthetic images
real_folder = "Images/Rain_scenario/RD"
synthetic_folder = "Images/Rain_scenario/SD"
output_folder = "SSIM/SSIM_Rain"
os.makedirs(output_folder, exist_ok=True)

# Ensure both folders contain the same number of images with matching names
real_images = sorted(os.listdir(real_folder))
synthetic_images = sorted(os.listdir(synthetic_folder))

if len(real_images) != len(synthetic_images):
    print("Error: Mismatch in the number of images in the folders.")
    exit()

# Loop through the images and calculate SSIM map
for real_img_name, synth_img_name in zip(real_images, synthetic_images):
    if real_img_name != synth_img_name:
        print(f"Error: File name mismatch - {real_img_name} and {synth_img_name}")
        exit()

    # Read the images
    real_img_path = os.path.join(real_folder, real_img_name)
    synth_img_path = os.path.join(synthetic_folder, synth_img_name)

    real_img = cv2.imread(real_img_path, cv2.IMREAD_GRAYSCALE)
    synth_img = cv2.imread(synth_img_path, cv2.IMREAD_GRAYSCALE)

    # Ensure images are loaded correctly
    if real_img is None or synth_img is None:
        print(f"Error: Could not read {real_img_name} or {synth_img_name}")
        continue

    # Resize images to the same dimensions
    height, width = real_img.shape
    synth_img = cv2.resize(synth_img, (width, height))

    # Compute SSIM and SSIM map
    ssim_score, ssim_map = ssim(real_img, synth_img, full=True, data_range=synth_img.max() - synth_img.min())

    # Create a figure for the SSIM map
    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot SSIM map
    im = ax.imshow(ssim_map, cmap='PuBuGn_r')
    ax.set_title(f"SSIM Map for {real_img_name}")
    ax.axis('off')

    # Create a colorbar that matches the height of the image
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)

    # Print SSIM value on the map
    ax.text(0.5, -0.1, f"SSIM: {ssim_score:.4f}", ha='center', va='center', transform=ax.transAxes, fontsize=12, color='red')

    # Save the figure
    ssim_map_path = os.path.join(output_folder, f"SSIM_map_{real_img_name}.png")
    plt.tight_layout()
    plt.savefig(ssim_map_path)
    plt.close()

    print(f"SSIM map saved to: {ssim_map_path}")