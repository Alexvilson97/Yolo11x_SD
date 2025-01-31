import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray, rgba2rgb
from skimage.util import img_as_float
import matplotlib.pyplot as plt

# Constants for SSIM calculation
K1 = 0.01
K2 = 0.03
L = 1  # Dynamic range of the pixel values (assuming normalized [0, 1] images)

# Function to preprocess images (handle RGBA, RGB, or grayscale)
def preprocess_image(image):
    if image.shape[-1] == 4:  # RGBA image
        image = rgba2rgb(image)  # Convert RGBA to RGB
    if image.ndim == 3:  # Convert RGB to grayscale
        image = rgb2gray(image)
    return image

# Function to calculate luminance, contrast, and structure components
def calculate_ssim_components(image1, image2):
    # Compute means
    mu1 = image1.mean()
    mu2 = image2.mean()

    # Compute variances and covariance
    sigma1_sq = ((image1 - mu1) ** 2).mean()
    sigma2_sq = ((image2 - mu2) ** 2).mean()
    sigma12 = ((image1 - mu1) * (image2 - mu2)).mean()

    # Compute constants
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    # Luminance component
    luminance = (2 * mu1 * mu2 + C1) / (mu1 ** 2 + mu2 ** 2 + C1)

    # Contrast component
    contrast = (2 * np.sqrt(sigma1_sq) * np.sqrt(sigma2_sq) + C2) / (sigma1_sq + sigma2_sq + C2)

    # Structure component
    structure = (sigma12 + C2 / 2) / (np.sqrt(sigma1_sq) * np.sqrt(sigma2_sq) + C2 / 2)

    return luminance, contrast, structure

# Function to plot the bar chart
def plot_ssim_components(luminance, contrast, structure):
    components = ['Luminance', 'Contrast', 'Structure']
    values = [luminance, contrast, structure]

    plt.bar(components, values, color=['skyblue', 'lightgreen', 'salmon'])
    plt.title('SSIM Components')
    plt.ylabel('Component Contribution')
    plt.ylim(0, 1)
    plt.show()

# Main code
try:
    # Update paths to your actual image files
    image1_path = r'Images\fog_Highway\SSIM_single_component_study\RD\Frame_540.png'
    image2_path = r'Images\fog_Highway\SSIM_single_component_study\SD\Frame_540.png'

    # Load images
    image1 = img_as_float(imread(image1_path))
    image2 = img_as_float(imread(image2_path))

    # Preprocess images
    image1 = preprocess_image(image1)
    image2 = preprocess_image(image2)

    # Calculate SSIM components
    luminance, contrast, structure = calculate_ssim_components(image1, image2)
    print(f"Luminance: {luminance:.4f}, Contrast: {contrast:.4f}, Structure: {structure:.4f}")

    # Plot the bar chart
    plot_ssim_components(luminance, contrast, structure)

except FileNotFoundError as e:
    print(f"File not found: {e.filename}")
except Exception as e:
    print(f"An error occurred: {e}")
