import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray, rgba2rgb
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

# Function to preprocess images (handle RGBA, RGB, or grayscale)
def preprocess_image(image):
    if image.shape[-1] == 4:  # RGBA image
        image = rgba2rgb(image)  # Convert RGBA to RGB
    if image.ndim == 3:  # Convert RGB to grayscale
        image = rgb2gray(image)
    return image

def calculate_ssim_components(image1, image2):
    ssim_index, ssim_map = ssim(
        image1,
        image2,
        full=True,
        data_range=image1.max() - image1.min(),
        gaussian_weights=True,
        use_sample_covariance=False,
        sigma=1.5,
    )
    luminance = np.mean(ssim_map)  # Overall SSIM map
    contrast = np.std(ssim_map)    # Standard deviation as a proxy for contrast
    structure = np.mean(ssim_map)  # Mean as a proxy for structure
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
    image1_path = r'Images\fog_Highway\SSIM_single_component_study\RD\Frame_1044.png'
    image2_path = r'Images\fog_Highway\SSIM_single_component_study\SD\Frame_1044.png'

    # Load images
    image1 = imread(image1_path) / 255.0
    image2 = imread(image2_path) / 255.0

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
