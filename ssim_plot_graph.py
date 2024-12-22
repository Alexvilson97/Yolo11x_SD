import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.util import random_noise
import matplotlib.pyplot as plt

# Function to calculate MSE
def mse(imageA, imageB):
    return np.mean((imageA - imageB) ** 2)

# Load the real image
real_image = cv2.imread(r"300_pass_UV.png", cv2.IMREAD_GRAYSCALE)

# Convert image to double for processing
real_image = real_image.astype(np.float64) / 255.0

# Apply Custom Poisson Noise
lambda_param = 25  # Define the lambda for Poisson noise
scaled_real_image = real_image * lambda_param  # Scale image

# Add Poisson noise manually
noisy_real_image = np.random.poisson(scaled_real_image)

# Normalize noisy image to the range [0, 1]
noisy_real_image = noisy_real_image / np.max(noisy_real_image)

# Apply Average Blur
average_filter_size = (11, 11)  # Define the filter size
blurred_real_image = cv2.blur(real_image, average_filter_size)

# Convert images back to uint8 for metric calculation
noisy_real_image_uint8 = (noisy_real_image * 255).astype(np.uint8)
blurred_real_image_uint8 = (blurred_real_image * 255).astype(np.uint8)
real_image_uint8 = (real_image * 255).astype(np.uint8)

# Calculate Metrics for Noisy Real Image
ssim_noisy, ssim_map_noisy = ssim(real_image_uint8, noisy_real_image_uint8, full=True)
psnr_noisy = cv2.PSNR(real_image_uint8, noisy_real_image_uint8)
mse_noisy = mse(real_image_uint8, noisy_real_image_uint8)

print('Noisy Real Image Metrics:')
print(f'SSIM Value: {ssim_noisy}')
print(f'PSNR Value: {psnr_noisy} dB')
print(f'MSE Value: {mse_noisy}')

# Calculate Metrics for Blurred Real Image
ssim_blurred, ssim_map_blurred = ssim(real_image_uint8, blurred_real_image_uint8, full=True)
psnr_blurred = cv2.PSNR(real_image_uint8, blurred_real_image_uint8)
mse_blurred = mse(real_image_uint8, blurred_real_image_uint8)

print('Blurred Real Image Metrics:')
print(f'SSIM Value: {ssim_blurred}')
print(f'PSNR Value: {psnr_blurred} dB')
print(f'MSE Value: {mse_blurred}')

# Plotting Results
fig, axs = plt.subplots(3, 3, figsize=(12, 12))

axs[0, 0].imshow(real_image, cmap='gray')
axs[0, 0].set_title('Original Real Image')

axs[0, 1].imshow(noisy_real_image_uint8, cmap='gray')
axs[0, 1].set_title(f'Noisy Real Image (Poisson)\nSSIM: {ssim_noisy:.4f}')

axs[0, 2].imshow(ssim_map_noisy, cmap='gray')
axs[0, 2].set_title('SSIM Map for Noisy Image')

axs[1, 0].imshow(blurred_real_image_uint8, cmap='gray')
axs[1, 0].set_title(f'Blurred Real Image (Average)\nSSIM: {ssim_blurred:.4f}')

axs[1, 1].imshow(ssim_map_blurred, cmap='gray')
axs[1, 1].set_title('SSIM Map for Blurred Image')

axs[1, 2].imshow(real_image, cmap='gray')
axs[1, 2].set_title('Original Real Image\nSSIM: 1 (Reference)')

# Hide empty subplots
for ax in axs.flat:
    ax.axis('off')

plt.tight_layout()
plt.show()
