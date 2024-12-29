import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt

# Load images
synthetic_img = imageio.imread('Images/try/SD/Frame_900.png')
real_img = imageio.imread('Images/try/SD/Frame_900.png')  # Replace with your real image

# Resize images to the same size
synthetic_img = imageio.imread('Images/try/SD/Frame_900.png')  # Replace with your synthetic image
real_img = imageio.imread('Images/try/SD/Frame_900.png')  # Replace with your real image

# Calculate SSIM
synthetic_img = synthetic_img.astype(np.float32) / 255.0
real_img = real_img.astype(np.float32) / 255.0

def calculate_ssim(img1, img2):
    C1 = (0.01 ** 2)
    C2 = (0.03 ** 2)
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    sigma1 = np.sqrt(np.mean((img1 - mu1) ** 2))
    sigma2 = np.sqrt(np.mean((img2 - mu2) ** 2))
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))

    K = sigma12 + C2
    L = 1
    S = (2 * mu1 * mu2 + C1) / (mu1 ** 2 + mu2 ** 2 + C1)
    R = (2 * sigma1 * sigma2 + C2) / (sigma1 ** 2 + sigma2 ** 2 + C2)
    ssim = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2) / ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 ** 2 + sigma2 ** 2 + C2))
    return ssim

ssim_value = calculate_ssim(synthetic_img, real_img)

# Plot SSIM components
mse_value = np.mean((synthetic_img - real_img) ** 2)
print(mse_value)

plt.imshow(synthetic_img)
plt.show()
plt.imshow(real_img)
plt.show()