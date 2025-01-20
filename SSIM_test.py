import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import csv
from skimage.metrics import structural_similarity as ssim

# Paths for real and synthetic images
real_folder = "Images/Rain_scenario/RD"
synthetic_folder = "Images/Rain_scenario/SD"
output_folder = "SSIM_outputs_Rain"
os.makedirs(output_folder, exist_ok=True)

# Helper function to compute metrics
def compute_metrics(image1, image2, output_folder, image_name):
    # Calculate luminance (mean intensity) and contrast (standard deviation)
    luminance1, contrast1 = np.mean(image1), np.std(image1)
    luminance2, contrast2 = np.mean(image2), np.std(image2)

    # Compute gradient magnitude for structure
    grad_x1, grad_y1 = np.gradient(image1)
    grad_x2, grad_y2 = np.gradient(image2)
    structure1 = np.mean(np.sqrt(grad_x1**2 + grad_y1**2))
    structure2 = np.mean(np.sqrt(grad_x2**2 + grad_y2**2))

    # Compute SSIM and save SSIM map
    ssim_score, ssim_map = ssim(image1, image2, full=True, data_range=1.0)
    ssim_map_path = os.path.join(output_folder, f"SSIM_map_{image_name}.png")
    plt.figure(figsize=(6, 5))
    plt.imshow(ssim_map, cmap='viridis')
    plt.colorbar(label="SSIM Value")
    plt.title(f"SSIM Map - {image_name}")
    plt.tight_layout()
    plt.savefig(ssim_map_path)
    plt.close()

    print(f"Metrics for {image_name} -> Luminance1: {luminance1}, Contrast1: {contrast1}, Structure1: {structure1}")
    print(f"Metrics for {image_name} -> Luminance2: {luminance2}, Contrast2: {contrast2}, Structure2: {structure2}")
    return (luminance1, contrast1, structure1), (luminance2, contrast2, structure2), ssim_score

# Get common files
real_files = {os.path.splitext(f)[0]: f for f in os.listdir(real_folder)}
synthetic_files = {os.path.splitext(f)[0]: f for f in os.listdir(synthetic_folder)}
common_files = sorted(real_files.keys() & synthetic_files.keys())

# Initialize metric lists
metrics = {"Real": [], "Synthetic": [], "SSIM": []}

# Process images
for name in common_files:
    real_img = cv2.imread(os.path.join(real_folder, real_files[name]), cv2.IMREAD_GRAYSCALE)
    synthetic_img = cv2.imread(os.path.join(synthetic_folder, synthetic_files[name]), cv2.IMREAD_GRAYSCALE)

    if real_img.shape != synthetic_img.shape:
        synthetic_img = cv2.resize(synthetic_img, (real_img.shape[1], real_img.shape[0]))

    # Normalize to [0, 1]
    real_img_norm = real_img.astype(np.float64) / 255.0
    synthetic_img_norm = synthetic_img.astype(np.float64) / 255.0

    # Compute metrics
    real_metrics, synthetic_metrics, ssim_score = compute_metrics(
        real_img_norm, synthetic_img_norm, output_folder, name
    )
    metrics["Real"].append(real_metrics)
    metrics["Synthetic"].append(synthetic_metrics)
    metrics["SSIM"].append(ssim_score)

# Compute averages
real_means = np.mean(metrics["Real"], axis=0)
synthetic_means = np.mean(metrics["Synthetic"], axis=0)
avg_ssim = np.mean(metrics["SSIM"])

# Plot comparison
labels = ["Luminance", "Contrast", "Structure"]
x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(8, 6))
plt.bar(x - width / 2, real_means, width, label="Real", color="blue")
plt.bar(x + width / 2, synthetic_means, width, label="Synthetic", color="orange")
plt.ylabel("Average Value")
plt.title("Average Metrics Comparison")
plt.xticks(x, labels)
plt.legend()
plt.tight_layout()

# Save and display
chart_path = os.path.join(output_folder, "Metrics_Comparison.png")
plt.savefig(chart_path)
plt.show()

# Save metrics to CSV
csv_path = os.path.join(output_folder, "Metrics_Real_vs_Synthetic.csv")
with open(csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Component", "Real Average", "Synthetic Average"])
    for i, label in enumerate(labels):
        writer.writerow([label, real_means[i], synthetic_means[i]])
    writer.writerow(["SSIM", avg_ssim, ""])

print(f"Comparison chart saved at: {chart_path}")
print(f"Metrics saved at: {csv_path}")
