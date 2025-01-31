import matplotlib.pyplot as plt

# Function to plot the bar chart with values on top of the bars
def plot_ssim_components(luminance, contrast, structure):
    components = ['Luminance', 'Contrast', 'Structure']
    values = [luminance, contrast, structure]

    bars = plt.bar(components, values, color=['skyblue', 'lightgreen', 'salmon'])
    plt.title('SSIM Components Rainy Frame_540', fontsize=16)  # Increase title font size
    plt.xlabel('Components', fontsize=14)      # Increase x-axis font size
    plt.ylabel('Component Contribution', fontsize=14)  # Increase y-axis font size
    plt.ylim(0, 1)

    # Add value labels on top of the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.02, f'{height:.4f}', 
                 ha='center', va='bottom', fontsize=10)

    plt.xticks(fontsize=12)  # Increase x-axis tick font size
    plt.yticks(fontsize=12)  # Increase y-axis tick font size
    plt.show()

# Provide your values for luminance, contrast, and structure
luminance = 0.6893 # Example value
contrast = 0.9245   # Example value
structure = 0.6270  # Example value

# Plot the bar chart
plot_ssim_components(luminance, contrast, structure)
