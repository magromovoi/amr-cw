import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Filepaths to your saved subfigure images
image_files = [
    "images/bbcsport_cricket_gcn_conv_False.png",
    "images/bbcsport_football_gcn_conv_False.png",
    "images/bbcsport_tennis_gcn_conv_False.png"
]

# Create a 2-row, 2-column grid with per-subplot size = 6×6
fig, axes = plt.subplots(2, 2, figsize=(12, 12))  # 2x2 grid with same subfigure size as 3x2

axes = axes.flatten()

# Load and display each image
for i, ax in enumerate(axes):
    if i < len(image_files):
        img = mpimg.imread(image_files[i])
        ax.imshow(img)
        ax.set_aspect('auto')
    ax.axis('off')

# Minimize spacing
plt.subplots_adjust(hspace=0.05, wspace=0.05)

# Save the final figure
plt.savefig("images/parak7.png", dpi=600, bbox_inches='tight', pad_inches=0.1)
plt.close()
