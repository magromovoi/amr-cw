import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Filepaths to your saved subfigure images
image_files = [
    "images/ten_newsgroups_entertainment_gcn_conv_False.png",
    "images/ten_newsgroups_graphics_gcn_conv_False.png",
    "images/ten_newsgroups_historical_gcn_conv_False.png",
    "images/ten_newsgroups_politics_gcn_conv_False.png",
    "images/ten_newsgroups_sport_gcn_conv_False.png"
]

# Create a 3-row, 2-column grid
fig, axes = plt.subplots(3, 2, figsize=(12, 18))  # Adjust size as needed
axes = axes.flatten()

# Load and display each image
for i, ax in enumerate(axes):
    if i < len(image_files):
        img = mpimg.imread(image_files[i])
        ax.imshow(img)
        ax.set_aspect('auto')
    ax.axis('off')  # Hide axes for all

# Save the final grid image
plt.subplots_adjust(hspace=0.05, wspace=0.05)
plt.savefig("images/parak5.png", dpi=600, bbox_inches='tight', pad_inches=0.1)
plt.close()
