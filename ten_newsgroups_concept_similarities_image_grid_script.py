import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Filepaths to your saved heatmap subfigure images
image_files = [
    "images/ten_newsgroups_gcn_conv_False_weighted_frequent_subgraphs_black-box.png",
    "images/ten_newsgroups_gcn_conv_False_weighted_frequent_subgraphs_white-box.png",
    "images/ten_newsgroups_gcn_conv_False_weighted_pattern_concepts_black-box.png",
    "images/ten_newsgroups_gcn_conv_False_weighted_pattern_concepts_white-box.png",
    "images/ten_newsgroups_gcn_conv_False_weighted_filtered_equivalence_classes_black-box.png",
    "images/ten_newsgroups_gcn_conv_False_weighted_filtered_equivalence_classes_white-box.png",
    "images/ten_newsgroups_gcn_conv_False_weighted_closed_subgraphs_black-box.png",
    "images/ten_newsgroups_gcn_conv_False_weighted_closed_subgraphs_white-box.png"]

# Create a 4-row, 2-column grid
fig, axes = plt.subplots(4, 2, figsize=(14, 20))  # Adjusted for portrait layout
axes = axes.flatten()

# Load and display each image
for i, ax in enumerate(axes):
    if i < len(image_files):
        img = mpimg.imread(image_files[i])
        ax.imshow(img)
        ax.set_aspect('auto')
    ax.axis('off')  # Hide all axes

# Adjust layout and save
plt.subplots_adjust(hspace=0.05, wspace=0.05)
plt.savefig("images/parak6.png", dpi=600, bbox_inches='tight', pad_inches=0.1)
plt.close()
