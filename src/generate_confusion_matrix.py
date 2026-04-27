import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

OUT = Path("outputs/visualizations")
OUT.mkdir(parents=True, exist_ok=True)

# Data based on the YOLO val2 results:
# TP=139, FP=14
# FN=6,  TN=0 (Or NA for background-background)
matrix = np.array([[139, 14],
                   [6, 0]])

# Define labels
x_labels = ["Fish", "Background"]
y_labels = ["Fish", "Background"]

plt.figure(figsize=(8, 6), facecolor='white')

# Use a nice blue color scheme
cmap = sns.light_palette("#1a5fb4", as_cmap=True)

# Plot heatmap
ax = sns.heatmap(matrix, annot=False, cmap=cmap, cbar=True, 
                 xticklabels=x_labels, yticklabels=y_labels,
                 linewidths=1, linecolor='white')

# Manually add the text with custom formatting
texts = [
    ("139\n(True Positives)", 0, 0, "white"),
    ("14\n(False Positives)", 1, 0, "black"),
    ("6\n(False Negatives)", 0, 1, "black"),
    ("-", 1, 1, "black")
]

for text, x, y, color in texts:
    ax.text(x + 0.5, y + 0.5, text, 
            ha="center", va="center", color=color, 
            fontsize=12, fontweight="bold")

plt.title("Object Detection Confusion Matrix", fontsize=16, fontweight='bold', pad=20, color="#1a1a2e")
plt.xlabel("True Class", fontsize=13, labelpad=10, fontweight='bold', color="#2d2d4e")
plt.ylabel("Predicted Class", fontsize=13, labelpad=10, fontweight='bold', color="#2d2d4e")

# Tweak axes
plt.xticks(fontsize=12)
plt.yticks(fontsize=12, rotation=0)

out_path = OUT / "presentation_confusion_matrix.png"
plt.tight_layout()
plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
print(f"Saved: {out_path}")
