"""Generate a clean comparison table image: Actual Weight vs Predicted Weight."""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "fish_data_with_predictions.csv")
df = df[df["Weight"] > 0].copy()

# Pick a representative sample: 3 from each species (smallest, median, largest by weight)
rows = []
for species in sorted(df["Species"].unique()):
    sub = df[df["Species"] == species].sort_values("Weight").reset_index(drop=True)
    if len(sub) >= 3:
        idx = [0, len(sub) // 2, len(sub) - 1]
    else:
        idx = list(range(len(sub)))
    rows.append(sub.iloc[idx])

sample = pd.concat(rows).reset_index(drop=True)

# Build table data
table_data = []
for _, r in sample.iterrows():
    err = abs(r["Weight"] - r["Predicted_Weight"])
    pct = err / r["Weight"] * 100
    acc = 100 - pct
    table_data.append([
        r["Species"],
        f'{r["Length"]:.1f}',
        f'{r["Weight"]:.1f}',
        f'{r["Predicted_Weight"]:.1f}',
        f'{err:.1f}',
        f'{pct:.1f}%',
        f'{acc:.1f}%',
    ])

col_labels = ["Species", "Length\n(cm)", "Actual\nWeight (g)", "Predicted\nWeight (g)",
              "Abs. Error\n(g)", "Error\n(%)", "Accuracy\n(%)"]

n_rows = len(table_data)
fig_height = 1.2 + n_rows * 0.42
fig, ax = plt.subplots(figsize=(12, fig_height))
ax.axis("off")

# Title
fig.text(0.5, 0.97, "Comparative Validation: Ground Truth vs Predicted Fish Weight",
         ha="center", va="top", fontsize=15, fontweight="bold", color="#1a1a2e")
fig.text(0.5, 0.94, f"Representative samples from {df['Species'].nunique()} species  |  "
         f"Overall R\u00b2 = 0.9817  |  MAPE = 7.69%  |  Accuracy = 92.31%",
         ha="center", va="top", fontsize=9.5, color="#555555")

table = ax.table(cellText=table_data, colLabels=col_labels, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(9.5)
table.scale(1, 1.55)

# Style header
for j in range(len(col_labels)):
    cell = table[0, j]
    cell.set_facecolor("#1a1a2e")
    cell.set_text_props(color="white", fontweight="bold", fontsize=9)
    cell.set_edgecolor("#2d2d4e")

# Style body rows
species_colors = {
    "Bream": "#e8f0fe", "Parkki": "#f3e8ff", "Perch": "#e6f9ed",
    "Pike": "#e6faf7", "Roach": "#fff8e1", "Smelt": "#fce4ec", "Whitefish": "#f0f0f0"
}

for i in range(n_rows):
    species = table_data[i][0]
    bg = species_colors.get(species, "#ffffff")
    alt_bg = "#ffffff" if i % 2 == 0 else "#f7f9fc"
    # Use species color for first col, alternating for rest
    for j in range(len(col_labels)):
        cell = table[i + 1, j]
        cell.set_edgecolor("#d0d7de")
        if j == 0:
            cell.set_facecolor(bg)
            cell.set_text_props(fontweight="bold", fontsize=9)
        else:
            cell.set_facecolor(alt_bg)

        # Color the accuracy column
        if j == 6:
            val = float(table_data[i][6].replace("%", ""))
            if val >= 95:
                cell.set_text_props(color="#16a34a", fontweight="bold")
            elif val >= 85:
                cell.set_text_props(color="#ca8a04", fontweight="bold")
            else:
                cell.set_text_props(color="#dc2626", fontweight="bold")

        # Color the error % column
        if j == 5:
            val = float(table_data[i][5].replace("%", ""))
            if val <= 5:
                cell.set_text_props(color="#16a34a")
            elif val <= 15:
                cell.set_text_props(color="#ca8a04")
            else:
                cell.set_text_props(color="#dc2626")

out_path = PROJECT_ROOT / "outputs" / "visualizations" / "comparison_table.png"
fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white", pad_inches=0.3)
print(f"Saved: {out_path}")
plt.close(fig)
