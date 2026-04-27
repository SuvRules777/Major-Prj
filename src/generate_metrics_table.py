"""Generate a Core Performance Metrics table image for presentation."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / "outputs" / "visualizations"
OUT.mkdir(parents=True, exist_ok=True)

# ── Table 1: Object Detection (YOLOv8) ──
det_cols = ["Metric", "Score", "Description"]
det_data = [
    ["Precision",           "93.8%",  "Of all predicted fish, 93.8% were correct"],
    ["Recall",              "94.4%",  "Of all actual fish, 94.4% were found"],
    ["mAP@50",              "94.9%",  "Overall detection accuracy (IoU > 0.50)"],
    ["mAP@50:95",           "72.0%",  "Strict detection accuracy (IoU 0.50–0.95)"],
    ["F1-Score",            "94.1%",  "Harmonic mean of Precision & Recall"],
    ["Dataset (Train)",     "1,156",  "80% of 1,445 annotated fish images"],
    ["Dataset (Val/Test)",  "289",    "10% validation + 10% test split"],
]

# ── Table 2: Biomass Estimation Accuracy ──
bio_cols = ["Metric", "Value", "Description"]
bio_data = [
    ["R² Score",        "0.9817",    "98.2% of weight variance explained"],
    ["MAPE",            "7.69%",     "Average prediction error"],
    ["Accuracy",        "92.31%",    "100% minus MAPE"],
    ["MAE",             "26.65 g",   "Mean absolute deviation"],
    ["RMSE",            "48.47 g",   "Root mean squared error"],
    ["Max Error",       "257.75 g",  "Worst single prediction"],
    ["Validation Set",  "129 / 7",   "Samples across 7 species"],
]

fig, axes = plt.subplots(2, 1, figsize=(11, 10.5))
fig.patch.set_facecolor("white")

# ── Title ──
fig.text(0.5, 0.98, "Core Performance Metrics",
         ha="center", va="top", fontsize=18, fontweight="bold", color="#1a1a2e")
fig.text(0.5, 0.95, "YOLOv8-based Fish Detection & Allometric Biomass Estimation",
         ha="center", va="top", fontsize=10.5, color="#666666")


def style_table(ax, title, col_labels, data, header_color, accent_col=1):
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold", color="#1a1a2e",
                 pad=14, loc="left")

    table = ax.table(cellText=data, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 1.65)

    n_cols = len(col_labels)
    n_rows = len(data)

    # Header
    for j in range(n_cols):
        cell = table[0, j]
        cell.set_facecolor(header_color)
        cell.set_text_props(color="white", fontweight="bold", fontsize=9.5)
        cell.set_edgecolor(header_color)

    # Body
    for i in range(n_rows):
        for j in range(n_cols):
            cell = table[i + 1, j]
            cell.set_edgecolor("#d0d7de")
            bg = "#f6f8fa" if i % 2 == 0 else "#ffffff"
            cell.set_facecolor(bg)

            if j == 0:
                cell.set_text_props(fontweight="bold", fontsize=9.5)
            elif j == accent_col:
                cell.set_text_props(fontweight="bold", fontsize=10, color=header_color)
            elif j == 2:
                cell.set_text_props(fontsize=8.8, color="#555555", style="italic")

    # Column widths
    for j in range(n_cols):
        for i in range(n_rows + 1):
            cell = table[i, j]
            if j == 0:
                cell.set_width(0.22)
            elif j == 1:
                cell.set_width(0.14)
            else:
                cell.set_width(0.50)


style_table(axes[0], "A. Object Detection Performance (YOLOv8n)",
            det_cols, det_data, "#1a5fb4")

style_table(axes[1], "B. Biomass Estimation Accuracy (Allometric Model)",
            bio_cols, bio_data, "#1a7d3e")

plt.subplots_adjust(top=0.88, bottom=0.03, hspace=0.38)

out_path = OUT / "core_metrics_table.png"
fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white", pad_inches=0.3)
print(f"Saved: {out_path}")
plt.close(fig)
