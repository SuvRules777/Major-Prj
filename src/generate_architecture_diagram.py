"""Generate a System Architecture Block Diagram using Matplotlib."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / "outputs" / "visualizations"
OUT.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(11, 7))
fig.patch.set_facecolor('white')
ax.axis('off')

# Title
fig.text(0.5, 0.98, "System Architecture: Real-time Fish Biomass Estimation",
         ha="center", va="top", fontsize=18, fontweight="bold", color="#1a1a2e")

def draw_box(ax, x, y, width, height, text, bg_color, text_color="white"):
    rect = patches.FancyBboxPatch((x, y), width, height,
                                  boxstyle="round,pad=0.1,rounding_size=0.15",
                                  linewidth=1, edgecolor="#2d2d4e", facecolor=bg_color)
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, text, ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=text_color)

def draw_arrow(ax, start_pos, end_pos, text=None, connectionstyle="arc3"):
    ax.annotate("", xy=end_pos, xytext=start_pos,
                arrowprops=dict(arrowstyle="->", lw=2, color="#555555", connectionstyle=connectionstyle))
    if text:
        cx = (start_pos[0] + end_pos[0]) / 2
        cy = (start_pos[1] + end_pos[1]) / 2 + 0.15
        ax.text(cx, cy, text, ha="center", va="bottom", fontsize=9.5, color="#555")

# Coordinate system
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)

# Group: Client Side
client_bg = "#f3f4f6"
rect = patches.FancyBboxPatch((0.2, 2.5), 3.0, 4.5, boxstyle="round,pad=0.1",
                              linewidth=1.5, edgecolor="#d1d5db", facecolor=client_bg, linestyle="--")
ax.add_patch(rect)
ax.text(1.7, 7.2, "Client UI (Phone / Browser)", ha="center", va="center", fontsize=11.5, fontweight="bold", color="#374151")

draw_box(ax, 0.6, 5.5, 2.2, 1.2, "Image Upload\n& Parameters\n(Conf., px/cm)", "#1a5fb4")
draw_box(ax, 0.6, 3.0, 2.2, 1.2, "Results Panel\n(Biomass & BBoxes)", "#1a7d3e")

# Group: API Server
serve_bg = "#eef2ff"
rect = patches.FancyBboxPatch((4.0, 1.0), 7.8, 6.0, boxstyle="round,pad=0.1",
                              linewidth=1.5, edgecolor="#c7d2fe", facecolor=serve_bg, linestyle="--")
ax.add_patch(rect)
ax.text(7.9, 7.2, "FastAPI Application Server", ha="center", va="center", fontsize=11.5, fontweight="bold", color="#312e81")

draw_box(ax, 4.5, 5.5, 2.8, 1.2, "API Gateway\n(POST /predict-biomass)", "#4f46e5")
draw_box(ax, 8.5, 5.5, 2.8, 1.2, "Computer Vision\n(YOLOv8n Inference)", "#ea580c")
draw_box(ax, 8.5, 3.25, 2.8, 1.2, "Geometry Extractor\n(Length, Area, Pixels)", "#0891b2")
draw_box(ax, 8.5, 1.0, 2.8, 1.2, "Allometric Model\n(W = a × L^b)", "#ca8a04")

# Flow Arrows
# Client -> API
draw_arrow(ax, (2.8, 6.1), (4.5, 6.1), "Image +\nForm Data")

# API -> YOLO
draw_arrow(ax, (7.3, 6.1), (8.5, 6.1), "Decoded\nArray")

# YOLO -> Geometry
draw_arrow(ax, (9.9, 5.5), (9.9, 4.45), "BBox & Mask")

# Geometry -> Allometric
draw_arrow(ax, (9.9, 3.25), (9.9, 2.2), "Length (cm)")

# Allometric -> API Gateway (Return result)
ax.annotate("", xy=(5.9, 5.5), xytext=(5.9, 1.6),
            arrowprops=dict(arrowstyle="->", lw=2, color="#555555", connectionstyle="angle,angleA=180,angleB=90,rad=0"))
ax.annotate("", xy=(5.9, 1.6), xytext=(8.5, 1.6),
            arrowprops=dict(arrowstyle="-", lw=2, color="#555555"))
ax.text(7.2, 1.75, "Predicted\nWeight (g)", ha="center", va="bottom", fontsize=9.5, color="#555")

# API Gateway -> Client (JSON Response)
# Path: API Gateway left-lower (4.5, 5.8) -> Left to 3.65 -> Down to 3.6 -> Left to Results (2.8, 3.6)
ax.annotate("", xy=(3.65, 5.8), xytext=(4.5, 5.8), arrowprops=dict(arrowstyle="-", lw=2, color="#555555"))
ax.annotate("", xy=(3.65, 3.6), xytext=(3.65, 5.8), arrowprops=dict(arrowstyle="-", lw=2, color="#555555"))
ax.annotate("", xy=(2.8, 3.6), xytext=(3.65, 3.6), arrowprops=dict(arrowstyle="->", lw=2, color="#555555"))
ax.text(3.65, 4.7, "JSON\nResult", ha="center", va="center", fontsize=9.5, color="#555", 
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="white"))

out_path = OUT / "system_architecture.png"
fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white", pad_inches=0.3)
print(f"Saved: {out_path}")
plt.close(fig)
