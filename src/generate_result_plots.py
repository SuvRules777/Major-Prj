"""
Generate publication-ready diagnostic plots for the research paper.

Outputs (saved to outputs/visualizations/):
  1. r2_scatter_plot.png       — Predicted vs Actual with R² and regression line
  2. mae_by_species.png        — MAE per species (bar chart)
  3. rmse_by_species.png       — RMSE per species (bar chart)
  4. bland_altman_plot.png     — Bland–Altman (Difference vs Mean, ±1.96 SD)

Usage:
    python src/generate_result_plots.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VISUALS_DIR = PROJECT_ROOT / "outputs" / "visualizations"
VISUALS_DIR.mkdir(parents=True, exist_ok=True)

# ── Colour palette (muted, print-friendly) ──
SPECIES_COLORS = {
    "Bream":     "#4C72B0",
    "Parkki":    "#8172B3",
    "Perch":     "#55A868",
    "Pike":      "#23A5A5",
    "Roach":     "#C4A035",
    "Smelt":     "#C44E52",
    "Whitefish": "#8C8C8C",
}

def species_color(name: str) -> str:
    return SPECIES_COLORS.get(name, "#666666")


def load_data() -> pd.DataFrame:
    csv_path = PROJECT_ROOT / "data" / "processed" / "fish_data_with_predictions.csv"
    if not csv_path.exists():
        print(f"[FAIL] File not found: {csv_path}")
        sys.exit(1)
    df = pd.read_csv(csv_path)
    df = df[df["Weight"] > 0].copy()
    df["Absolute_Error"] = (df["Weight"] - df["Predicted_Weight"]).abs()
    df["Percentage_Error"] = df["Absolute_Error"] / df["Weight"] * 100
    return df


# ────────────────────────────────────────────
#  1. Predicted vs Actual  (R² Scatter)
# ────────────────────────────────────────────
def plot_r2_scatter(df: pd.DataFrame) -> None:
    actual = df["Weight"].values
    predicted = df["Predicted_Weight"].values

    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - ss_res / ss_tot

    fig, ax = plt.subplots(figsize=(6, 6))

    for species in sorted(df["Species"].unique()):
        sub = df[df["Species"] == species]
        ax.scatter(sub["Weight"], sub["Predicted_Weight"],
                   label=species, color=species_color(species),
                   s=40, alpha=0.8, edgecolors="white", linewidths=0.4)

    # Perfect prediction line
    lim = max(actual.max(), predicted.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", linewidth=1, alpha=0.5, label="Perfect prediction")

    ax.set_xlabel("Actual Weight (g)", fontsize=11)
    ax.set_ylabel("Predicted Weight (g)", fontsize=11)
    ax.set_title("Predicted vs Actual Fish Weight", fontsize=13, fontweight="bold")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.25)

    ax.text(0.05, 0.93, f"R² = {r2:.4f}", transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#e6f4ea", edgecolor="#55A868", alpha=0.9))

    fig.tight_layout()
    out = VISUALS_DIR / "r2_scatter_plot.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"  [OK] {out}")
    plt.close(fig)


# ────────────────────────────────────────────
#  2. MAE by Species
# ────────────────────────────────────────────
def plot_mae_by_species(df: pd.DataFrame) -> None:
    stats = (df.groupby("Species")["Absolute_Error"]
               .mean()
               .sort_values(ascending=True))

    overall_mae = df["Absolute_Error"].mean()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.barh(stats.index, stats.values,
                   color=[species_color(s) for s in stats.index],
                   edgecolor="white", height=0.6)

    # Value labels
    for bar, val in zip(bars, stats.values):
        ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f} g", va="center", fontsize=9, fontweight="500")

    ax.axvline(overall_mae, color="#C44E52", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.text(overall_mae + 0.5, len(stats) - 0.3,
            f"Overall MAE = {overall_mae:.2f} g",
            fontsize=8.5, color="#C44E52", fontweight="bold")

    ax.set_xlabel("Mean Absolute Error (g)", fontsize=11)
    ax.set_title("MAE by Species", fontsize=13, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.25)
    ax.set_xlim(0, stats.max() * 1.25)

    fig.tight_layout()
    out = VISUALS_DIR / "mae_by_species.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"  [OK] {out}")
    plt.close(fig)


# ────────────────────────────────────────────
#  3. RMSE by Species
# ────────────────────────────────────────────
def plot_rmse_by_species(df: pd.DataFrame) -> None:
    def rmse(group):
        errors = group["Weight"] - group["Predicted_Weight"]
        return np.sqrt(np.mean(errors ** 2))

    stats = (df.groupby("Species")
               .apply(rmse)
               .sort_values(ascending=True))

    errors_all = df["Weight"] - df["Predicted_Weight"]
    overall_rmse = np.sqrt(np.mean(errors_all ** 2))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.barh(stats.index, stats.values,
                   color=[species_color(s) for s in stats.index],
                   edgecolor="white", height=0.6)

    for bar, val in zip(bars, stats.values):
        ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f} g", va="center", fontsize=9, fontweight="500")

    ax.axvline(overall_rmse, color="#C44E52", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.text(overall_rmse + 0.5, len(stats) - 0.3,
            f"Overall RMSE = {overall_rmse:.2f} g",
            fontsize=8.5, color="#C44E52", fontweight="bold")

    ax.set_xlabel("Root Mean Square Error (g)", fontsize=11)
    ax.set_title("RMSE by Species", fontsize=13, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.25)
    ax.set_xlim(0, stats.max() * 1.25)

    fig.tight_layout()
    out = VISUALS_DIR / "rmse_by_species.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"  [OK] {out}")
    plt.close(fig)


# ────────────────────────────────────────────
#  4. Bland–Altman Plot
# ────────────────────────────────────────────
def plot_bland_altman(df: pd.DataFrame) -> None:
    actual = df["Weight"].values
    predicted = df["Predicted_Weight"].values

    mean_vals = (actual + predicted) / 2
    diff_vals = actual - predicted  # actual − predicted

    mean_diff = np.mean(diff_vals)
    sd_diff = np.std(diff_vals, ddof=1)
    upper_loa = mean_diff + 1.96 * sd_diff
    lower_loa = mean_diff - 1.96 * sd_diff

    fig, ax = plt.subplots(figsize=(7, 5))

    for species in sorted(df["Species"].unique()):
        mask = df["Species"] == species
        ax.scatter(mean_vals[mask], diff_vals[mask],
                   label=species, color=species_color(species),
                   s=40, alpha=0.8, edgecolors="white", linewidths=0.4)

    # Reference lines
    ax.axhline(mean_diff, color="#4C72B0", linestyle="-", linewidth=1.3,
               label=f"Mean diff = {mean_diff:+.2f} g")
    ax.axhline(upper_loa, color="#C44E52", linestyle="--", linewidth=1,
               label=f"+1.96 SD = {upper_loa:+.2f} g")
    ax.axhline(lower_loa, color="#C44E52", linestyle="--", linewidth=1,
               label=f"−1.96 SD = {lower_loa:+.2f} g")
    ax.axhline(0, color="grey", linestyle=":", linewidth=0.7, alpha=0.5)

    ax.set_xlabel("Mean of Actual & Predicted (g)", fontsize=11)
    ax.set_ylabel("Difference: Actual − Predicted (g)", fontsize=11)
    ax.set_title("Bland–Altman Plot", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.9, loc="upper left")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    out = VISUALS_DIR / "bland_altman_plot.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"  [OK] {out}")
    plt.close(fig)


# ────────────────────────────────────────────
#  Main
# ────────────────────────────────────────────
def main() -> None:
    print("=" * 55)
    print("  GENERATING RESEARCH PAPER PLOTS")
    print("=" * 55)

    df = load_data()
    print(f"\n  Loaded {len(df)} samples, {df['Species'].nunique()} species\n")

    plot_r2_scatter(df)
    plot_mae_by_species(df)
    plot_rmse_by_species(df)
    plot_bland_altman(df)

    print(f"\n  All plots saved to: {VISUALS_DIR}")
    print("=" * 55)


if __name__ == "__main__":
    main()
