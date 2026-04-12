"""
Comparative Validation: Predicted Weight vs Actual Weight (Ground Truth)

This script performs a rigorous statistical comparison between the biomass
weights predicted by the computer vision pipeline (allometric equation:
W = a * L^b) and the actual ground-truth weights from the fish_measurements
dataset.

Metrics computed:
  - MAE   (Mean Absolute Error)
  - RMSE  (Root Mean Squared Error)
  - MAPE  (Mean Absolute Percentage Error)
  - R²    (Coefficient of Determination)
  - Max Error
  - Per-species breakdowns

Outputs:
  - outputs/results/validation_report.csv          (per-sample results)
  - outputs/results/validation_summary.csv          (overall + per-species)
  - outputs/visualizations/predicted_vs_actual.png  (scatter plot)
  - outputs/visualizations/error_distribution.png   (error histogram)
  - outputs/visualizations/species_accuracy.png     (per-species bar chart)
  - Console report

Usage:
    python src/validate_biomass_accuracy.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving figures


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
VISUALS_DIR = PROJECT_ROOT / "outputs" / "visualizations"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
VISUALS_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────
#  Metrics
# ──────────────────────────────────────────────

def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Compute regression validation metrics."""
    errors = actual - predicted
    abs_errors = np.abs(errors)
    pct_errors = np.where(actual != 0, abs_errors / actual * 100, 0)

    n = len(actual)
    mae = np.mean(abs_errors)
    rmse = np.sqrt(np.mean(errors ** 2))
    mape = np.mean(pct_errors)
    max_err = np.max(abs_errors)

    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0

    return {
        "n_samples": n,
        "MAE (g)": round(mae, 3),
        "RMSE (g)": round(rmse, 3),
        "MAPE (%)": round(mape, 2),
        "R²": round(r2, 4),
        "Max Error (g)": round(max_err, 3),
        "Mean Actual (g)": round(np.mean(actual), 3),
        "Mean Predicted (g)": round(np.mean(predicted), 3),
    }


# ──────────────────────────────────────────────
#  Visualizations
# ──────────────────────────────────────────────

def plot_predicted_vs_actual(df: pd.DataFrame) -> None:
    """Scatter plot: Predicted vs Actual with perfect-correlation line."""
    fig, ax = plt.subplots(figsize=(8, 8))

    species_list = df["Species"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(species_list)))

    for species, color in zip(species_list, colors):
        subset = df[df["Species"] == species]
        ax.scatter(subset["Weight"], subset["Predicted_Weight"],
                   label=species, color=color, s=60, alpha=0.8, edgecolors="white", linewidths=0.5)

    # Perfect prediction line
    lims = [0, max(df["Weight"].max(), df["Predicted_Weight"].max()) * 1.05]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect Prediction", zorder=0)

    ax.set_xlabel("Actual Weight (g)", fontsize=12)
    ax.set_ylabel("Predicted Weight (g)", fontsize=12)
    ax.set_title("Predicted vs Actual Fish Weight — Validation", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(VISUALS_DIR / "predicted_vs_actual.png", dpi=150)
    print(f"  [OK] Saved: {VISUALS_DIR / 'predicted_vs_actual.png'}")
    plt.close(fig)


def plot_error_distribution(df: pd.DataFrame) -> None:
    """Histogram of prediction errors."""
    errors = df["Weight"] - df["Predicted_Weight"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Absolute error histogram
    abs_errors = np.abs(errors)
    axes[0].hist(abs_errors, bins=20, color="#58a6ff", edgecolor="white", alpha=0.85)
    axes[0].axvline(abs_errors.mean(), color="red", linestyle="--", linewidth=1.5,
                    label=f"Mean = {abs_errors.mean():.1f}g")
    axes[0].set_xlabel("Absolute Error (g)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Distribution of Absolute Errors")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Percentage error histogram
    pct_errors = df["Percentage_Error"]
    axes[1].hist(pct_errors, bins=20, color="#3fb950", edgecolor="white", alpha=0.85)
    axes[1].axvline(pct_errors.mean(), color="red", linestyle="--", linewidth=1.5,
                    label=f"Mean = {pct_errors.mean():.1f}%")
    axes[1].set_xlabel("Percentage Error (%)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Distribution of Percentage Errors")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Biomass Estimation Error Analysis", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(VISUALS_DIR / "error_distribution.png", dpi=150, bbox_inches="tight")
    print(f"  [OK] Saved: {VISUALS_DIR / 'error_distribution.png'}")
    plt.close(fig)


def plot_species_accuracy(summary_df: pd.DataFrame) -> None:
    """Bar chart: per-species MAPE and R²."""
    species_rows = summary_df[summary_df["Group"] != "OVERALL"].copy()
    if species_rows.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # MAPE by species
    bars1 = axes[0].barh(species_rows["Group"], species_rows["MAPE (%)"],
                         color="#58a6ff", edgecolor="white")
    axes[0].set_xlabel("MAPE (%)")
    axes[0].set_title("Mean Absolute Percentage Error by Species")
    axes[0].grid(True, axis="x", alpha=0.3)

    for bar, val in zip(bars1, species_rows["MAPE (%)"]):
        axes[0].text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                     f"{val:.1f}%", va="center", fontsize=9)

    # R² by species
    bars2 = axes[1].barh(species_rows["Group"], species_rows["R²"],
                         color="#3fb950", edgecolor="white")
    axes[1].set_xlabel("R² Score")
    axes[1].set_title("R² (Coefficient of Determination) by Species")
    axes[1].set_xlim(0, 1.1)
    axes[1].grid(True, axis="x", alpha=0.3)

    for bar, val in zip(bars2, species_rows["R²"]):
        axes[1].text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                     f"{val:.4f}", va="center", fontsize=9)

    fig.suptitle("Per-Species Biomass Estimation Accuracy", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(VISUALS_DIR / "species_accuracy.png", dpi=150, bbox_inches="tight")
    print(f"  [OK] Saved: {VISUALS_DIR / 'species_accuracy.png'}")
    plt.close(fig)


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────

def main() -> None:
    print("=" * 65)
    print("  COMPARATIVE VALIDATION: Predicted vs Actual Fish Weight")
    print("=" * 65)

    # ── Load data ──
    csv_path = PROJECT_ROOT / "data" / "processed" / "fish_data_with_predictions.csv"
    if not csv_path.exists():
        print(f"\n[FAIL] File not found: {csv_path}")
        print("  Run the biomass estimation pipeline first.")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    # Drop rows with zero/missing actual weight (some Roach rows have Weight=0)
    df = df[df["Weight"] > 0].copy()

    print(f"\n  Dataset loaded: {len(df)} samples across {df['Species'].nunique()} species")
    print(f"  Species: {', '.join(df['Species'].unique())}")

    # ── Per-sample error ──
    df["Absolute_Error"] = np.abs(df["Weight"] - df["Predicted_Weight"])
    df["Percentage_Error"] = df["Absolute_Error"] / df["Weight"] * 100

    # ── Overall metrics ──
    overall = compute_metrics(df["Weight"].values, df["Predicted_Weight"].values)
    overall["Group"] = "OVERALL"

    print(f"\n{'-' * 50}")
    print("  OVERALL RESULTS")
    print(f"{'-' * 50}")
    for k, v in overall.items():
        if k != "Group":
            print(f"    {k:25s}: {v}")

    # ── Per-species metrics ──
    species_metrics = []
    print(f"\n{'-' * 50}")
    print("  PER-SPECIES RESULTS")
    print(f"{'-' * 50}")

    for species in sorted(df["Species"].unique()):
        subset = df[df["Species"] == species]
        m = compute_metrics(subset["Weight"].values, subset["Predicted_Weight"].values)
        m["Group"] = species
        species_metrics.append(m)
        print(f"\n  {species} (n={m['n_samples']}):")
        print(f"    MAE  = {m['MAE (g)']:>10.3f} g")
        print(f"    RMSE = {m['RMSE (g)']:>10.3f} g")
        print(f"    MAPE = {m['MAPE (%)']:>10.2f} %")
        print(f"    R²   = {m['R²']:>10.4f}")

    # ── Build summary DataFrame ──
    summary_rows = [overall] + species_metrics
    summary_df = pd.DataFrame(summary_rows)
    cols = ["Group", "n_samples", "MAE (g)", "RMSE (g)", "MAPE (%)", "R²",
            "Max Error (g)", "Mean Actual (g)", "Mean Predicted (g)"]
    summary_df = summary_df[cols]

    # ── Save outputs ──
    print(f"\n{'-' * 50}")
    print("  SAVING OUTPUTS")
    print(f"{'-' * 50}")

    report_path = RESULTS_DIR / "validation_report.csv"
    df.to_csv(report_path, index=False)
    print(f"  [OK] Per-sample report: {report_path}")

    summary_path = RESULTS_DIR / "validation_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"  [OK] Summary report:    {summary_path}")

    # ── Generate plots ──
    print(f"\n{'-' * 50}")
    print("  GENERATING VISUALIZATIONS")
    print(f"{'-' * 50}")
    plot_predicted_vs_actual(df)
    plot_error_distribution(df)
    plot_species_accuracy(summary_df)

    # ── Final verdict ──
    mape = overall["MAPE (%)"]
    r2 = overall["R²"]
    print(f"\n{'=' * 65}")
    print(f"  MODEL ACCURACY VERDICT")
    print(f"{'=' * 65}")
    print(f"    Overall MAPE:  {mape:.2f}%  (lower is better)")
    print(f"    Overall R²:    {r2:.4f}  (closer to 1.0 is better)")

    if mape < 15:
        print(f"    [OK] Excellent - prediction error is under 15%")
    elif mape < 25:
        print(f"    [OK] Good - prediction error is under 25%")
    else:
        print(f"    [!!] Moderate - prediction error above 25%, consider refining model")

    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    main()
