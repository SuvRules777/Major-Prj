# Smart Vision-Based Fish Biomass Estimation

This project estimates fish biomass from images using YOLOv8-based fish detection and length-weight estimation.

## Repository Layout

Major-Prj/
- data/
  - raw/                     # Source measurement data
  - processed/               # Processed tables and derived parameters
  - noaa_fish/               # Detection dataset (images and labels)
  - test_images/             # Inference test images
- models/
  - cnn_models.py            # CNN model definitions
  - weights/                 # Pretrained model weights
- notebooks/
  - phase1_data_analysis.ipynb
  - phase2_yolo_vision_and_comparison.ipynb
- outputs/
  - results/                 # CSV outputs
  - visualizations/          # Plot/image outputs
    - training/fish_detector/# Training preview plots/images
- runs/
  - train/
    - fish_detector/         # Latest YOLO training run artifacts
- src/
  - train_fish_model.py      # Dataset prep + YOLO training
  - run_biomass_estimation.py# Inference + biomass estimation
  - model_comparison.py      # Biomass regressor comparison
- requirements.txt
- README.md

## What Was Cleaned

- Removed stale root-level result files and duplicate plots.
- Consolidated model weights into models/weights.
- Consolidated latest detector run into runs/train/fish_detector.
- Removed temporary/editor-generated files.
- Updated scripts to write outputs only under outputs/results and outputs/visualizations.

## How To Run

Install dependencies:

pip install -r requirements.txt

Train detector:

python src/train_fish_model.py

Run biomass estimation:

python src/run_biomass_estimation.py

Compare biomass models:

python src/model_comparison.py

## Output Locations

- Biomass estimation CSV: outputs/results/biomass_estimation_results.csv
- Detection metrics CSV: outputs/results/detection_metrics.csv
- Model comparison CSV: outputs/results/model_comparison.csv
- Generated plots: outputs/visualizations/
- Training run visuals: outputs/visualizations/training/fish_detector/

## Notes

- If pretrained YOLO weights are missing in models/weights, scripts fall back to Ultralytics auto-download.
- Keep large generated artifacts in outputs/ and runs/ to avoid root-level clutter.
