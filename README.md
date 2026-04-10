# Smart Vision-Based Fish Biomass Estimation

Computer vision pipeline for estimating fish biomass from images using YOLOv8 detection/segmentation and allometric length-weight equations.

## Project Overview

This project combines:

- Fish detection and segmentation from images (YOLOv8)
- Pixel-to-length conversion and biomass estimation
- Regression model benchmarking for biomass prediction from tabular measurements

It includes data preparation, model training, inference, evaluation outputs, and notebooks for analysis.

## Key Features

- YOLOv8-based fish detection training pipeline
- Automatic pseudo-label generation for dataset bootstrapping
- Image-based biomass estimation using $W = a \cdot L^b$
- Tabular regression model comparison (Linear + MLP variants)
- Organized results in dedicated outputs and runs folders

## Repository Structure

```text
Major-Prj/
├─ data/
│  ├─ raw/                     # Original fish measurement data
│  ├─ processed/               # Derived parameters / processed tables
│  ├─ noaa_fish/               # YOLO dataset (images + labels)
│  └─ test_images/             # Input images for biomass inference
├─ models/
│  ├─ cnn_models.py
│  └─ weights/                 # Pretrained YOLO weights
├─ notebooks/
│  ├─ phase1_data_analysis.ipynb
│  └─ phase2_yolo_vision_and_comparison.ipynb
├─ outputs/
│  ├─ results/                 # CSV metrics and predictions
│  └─ visualizations/          # Charts and plots
├─ runs/
│  └─ train/fish_detector/     # YOLO training artifacts
├─ src/
│  ├─ train_fish_model.py
│  ├─ run_biomass_estimation.py
│  └─ model_comparison.py
├─ requirements.txt
└─ README.md
```

## Requirements

- Python 3.10+
- pip
- Optional GPU with CUDA for faster YOLO training/inference

Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

### 1) Train Fish Detector

```bash
python src/train_fish_model.py
```

What it does:

- Prepares data and pseudo-labels from images in `data/test_images`
- Splits data into train/validation sets under `data/noaa_fish`
- Fine-tunes YOLOv8 and stores outputs in `runs/train/fish_detector`

### 2) Run Biomass Estimation on Images

```bash
python src/run_biomass_estimation.py
```

Optional arguments:

```bash
python src/run_biomass_estimation.py --model runs/train/fish_detector/weights/best.pt --conf 0.25 --images data/test_images
```

What it does:

- Detects fish in input images
- Estimates fish length from detection geometry
- Computes estimated biomass using allometric parameters
- Saves predictions to `outputs/results/biomass_estimation_results.csv`

### 3) Compare Regression Models (Tabular)

```bash
python src/model_comparison.py
```

What it does:

- Loads `data/raw/fish_measurements.csv`
- Trains multiple regression baselines
- Saves ranking metrics to `outputs/results/model_comparison.csv`
- Generates plots in `outputs/visualizations`

### 4) Realtime Phone Capture App (MVP)

This project now includes a realtime API + phone camera UI:

- API endpoint: `POST /predict-biomass`
- Phone UI: browser page at `/` (captures camera frame and sends it to API)

Run it:

```bash
pip install -r requirements.txt
python src/realtime_biomass_api.py --host 0.0.0.0 --port 8000
```

Then:

1. Connect phone and PC to the same network.
2. Find your PC local IP (for example `192.168.1.10`).
3. Open `http://<PC_IP>:8000` on phone browser.
4. Tap **Capture + Estimate** to get fish count + estimated biomass.

Request form fields for `/predict-biomass`:

- `file`: image upload (jpg/png)
- `conf`: detection confidence threshold (default `0.25`)
- `pixels_per_cm`: camera calibration (default `10`)

Response includes:

- `fish_count`
- `total_estimated_biomass_g`
- per-detection length and weight estimates

## Main Outputs

- `outputs/results/biomass_estimation_results.csv`
- `outputs/results/model_comparison.csv`
- `outputs/visualizations/biomass_model_comparison_metrics.png`
- `outputs/visualizations/biomass_training_curves.png`
- `runs/train/fish_detector/weights/best.pt`

## Configuration Note

The file `data/fish_noaa.yaml` currently contains an absolute `path` value.
If training fails because of dataset path mismatch, update `path` to match your local workspace location.

## Reproducibility Tips

- Keep generated artifacts in `outputs/` and `runs/`
- Keep source data in `data/raw/`
- Pin package versions via `requirements.txt`

## License

Add your project license here (for example: MIT).
