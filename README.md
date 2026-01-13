# 🐟 Smart Vision-Based Fish Biomass Estimation Using Enhanced Deep Learning

**Final Year Project**  
*Software-Only Implementation*

---

## 📋 Project Overview

This project implements an intelligent system for estimating fish biomass from images using computer vision and deep learning. The system combines **YOLOv8 instance segmentation** for fish detection with a **biometric regression model** to estimate fish weight.

### Key Features
- ✅ YOLOv8-Seg for accurate fish detection and segmentation
- ✅ Automated 2D dimension extraction from fish images
- ✅ Species-specific Length-Weight regression model ($W = a \cdot L^b$)
- ✅ Real-time biomass estimation pipeline
- ✅ Comprehensive validation and visualization tools

---

## 🏗️ Project Architecture

```
Input Image → YOLOv8 Detection → Pixel Measurement → Weight Estimation → Output Biomass
```

### Two-Module System:
1. **Visual Module**: YOLOv8-Seg extracts 2D fish dimensions (length/area) from images
2. **Biometric Module**: Species-specific regression converts dimensions to weight estimates

---

## 📁 Directory Structure

```
Major-Prj/
│
├── data/
│   ├── raw/                      # Original datasets
│   │   └── fish_measurements.csv # Your 159-row biometric dataset
│   ├── processed/                # Processed data & parameters
│   └── test_images/              # Sample fish images for testing
│
├── notebooks/
│   ├── phase1_data_analysis.ipynb      # Length-Weight parameter calculation
│   ├── phase2_yolo_vision.ipynb        # YOLOv8 setup & testing
│   ├── phase3_biomass_calculation.ipynb # Integration & logic
│   └── phase4_final_pipeline.ipynb     # Complete system demo
│
├── src/
│   ├── __init__.py
│   ├── detection.py              # YOLOv8 detection functions
│   ├── biomass_calculator.py     # Weight estimation logic
│   ├── utils.py                  # Helper functions
│   └── main.py                   # Final pipeline script
│
├── models/
│   └── yolov8_fish.pt            # Trained/fine-tuned YOLO model
│
├── outputs/
│   ├── results/                  # Numerical results & CSVs
│   └── visualizations/           # Plots & annotated images
│
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- Virtual environment (recommended)
- Your `fish_measurements.csv` file (159 rows)

### Installation

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Place Your Dataset**
   - Copy `fish_measurements.csv` to `data/raw/`
   - The CSV should have columns: `Species`, `Length`, `Weight`

---

## 📚 Phase-by-Phase Implementation

### ✅ Phase 1: Data Analysis & Regression Model
**Notebook:** `notebooks/phase1_data_analysis.ipynb`

**Tasks:**
- Load and explore fish measurement dataset
- Calculate Length-Weight relationship ($W = a \cdot L^b$) for each species
- Validate regression model with R² scores
- Generate "Predicted vs Actual" plots for report

**Output:**
- `data/processed/length_weight_parameters.csv` (a & b constants)
- Visualization plots

**Status:** 🟢 Ready to run!

---

### 🔲 Phase 2: YOLOv8 Vision System
**Notebook:** `notebooks/phase2_yolo_vision.ipynb` (Coming next)

**Tasks:**
- Install and configure YOLOv8 from Ultralytics
- Run inference on test fish images
- Extract bounding boxes and segmentation masks
- Calculate pixel-based dimensions

**Output:**
- Detection results with coordinates
- Annotated images

---

### 🔲 Phase 3: Biomass Calculation Logic
**Notebook:** `notebooks/phase3_biomass_calculation.ipynb`

**Tasks:**
- Implement `calculate_biomass()` function
- Apply pixel-to-cm conversion (calibration factor)
- Use regression parameters from Phase 1
- Test on sample detections

**Output:**
- Working biomass estimation function

---

### 🔲 Phase 4: Final Pipeline & Integration
**Script:** `src/main.py`

**Tasks:**
- Integrate detection + calculation
- Process complete images end-to-end
- Generate final results and reports
- Create demo for presentation

**Output:**
- Complete working system
- Documentation for report

---

## 🧮 Mathematical Foundation

### Length-Weight Relationship
The biomass estimation is based on the allometric equation:

$$W = a \cdot L^b$$

Where:
- $W$ = Weight (grams)
- $L$ = Length (centimeters)
- $a$ = Species-specific scaling constant
- $b$ = Allometric exponent (typically ≈ 3)

### Pixel-to-Metric Conversion
For simulated measurements (no physical rig):

$$L_{cm} = L_{pixels} \times \text{calibration\\_factor}$$

**Note:** You'll define a reasonable calibration factor based on typical fish sizes in your dataset.

---

## 🛠️ Tech Stack

- **Deep Learning**: PyTorch, Ultralytics YOLOv8
- **Computer Vision**: OpenCV, Pillow
- **Data Science**: Pandas, NumPy, SciPy, scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Development**: Jupyter Notebooks, Python 3.9

---

## 📊 Expected Results

### Model Performance Metrics
- **Length-Weight Regression**: R² > 0.90 (per species)
- **Detection**: YOLOv8 mAP > 0.85 (if using labeled fish images)
- **Biomass Estimation**: MAPE < 15%

### Deliverables for Report
1. ✅ Length-Weight parameters table
2. ✅ Predicted vs Actual scatter plots
3. ✅ Annotated detection images
4. ✅ System architecture diagram
5. ✅ Performance metrics comparison

---

## 🎯 Current Status: Phase 1 Ready

### ✅ Completed:
- [x] Project structure created
- [x] Dependencies defined
- [x] Phase 1 notebook ready

### 📍 Next Steps:
1. Open `notebooks/phase1_data_analysis.ipynb`
2. Place your CSV in `data/raw/fish_measurements.csv`
3. Run all cells to calculate regression parameters
4. Review output visualizations
5. Proceed to Phase 2

---

## 📖 Usage Example

Once complete, the system will work like this:

```python
from src.main import estimate_fish_biomass

# Process an image
result = estimate_fish_biomass(
    image_path='data/test_images/fish_sample.jpg',
    species='Perch',  # or auto-detect
    calibration_factor=0.5  # cm per pixel
)

print(f"Detected: {result['species']}")
print(f"Estimated Weight: {result['weight_grams']:.2f} grams")
```

**Output:**
```
Detected: Perch
Estimated Weight: 342.56 grams
```

---

## 📝 Notes for Your Report

### Software-Only Constraint
This implementation **does not** require a physical stereo camera rig. Instead:
- Uses existing fish image datasets
- Applies assumed calibration factors
- Focuses on the **algorithmic pipeline** and model accuracy

### Novel Contributions
1. Integration of YOLOv8-Seg with biometric modeling
2. Species-specific regression parameter calculation
3. End-to-end automated biomass estimation
4. Validation framework for accuracy assessment

---

## 🤝 Support

If you encounter issues:
1. Check that all dependencies are installed
2. Verify CSV file format matches expected columns
3. Ensure YOLOv8 model is downloaded (handled automatically)

---

## 📧 Project Info

**Title:** Smart Vision-Based Fish Biomass Estimation Using Enhanced Deep Learning  
**Type:** Final Year Project (Software Implementation)  
**Dataset:** Fish Market Dataset (159 samples)  
**Status:** Phase 1 Complete ✅

---

**Ready to begin? Open Phase 1 notebook and let's calculate those regression parameters! 🚀**
