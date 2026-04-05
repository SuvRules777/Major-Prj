"""
Biomass Estimation Script for Fish Images
Uses YOLOv8 segmentation to detect fish and estimate biomass from pixel measurements.

Usage:
    python run_biomass_estimation.py                    # Use default model
    python run_biomass_estimation.py --model path/to/model.pt  # Use custom model
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Try to import ultralytics
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. Install with: pip install ultralytics")

# Configuration
IMAGE_DIR = project_root / "data" / "test_images"
OUTPUT_DIR = project_root / "outputs" / "results"
VISUALIZATION_DIR = project_root / "outputs" / "visualizations"
MODEL_PATH = project_root / "models" / "weights" / "yolov8n-seg.pt"
TRAINED_MODEL_PATH = project_root / "runs" / "train" / "fish_detector" / "weights" / "best.pt"

# Default length-weight parameters (allometric equation: W = a * L^b)
# Using average fish parameters when species is unknown
DEFAULT_PARAMS = {
    'a': 0.01,  # Coefficient
    'b': 3.0,   # Exponent (typically ~3 for fish)
    'pixels_per_cm': 10  # Assumed pixel density (adjust based on camera setup)
}

def load_model(model_path=None):
    """Load YOLOv8 segmentation model."""
    if not YOLO_AVAILABLE:
        return None
    
    # Priority: custom path > trained fish model > default model
    if model_path and Path(model_path).exists():
        print(f"Loading custom model: {model_path}")
        return YOLO(str(model_path))
    
    if TRAINED_MODEL_PATH.exists():
        print(f"Loading trained fish model: {TRAINED_MODEL_PATH}")
        return YOLO(str(TRAINED_MODEL_PATH))
    
    print("Loading default YOLOv8 segmentation model...")
    if MODEL_PATH.exists():
        model = YOLO(str(MODEL_PATH))
    else:
        print("Downloading YOLOv8n-seg model...")
        model = YOLO('yolov8n-seg.pt')
    
    print(f"Model loaded successfully")
    return model

def detect_fish(image_path, model, conf_threshold=0.25):
    """
    Run YOLOv8 detection on an image.
    Returns list of detections with pixel measurements.
    """
    results = model(str(image_path), conf=conf_threshold, verbose=False)
    
    detections = []
    
    for result in results:
        boxes = result.boxes
        masks = result.masks
        
        if boxes is not None:
            for idx, box in enumerate(boxes):
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                detection = {
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(box.conf[0]),
                    'class_id': class_id,
                    'class_name': class_name,
                    'pixel_length': x2 - x1,  # Width of bounding box
                    'pixel_height': y2 - y1,  # Height of bounding box
                    'bbox_area': (x2 - x1) * (y2 - y1),
                }
                
                # Calculate mask-based measurements if available
                if masks is not None and idx < len(masks):
                    try:
                        mask_coords = masks[idx].xy[0]
                        if len(mask_coords) > 0:
                            mask_area = cv2.contourArea(mask_coords.astype(np.int32))
                            detection['mask_area'] = mask_area
                            
                            # Calculate major axis length (fish length estimate)
                            if len(mask_coords) >= 5:
                                ellipse = cv2.fitEllipse(mask_coords.astype(np.int32))
                                major_axis = max(ellipse[1])
                                detection['major_axis_length'] = major_axis
                            else:
                                detection['major_axis_length'] = detection['pixel_length']
                    except:
                        detection['mask_area'] = detection['bbox_area']
                        detection['major_axis_length'] = detection['pixel_length']
                else:
                    detection['mask_area'] = detection['bbox_area']
                    detection['major_axis_length'] = detection['pixel_length']
                
                detections.append(detection)
    
    return detections

def estimate_biomass(pixel_length, params=DEFAULT_PARAMS):
    """
    Estimate fish biomass using allometric equation: W = a * L^b
    
    Args:
        pixel_length: Length in pixels
        params: Dictionary with 'a', 'b', and 'pixels_per_cm'
    
    Returns:
        Estimated weight in grams
    """
    # Convert pixels to cm
    length_cm = pixel_length / params['pixels_per_cm']
    
    # Apply allometric equation
    weight_g = params['a'] * (length_cm ** params['b'])
    
    return weight_g, length_cm

def process_images(image_dir, model, conf_threshold=0.25, output_csv=None):
    """
    Process all images in directory and estimate biomass.
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = [f for f in Path(image_dir).iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    print(f"\nFound {len(image_files)} images to process")
    print("=" * 60)
    
    all_results = []
    
    for idx, img_path in enumerate(image_files, 1):
        print(f"[{idx}/{len(image_files)}] Processing: {img_path.name}")
        
        try:
            # Detect fish in image
            detections = detect_fish(img_path, model, conf_threshold)
            
            if len(detections) == 0:
                # No detections - still record the image
                all_results.append({
                    'image_name': img_path.name,
                    'detection_id': 0,
                    'class_name': 'no_detection',
                    'confidence': 0.0,
                    'pixel_length': 0,
                    'pixel_height': 0,
                    'bbox_area': 0,
                    'mask_area': 0,
                    'estimated_length_cm': 0,
                    'estimated_weight_g': 0,
                })
                print(f"    No fish detected")
            else:
                for det_idx, det in enumerate(detections):
                    # Estimate biomass
                    weight_g, length_cm = estimate_biomass(
                        det.get('major_axis_length', det['pixel_length'])
                    )
                    
                    result = {
                        'image_name': img_path.name,
                        'detection_id': det_idx + 1,
                        'class_name': det['class_name'],
                        'confidence': round(det['confidence'], 3),
                        'pixel_length': round(det['pixel_length'], 1),
                        'pixel_height': round(det['pixel_height'], 1),
                        'bbox_area': round(det['bbox_area'], 1),
                        'mask_area': round(det.get('mask_area', det['bbox_area']), 1),
                        'estimated_length_cm': round(length_cm, 2),
                        'estimated_weight_g': round(weight_g, 2),
                    }
                    all_results.append(result)
                    
                    print(f"    Detection {det_idx+1}: {det['class_name']} "
                          f"(conf: {det['confidence']:.2f}) - "
                          f"Est. length: {length_cm:.1f}cm, weight: {weight_g:.1f}g")
        
        except Exception as e:
            print(f"    Error processing: {e}")
            all_results.append({
                'image_name': img_path.name,
                'detection_id': 0,
                'class_name': 'error',
                'confidence': 0.0,
                'pixel_length': 0,
                'pixel_height': 0,
                'bbox_area': 0,
                'mask_area': 0,
                'estimated_length_cm': 0,
                'estimated_weight_g': 0,
            })
    
    # Create DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save to CSV
    if output_csv is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_csv = OUTPUT_DIR / f"biomass_estimation_{timestamp}.csv"
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_csv, index=False)
    print(f"\n✓ Results saved to: {output_csv}")
    
    return results_df

def print_summary(results_df):
    """Print summary statistics."""
    print("\n" + "=" * 60)
    print("BIOMASS ESTIMATION SUMMARY")
    print("=" * 60)
    
    # Filter out non-detections
    valid_detections = results_df[results_df['class_name'] != 'no_detection']
    valid_detections = valid_detections[valid_detections['class_name'] != 'error']
    
    print(f"\nTotal images processed: {results_df['image_name'].nunique()}")
    print(f"Total detections: {len(valid_detections)}")
    
    if len(valid_detections) > 0:
        print(f"\nDetected objects by class:")
        class_counts = valid_detections['class_name'].value_counts()
        for cls, count in class_counts.items():
            print(f"  - {cls}: {count}")
        
        print(f"\nBiomass Statistics:")
        print(f"  Average estimated length: {valid_detections['estimated_length_cm'].mean():.2f} cm")
        print(f"  Average estimated weight: {valid_detections['estimated_weight_g'].mean():.2f} g")
        print(f"  Total estimated biomass: {valid_detections['estimated_weight_g'].sum():.2f} g")
        print(f"  Min weight: {valid_detections['estimated_weight_g'].min():.2f} g")
        print(f"  Max weight: {valid_detections['estimated_weight_g'].max():.2f} g")
    else:
        print("\nNo valid fish detections found.")
        print("Note: The model detects general COCO objects. For fish-specific detection,")
        print("      fine-tune the model on a labeled fish dataset.")
    
    print("=" * 60)

def main():
    """Main function to run biomass estimation."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Fish Biomass Estimation from Images")
    parser.add_argument('--model', type=str, default=None, 
                        help='Path to custom YOLOv8 model (default: auto-detect)')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold for detection (default: 0.25)')
    parser.add_argument('--images', type=str, default=None,
                        help='Path to images directory (default: data/test_images)')
    args = parser.parse_args()
    
    print("=" * 60)
    print("FISH BIOMASS ESTIMATION FROM IMAGES")
    print("=" * 60)
    
    if not YOLO_AVAILABLE:
        print("\nError: ultralytics package not available.")
        print("Install with: pip install ultralytics")
        return
    
    # Determine image directory
    image_dir = Path(args.images) if args.images else IMAGE_DIR
    
    # Check if images exist
    if not image_dir.exists():
        print(f"\nError: Image directory not found: {image_dir}")
        return
    
    # Load model
    model = load_model(args.model)
    if model is None:
        return
    
    # Process images
    results_df = process_images(
        image_dir, 
        model, 
        conf_threshold=args.conf,
        output_csv=OUTPUT_DIR / "biomass_estimation_results.csv"
    )
    
    # Print summary
    print_summary(results_df)
    
    return results_df

if __name__ == "__main__":
    main()
