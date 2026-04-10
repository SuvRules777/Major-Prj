"""
Fish Detection Model Training Script
Fine-tunes YOLOv8 for fish-specific detection and segmentation.

This script can:
1. Auto-label fish images using semi-supervised learning
2. Train/fine-tune YOLOv8 on labeled fish data
3. Export the trained model for inference
"""

import os
import sys
import shutil
import random
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np
from tqdm import tqdm

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Error: ultralytics not installed. Run: pip install ultralytics")

# Configuration
DATA_DIR = project_root / "data"
TEST_IMAGES_DIR = DATA_DIR / "test_images"
NOAA_FISH_DIR = DATA_DIR / "noaa_fish"
CONFIG_FILE = DATA_DIR / "fish_noaa.yaml"
OUTPUT_DIR = project_root / "runs" / "train"
PRETRAINED_DET_MODEL = project_root / "models" / "weights" / "yolov8n.pt"
PRETRAINED_SEG_MODEL = project_root / "models" / "weights" / "yolov8n-seg.pt"


def create_pseudo_labels(image_dir, output_images_dir, output_labels_dir, model, conf_threshold=0.3):
    """
    Create pseudo-labels for fish images using a pre-trained detector.
    Uses heuristics to identify fish-like objects based on shape and context.
    
    For fish images, we assume each image contains one or more fish,
    and we create bounding box labels based on detected objects or full-image fallback.
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = [f for f in Path(image_dir).iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    print(f"\nCreating pseudo-labels for {len(image_files)} images...")
    
    # Create output directories
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    
    labeled_count = 0
    
    for img_path in tqdm(image_files, desc="Labeling images"):
        try:
            # Read image to get dimensions
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            height, width = img.shape[:2]
            
            # Run detection
            results = model(str(img_path), conf=conf_threshold, verbose=False)
            
            labels = []
            
            for result in results:
                boxes = result.boxes
                
                if boxes is not None and len(boxes) > 0:
                    # Use the largest detected object as fish candidate
                    # (fish images typically show fish prominently)
                    best_box = None
                    best_area = 0
                    
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        area = (x2 - x1) * (y2 - y1)
                        
                        # Fish typically occupy a significant portion of the image
                        # Take boxes that are reasonably sized
                        if area > best_area and area > (width * height * 0.05):
                            best_area = area
                            best_box = box
                    
                    if best_box is not None:
                        x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy()
                        
                        # Convert to YOLO format (center_x, center_y, width, height) normalized
                        center_x = ((x1 + x2) / 2) / width
                        center_y = ((y1 + y2) / 2) / height
                        box_width = (x2 - x1) / width
                        box_height = (y2 - y1) / height
                        
                        # Class 0 = fish
                        labels.append(f"0 {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}")
            
            # If no good detection, use heuristic: fish likely occupies center 80% of image
            if not labels:
                # Create a centered bounding box covering most of the image
                center_x = 0.5
                center_y = 0.5
                box_width = 0.8
                box_height = 0.6
                labels.append(f"0 {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}")
            
            # Copy image to training directory
            dest_img = output_images_dir / img_path.name
            shutil.copy2(img_path, dest_img)
            
            # Save label file
            label_file = output_labels_dir / (img_path.stem + ".txt")
            with open(label_file, 'w') as f:
                f.write('\n'.join(labels))
            
            labeled_count += 1
            
        except Exception as e:
            print(f"\nError processing {img_path.name}: {e}")
    
    print(f"✓ Created labels for {labeled_count} images")
    return labeled_count


def split_dataset(images_dir, labels_dir, train_images_dir, train_labels_dir, 
                  val_images_dir, val_labels_dir, val_split=0.2):
    """Split dataset into training and validation sets."""
    
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    
    random.seed(42)
    random.shuffle(image_files)
    
    val_count = int(len(image_files) * val_split)
    val_files = image_files[:val_count]
    train_files = image_files[val_count:]
    
    print(f"\nSplitting dataset: {len(train_files)} train, {len(val_files)} val")
    
    # Move files
    for img_path in train_files:
        label_path = labels_dir / (img_path.stem + ".txt")
        
        shutil.move(str(img_path), train_images_dir / img_path.name)
        if label_path.exists():
            shutil.move(str(label_path), train_labels_dir / label_path.name)
    
    for img_path in val_files:
        label_path = labels_dir / (img_path.stem + ".txt")
        
        shutil.move(str(img_path), val_images_dir / img_path.name)
        if label_path.exists():
            shutil.move(str(label_path), val_labels_dir / label_path.name)
    
    print("✓ Dataset split complete")


def prepare_dataset():
    """Prepare dataset for training."""
    
    print("=" * 60)
    print("PREPARING FISH DATASET FOR TRAINING")
    print("=" * 60)
    
    if not YOLO_AVAILABLE:
        print("Error: ultralytics not available")
        return False
    
    # Load pre-trained model for pseudo-labeling
    print("\nLoading pre-trained model for pseudo-labeling...")
    if PRETRAINED_DET_MODEL.exists():
        model = YOLO(str(PRETRAINED_DET_MODEL))
    else:
        model = YOLO('yolov8n.pt')  # Auto-download fallback
    
    # Temporary directories for initial labeling
    temp_images = NOAA_FISH_DIR / "temp_images"
    temp_labels = NOAA_FISH_DIR / "temp_labels"
    
    # Create pseudo-labels
    labeled = create_pseudo_labels(
        TEST_IMAGES_DIR,
        temp_images,
        temp_labels,
        model,
        conf_threshold=0.25
    )
    
    if labeled == 0:
        print("No images labeled. Cannot proceed with training.")
        return False
    
    # Split into train/val
    split_dataset(
        temp_images, temp_labels,
        NOAA_FISH_DIR / "images" / "train",
        NOAA_FISH_DIR / "labels" / "train",
        NOAA_FISH_DIR / "images" / "val",
        NOAA_FISH_DIR / "labels" / "val",
        val_split=0.2
    )
    
    # Clean up temp directories
    if temp_images.exists():
        shutil.rmtree(temp_images)
    if temp_labels.exists():
        shutil.rmtree(temp_labels)
    
    print("\n✓ Dataset preparation complete")
    return True


def train_fish_model(epochs=50, batch_size=16, img_size=640):
    """
    Fine-tune YOLOv8 for fish detection.
    """
    
    print("\n" + "=" * 60)
    print("TRAINING FISH DETECTION MODEL")
    print("=" * 60)
    
    if not YOLO_AVAILABLE:
        print("Error: ultralytics not available")
        return None
    
    # Check if dataset exists
    train_images = NOAA_FISH_DIR / "images" / "train"
    if not train_images.exists() or len(list(train_images.glob("*"))) == 0:
        print("Training data not found. Running dataset preparation...")
        if not prepare_dataset():
            return None
    
    # Count training images
    train_count = len(list(train_images.glob("*.jpg"))) + len(list(train_images.glob("*.png")))
    val_images = NOAA_FISH_DIR / "images" / "val"
    val_count = len(list(val_images.glob("*.jpg"))) + len(list(val_images.glob("*.png")))
    
    print(f"\nTraining images: {train_count}")
    print(f"Validation images: {val_count}")
    
    # Load pre-trained YOLOv8 model
    print("\nLoading pre-trained YOLOv8n detector model...")
    if PRETRAINED_DET_MODEL.exists():
        model = YOLO(str(PRETRAINED_DET_MODEL))
    else:
        model = YOLO('yolov8n.pt')
    
    # Train the model
    print(f"\nStarting training for {epochs} epochs...")
    print(f"Batch size: {batch_size}, Image size: {img_size}")
    
    results = model.train(
        data=str(CONFIG_FILE),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        patience=10,
        save=True,
        device='',  # Auto-select GPU if available, otherwise switch to CPU
        project=str(OUTPUT_DIR),
        name='fish_detector',
        exist_ok=True,
        pretrained=True,
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=3,
        augment=True,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        degrees=15,
        translate=0.2,
        scale=0.5,
        fliplr=0.5,
        flipud=0.0,
    )
    
    # Get best model path
    best_model_path = OUTPUT_DIR / "fish_detector" / "weights" / "best.pt"
    
    print(f"\n✓ Training complete!")
    print(f"  Best model saved to: {best_model_path}")
    
    return best_model_path


def validate_model(model_path):
    """Evaluate the trained model on exactly the test split."""
    
    print("\n" + "=" * 60)
    print("TESTING FISH DETECTION MODEL MAP")
    print("=" * 60)
    
    model = YOLO(str(model_path))
    
    # Run absolute test-set validation
    metrics = model.val(data=str(CONFIG_FILE), split="test")
    
    print(f"\nFinal Test Set Results:")
    print(f"  mAP@0.5: {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall: {metrics.box.mr:.4f}")
    
    return metrics


def main():
    """Main training pipeline."""
    
    print("=" * 60)
    print("FISH DETECTION MODEL TRAINING PIPELINE")
    print("=" * 60)
    
    if not YOLO_AVAILABLE:
        print("\nError: ultralytics not installed")
        print("Install with: pip install ultralytics")
        return
    
    # Check for test images
    if not TEST_IMAGES_DIR.exists():
        print(f"\nError: Test images not found at {TEST_IMAGES_DIR}")
        return
    
    image_count = len(list(TEST_IMAGES_DIR.glob("*.jpg"))) + len(list(TEST_IMAGES_DIR.glob("*.png")))
    print(f"\nFound {image_count} images in test_images/")
    
    # Prepare dataset
    print("\n[Step 1/3] Dataset already split via reorganize_dataset.py. Skipping redundant preparation...")
    # if not prepare_dataset():
    #     print("Dataset preparation failed.")
    #     return
    
    # Train model
    print("\n[Step 2/3] Training model...")
    model_path = train_fish_model(
        epochs=100,  # Increased for higher accuracy
        batch_size=8,
        img_size=640
    )
    
    if model_path is None:
        print("Training failed.")
        return
    
    # Validate model
    print("\n[Step 3/3] Validating model...")
    validate_model(model_path)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nTrained model saved at: {model_path}")
    print(f"\nTo use the model for biomass estimation, run:")
    print(f"  python src/run_biomass_estimation.py --model {model_path}")


if __name__ == "__main__":
    main()
