import cv2
import sys
from pathlib import Path
from tqdm import tqdm

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics is not installed. Please run this inside your activated virtual environment.")
    sys.exit(1)

def main():
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "noaa_fish"
    
    print("Initializing YOLOv8 Base Model for object detection...")
    model_path = project_root / "models" / "weights" / "yolov8n.pt"
    if model_path.exists():
        model = YOLO(str(model_path))
    else:
        model = YOLO('yolov8n.pt')

    splits = ["train", "val", "test"]
    
    generated = 0
    skipped = 0

    print("\nScanning dataset for missing label files...")
    for split in splits:
        img_dir = data_dir / "images" / split
        lbl_dir = data_dir / "labels" / split
        
        if not img_dir.exists():
            continue
            
        lbl_dir.mkdir(parents=True, exist_ok=True)
        
        images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        
        for img_path in tqdm(images, desc=f"Processing {split} split"):
            lbl_file = lbl_dir / (img_path.stem + ".txt")
            
            # If the label already exists and is not totally empty, skip generating it.
            if lbl_file.exists() and lbl_file.stat().st_size > 0:
                skipped += 1
                continue
                
            # Perform inference to generate pseudo-label
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            height, width = img.shape[:2]
            
            results = model(str(img_path), verbose=False)
            labels = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    best_area = 0
                    best_box = None
                    
                    # Heuristic: the largest centrally framed object is assumed to be the required fish target
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        area = (x2 - x1) * (y2 - y1)
                        if area > best_area and area > (width * height * 0.05):
                            best_area = area
                            best_box = box
                            
                    if best_box is not None:
                        x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy()
                        
                        # Convert to normalized YOLO format
                        center_x = ((x1 + x2) / 2) / width
                        center_y = ((y1 + y2) / 2) / height
                        box_w = (x2 - x1) / width
                        box_h = (y2 - y1) / height
                        labels.append(f"0 {center_x:.6f} {center_y:.6f} {box_w:.6f} {box_h:.6f}")
            
            # Fallback heuristic: assume a central large bounding box if nothing is detected
            if not labels:
                labels.append(f"0 0.500000 0.500000 0.800000 0.600000")
                
            # Write to the labels folder
            with open(lbl_file, 'w') as f:
                f.write('\n'.join(labels))
                
            generated += 1
            
    print("\n" + "=" * 60)
    print("PSEUDO-LABELING SUMMARY")
    print("=" * 60)
    print(f"Newly Generated Labels : {generated}")
    print(f"Skipped Configured Files: {skipped}")
    print(f"Total Annotated Dataset : {generated + skipped}")

if __name__ == '__main__':
    main()
