import os
import shutil
import random
from pathlib import Path

def reorganize_dataset():
    data_dir = Path("data/noaa_fish")
    images_train = data_dir / "images" / "train"
    images_val = data_dir / "images" / "val"
    images_test = data_dir / "images" / "test"
    labels_train = data_dir / "labels" / "train"
    labels_val = data_dir / "labels" / "val"
    labels_test = data_dir / "labels" / "test"
    
    # Ensure test directories exist
    images_test.mkdir(parents=True, exist_ok=True)
    labels_test.mkdir(parents=True, exist_ok=True)

    # Collect all existing images
    all_images = []
    for d in [images_train, images_val, images_test]:
        if d.exists():
            all_images.extend(list(d.glob("*.jpg")) + list(d.glob("*.png")))

    # Randomly shuffle
    random.seed(42)
    random.shuffle(all_images)
    
    total = len(all_images)
    train_end = int(total * 0.8)
    val_end = train_end + int(total * 0.1)

    train_files = all_images[:train_end]
    val_files = all_images[train_end:val_end]
    test_files = all_images[val_end:]

    print(f"Total images: {total}")
    print(f"Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")

    # Move files to their exact splits
    def move_to_split(files, dest_img_dir, dest_lbl_dir):
        for img_path in files:
            # Find the corresponding label (it could be in train, val, or test)
            label_name = img_path.stem + ".txt"
            label_path = None
            for lbl_dir in [labels_train, labels_val, labels_test]:
                if (lbl_dir / label_name).exists():
                    label_path = lbl_dir / label_name
                    break
            
            # Use appropriate move handles to avoid cross-device link errors, handle same dir
            if dest_img_dir / img_path.name != img_path:
                shutil.move(str(img_path), dest_img_dir / img_path.name)
            
            if label_path and dest_lbl_dir / label_name != label_path:
                shutil.move(str(label_path), dest_lbl_dir / label_name)

    print("Moving files to train split...")
    move_to_split(train_files, images_train, labels_train)
    
    print("Moving files to val split...")
    move_to_split(val_files, images_val, labels_val)
    
    print("Moving files to test split...")
    move_to_split(test_files, images_test, labels_test)
    
    print("Done reorganizing dataset!")

if __name__ == "__main__":
    reorganize_dataset()
