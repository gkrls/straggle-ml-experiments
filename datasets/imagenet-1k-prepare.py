#!/usr/bin/env python3
"""
ImageNet Dataset Preparation Script

Extracts and organizes ImageNet into train/val folders with matching class names.

Usage:
    python prepare_imagenet.py <input_dir> [output_dir]
    python prepare_imagenet.py --verify-only <dataset_dir>
"""

import os
import tarfile
import shutil
from pathlib import Path
import argparse


def extract_train(train_tar, output_dir):
    """Extract training data into class folders"""
    print("Extracting training data...")
    train_dir = output_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    
    with tarfile.open(train_tar) as tar:
        class_tars = [m for m in tar if m.name.endswith('.tar')]
        total = len(class_tars)
        
        for i, member in enumerate(class_tars, 1):
            if i % 10 == 0:
                print(f"  Extracting class {i}/{total}...", end='\r')
            
            # Extract class tar
            tar.extract(member, train_dir)
            class_tar_path = train_dir / member.name
            
            # Create class directory (e.g., n01440764)
            class_name = member.name.replace('.tar', '')
            class_dir = train_dir / class_name
            class_dir.mkdir(exist_ok=True)
            
            # Extract images into class directory
            with tarfile.open(class_tar_path) as class_tar:
                class_tar.extractall(class_dir)
            
            os.remove(class_tar_path)
    
    print(f"  Extracted {total} training classes")
    return sorted([d.name for d in train_dir.iterdir() if d.is_dir()])


def extract_val(val_tar, devkit_tar, output_dir, train_classes):
    """Extract validation data and organize using training class names"""
    print("Extracting validation data...")
    
    # Extract validation images
    val_temp = output_dir / "val_temp"
    val_temp.mkdir(exist_ok=True)
    
    with tarfile.open(val_tar) as tar:
        tar.extractall(val_temp)
    
    # Get validation labels from devkit
    print("  Extracting devkit for labels...")
    devkit_dir = output_dir / "devkit_temp"
    devkit_dir.mkdir(exist_ok=True)
    
    with tarfile.open(devkit_tar) as tar:
        # Only extract the ground truth file we need
        for member in tar:
            if 'val_ground_truth' in member.name or 'validation_ground_truth' in member.name:
                tar.extract(member, devkit_dir)
    
    # Find the ground truth file
    val_labels_file = None
    for path in devkit_dir.rglob('*val*ground_truth.txt'):
        val_labels_file = path
        break
    
    if not val_labels_file:
        print("ERROR: Could not find validation ground truth in devkit!")
        print("  Falling back to using numeric folders (0000, 0001, etc.)")
        print("  WARNING: This will NOT match training classes!")
        shutil.rmtree(devkit_dir)
        shutil.rmtree(val_temp)
        return []
    
    # Read labels (1-indexed class indices)
    with open(val_labels_file) as f:
        val_labels = [int(line.strip()) - 1 for line in f]  # Convert to 0-indexed
    
    # Create validation directory with proper class names
    val_dir = output_dir / "val"
    val_dir.mkdir(exist_ok=True)
    
    # Move images to class folders matching training structure
    val_images = sorted([f for f in os.listdir(val_temp) if f.endswith('.JPEG')])
    
    if len(val_images) != len(val_labels):
        print(f"ERROR: Image count ({len(val_images)}) != label count ({len(val_labels)})")
        return []
    
    print(f"  Organizing {len(val_images)} images into {len(set(val_labels))} classes...")
    
    for img_file, class_idx in zip(val_images, val_labels):
        if class_idx < len(train_classes):
            # Use the training class name for this index
            class_name = train_classes[class_idx]
            class_dir = val_dir / class_name
            class_dir.mkdir(exist_ok=True)
            
            src = val_temp / img_file
            dst = class_dir / img_file
            shutil.move(str(src), str(dst))
    
    # Cleanup
    shutil.rmtree(val_temp)
    shutil.rmtree(devkit_dir)
    
    val_classes = sorted([d.name for d in val_dir.iterdir() if d.is_dir()])
    print(f"  Created {len(val_classes)} validation classes")
    return val_classes


def verify_dataset(dataset_dir):
    """Verify train and validation folders match"""
    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "val"
    
    if not train_dir.exists() or not val_dir.exists():
        print("ERROR: train or val directory not found!")
        return False
    
    train_classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    val_classes = sorted([d.name for d in val_dir.iterdir() if d.is_dir()])
    
    print(f"\nDataset structure:")
    print(f"  Training: {len(train_classes)} classes")
    print(f"  Validation: {len(val_classes)} classes")
    
    if train_classes == val_classes:
        print("  ✓ Class folders match!")
        print(f"  First 5: {train_classes[:5]}")
        print(f"  Last 5: {train_classes[-5:]}")
        
        # Sample some statistics
        sample_train = sum(len(list((train_dir / c).glob('*.JPEG'))) for c in train_classes[:5])
        sample_val = sum(len(list((val_dir / c).glob('*.JPEG'))) for c in val_classes[:5])
        print(f"  Sample images (first 5 classes): train={sample_train}, val={sample_val}")
        return True
    else:
        print("  ✗ ERROR: Class folders don't match!")
        
        # Check what type of mismatch
        if val_classes and val_classes[0].isdigit():
            print("  Validation uses numeric folders (0000, 0001, etc.)")
            print("  Training uses WordNet IDs (n01440764, etc.)")
            print("  This is the dataset bug causing low accuracy!")
        else:
            only_train = set(train_classes) - set(val_classes)
            only_val = set(val_classes) - set(train_classes)
            if only_train:
                print(f"  Only in train: {list(only_train)[:5]}")
            if only_val:
                print(f"  Only in val: {list(only_val)[:5]}")
        
        return False


def main():
    parser = argparse.ArgumentParser(description='Prepare ImageNet dataset')
    parser.add_argument('dataset_dir', help='Input directory with tar files (or dataset to verify)')
    parser.add_argument('output_dir', nargs='?', help='Output directory (defaults to input)')
    parser.add_argument('--verify-only', action='store_true', help='Only verify existing dataset')
    
    args = parser.parse_args()
    
    if args.verify_only:
        # Just verify
        dataset_dir = Path(args.dataset_dir)
        print(f"Verifying dataset at: {dataset_dir}")
        
        if verify_dataset(dataset_dir):
            print("\n✓ Verification PASSED")
            return 0
        else:
            print("\n✗ Verification FAILED") 
            return 1
    
    # Prepare dataset
    input_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    
    print(f"Preparing ImageNet dataset")
    print(f"  Input: {input_dir}")
    print(f"  Output: {output_dir}")
    
    # Check for required files
    required = {
        'train': input_dir / "ILSVRC2012_img_train.tar",
        'val': input_dir / "ILSVRC2012_img_val.tar",
        'devkit': input_dir / "ILSVRC2012_devkit_t12.tar.gz"
    }
    
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        print(f"\nERROR: Missing required files: {missing}")
        return 1
    
    # Check for existing directories
    existing = [d for d in ['train', 'val'] if (output_dir / d).exists()]
    if existing:
        print(f"\nWARNING: These directories will be deleted: {existing}")
        if input("Continue? (yes/no): ").lower() not in ['yes', 'y']:
            print("Aborted.")
            return 0
        
        for d in existing:
            shutil.rmtree(output_dir / d)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract datasets
    train_classes = extract_train(required['train'], output_dir)
    
    if len(train_classes) != 1000:
        print(f"WARNING: Expected 1000 training classes, got {len(train_classes)}")
    
    val_classes = extract_val(required['val'], required['devkit'], output_dir, train_classes)
    
    # Verify
    print("\nVerifying dataset consistency...")
    if verify_dataset(output_dir):
        print("\n✓ Dataset prepared successfully!")
        print(f"  Train: {output_dir}/train")
        print(f"  Val: {output_dir}/val")
        return 0
    else:
        print("\n✗ Dataset preparation may have issues!")
        print("  Check the extraction process or try re-extracting")
        return 1


if __name__ == "__main__":
    exit(main())