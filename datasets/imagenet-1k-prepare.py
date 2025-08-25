#!/usr/bin/env python3
"""
ImageNet Dataset Preparation Script

Correctly extracts and organizes ImageNet using the ILSVRC2012 competition ordering.

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
        
        for i, member in enumerate(class_tars, 1):
            if i % 100 == 0:
                print(f"  Extracted {i}/{len(class_tars)} classes...")
            
            # Extract class tar
            tar.extract(member, train_dir)
            class_tar_path = train_dir / member.name
            
            # Create class directory
            class_name = member.name.replace('.tar', '')
            class_dir = train_dir / class_name
            class_dir.mkdir(exist_ok=True)
            
            # Extract images into class directory
            with tarfile.open(class_tar_path) as class_tar:
                class_tar.extractall(class_dir)
            
            os.remove(class_tar_path)
    
    print(f"  Extracted {len(class_tars)} training classes")


def extract_val(val_tar, devkit_tar, output_dir):
    """Extract validation data using ILSVRC2012 competition ordering"""
    print("Extracting validation data...")
    
    # Extract validation images to temp folder
    val_temp = output_dir / "val_temp"
    val_temp.mkdir(exist_ok=True)
    
    with tarfile.open(val_tar) as tar:
        tar.extractall(val_temp)
    
    val_images = sorted([f for f in os.listdir(val_temp) if f.endswith('.JPEG')])
    print(f"  Extracted {len(val_images)} validation images")
    
    # Extract devkit to get mappings
    print("  Extracting devkit for class mappings...")
    devkit_dir = output_dir / "devkit_temp"
    devkit_dir.mkdir(exist_ok=True)
    
    with tarfile.open(devkit_tar) as tar:
        tar.extractall(devkit_dir)
    
    # Find meta.mat for synset mapping
    meta_file = devkit_dir / "ILSVRC2012_devkit_t12" / "data" / "meta.mat"
    if not meta_file.exists():
        print("ERROR: meta.mat not found in devkit!")
        shutil.rmtree(val_temp)
        shutil.rmtree(devkit_dir)
        return False
    
    # Find ground truth file
    gt_file = devkit_dir / "ILSVRC2012_devkit_t12" / "data" / "ILSVRC2012_validation_ground_truth.txt"
    if not gt_file.exists():
        print("ERROR: validation ground truth not found in devkit!")
        shutil.rmtree(val_temp)
        shutil.rmtree(devkit_dir)
        return False
    
    # Load synset mapping using scipy
    try:
        import scipy.io
        meta = scipy.io.loadmat(meta_file)
    except ImportError:
        print("ERROR: scipy not installed. Please run: pip install scipy")
        shutil.rmtree(val_temp)
        shutil.rmtree(devkit_dir)
        return False
    
    # Build ILSVRC index to WordNet ID mapping
    synsets = meta['synsets']
    ilsvrc_to_wordnet = {}
    for i in range(1000):  # ImageNet-1K uses first 1000 classes
        ilsvrc_id = synsets[i][0][0][0][0]  # ILSVRC ID (1-1000)
        wordnet_id = synsets[i][0][1][0]    # WordNet ID (e.g., 'n01440764')
        ilsvrc_to_wordnet[ilsvrc_id] = wordnet_id
    
    # Read validation labels
    with open(gt_file, 'r') as f:
        val_labels = [int(line.strip()) for line in f]
    
    if len(val_labels) != len(val_images):
        print(f"ERROR: Image count ({len(val_images)}) != label count ({len(val_labels)})")
        shutil.rmtree(val_temp)
        shutil.rmtree(devkit_dir)
        return False
    
    # Create validation folders and move images
    print("  Organizing images into class folders...")
    val_dir = output_dir / "val"
    val_dir.mkdir(exist_ok=True)
    
    for img_file, label in zip(val_images, val_labels):
        wordnet_id = ilsvrc_to_wordnet[label]
        class_dir = val_dir / wordnet_id
        class_dir.mkdir(exist_ok=True)
        
        src = val_temp / img_file
        dst = class_dir / img_file
        shutil.move(str(src), str(dst))
    
    # Cleanup
    shutil.rmtree(val_temp)
    shutil.rmtree(devkit_dir)
    
    print(f"  Created {len(os.listdir(val_dir))} validation classes")
    return True


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
    
    if set(train_classes) == set(val_classes):
        print("  ✓ All classes present in both train and val")
        return True
    else:
        print("  ✗ ERROR: Class mismatch between train and val!")
        only_train = set(train_classes) - set(val_classes)
        only_val = set(val_classes) - set(train_classes)
        if only_train:
            print(f"    Only in train: {list(only_train)[:5]}")
        if only_val:
            print(f"    Only in val: {list(only_val)[:5]}")
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
    
    # Clean up existing directories
    for d in ['train', 'val', 'val_temp', 'devkit_temp']:
        if (output_dir / d).exists():
            print(f"  Removing existing {d} directory...")
            shutil.rmtree(output_dir / d)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract datasets
    extract_train(required['train'], output_dir)
    
    if not extract_val(required['val'], required['devkit'], output_dir):
        print("\n✗ Validation extraction failed!")
        return 1
    
    # Verify
    print("\nVerifying dataset consistency...")
    if verify_dataset(output_dir):
        print("\n✓ Dataset prepared successfully!")
        print(f"  Train: {output_dir}/train")
        print(f"  Val: {output_dir}/val")
        return 0
    else:
        print("\n✗ Dataset verification failed!")
        return 1


if __name__ == "__main__":
    exit(main())