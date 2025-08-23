#!/usr/bin/env python3
"""
Simple ImageNet Dataset Extraction Script

Just extracts and organizes the ImageNet files into train/val folders.
No PyTorch dependencies.

Usage:
    python prepare_imagenet.py ~/datasets/imagenet ~/datasets/imagenet_prepared
"""

import os
import tarfile
import sys
from pathlib import Path
import shutil


def extract_train(train_tar, output_dir):
    """Extract training data into class folders"""
    print("Extracting training data...")
    
    train_dir = output_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    
    with tarfile.open(train_tar) as tar:
        members = [m for m in tar if m.name.endswith('.tar')]
        total = len(members)
        
        for i, member in enumerate(members, 1):
            print(f"\rExtracting class tars: {i}/{total}", end='', flush=True)
            
            # Extract the class tar file
            tar.extract(member, train_dir)
            class_tar_path = train_dir / member.name
            
            # Create class directory
            class_name = member.name.replace('.tar', '')
            class_dir = train_dir / class_name
            class_dir.mkdir(exist_ok=True)
            
            # Extract images from class tar
            with tarfile.open(class_tar_path) as class_tar:
                class_tar.extractall(class_dir)
            
            # Remove the class tar file
            os.remove(class_tar_path)
        
        print()  # New line after progress


def extract_val(val_tar, devkit_tar, output_dir):
    """Extract validation data and organize by class"""
    print("Extracting validation data...")
    
    # Extract devkit to devkit directory
    devkit_dir = output_dir / "devkit"
    devkit_dir.mkdir(exist_ok=True)
    
    print("Extracting devkit...")
    with tarfile.open(devkit_tar) as tar:
        tar.extractall(devkit_dir)
    
    # Find validation ground truth file (search for it)
    val_labels_file = None
    for root, dirs, files in os.walk(devkit_dir):
        if "ILSVRC2012_validation_ground_truth.txt" in files:
            val_labels_file = Path(root) / "ILSVRC2012_validation_ground_truth.txt"
            break
    
    if not val_labels_file:
        print("Could not find ILSVRC2012_validation_ground_truth.txt in devkit!")
        sys.exit(1)
    
    print(f"Found validation labels at: {val_labels_file}")
    
    # Read validation ground truth
    val_labels = []
    with open(val_labels_file) as f:
        val_labels = [int(line.strip()) for line in f]
    
    # Extract validation images to temp folder
    val_temp = output_dir / "val_temp"
    val_temp.mkdir(exist_ok=True)
    
    with tarfile.open(val_tar) as tar:
        tar.extractall(val_temp)
    
    # Create val directory structure
    val_dir = output_dir / "val"
    val_dir.mkdir(exist_ok=True)
    
    # Get sorted list of validation images
    val_images = sorted([f for f in os.listdir(val_temp) if f.endswith('.JPEG')])
    
    # Move images to class folders
    for img_file, label in zip(val_images, val_labels):
        class_dir = val_dir / f"{label:04d}"
        class_dir.mkdir(exist_ok=True)
        
        src = val_temp / img_file
        dst = class_dir / img_file
        shutil.move(str(src), str(dst))
    
    # Clean up
    shutil.rmtree(val_temp)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python prepare_imagenet.py <input_dir> [output_dir]")
        sys.exit(1)
    
    input_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else input_dir
    
    print("preparing imagenet...")
    # Check if output directory exists and delete specific subdirs
    if output_dir.exists():
        for subdir in ['train', 'val', 'devkit']:
            subdir_path = output_dir / subdir
            if subdir_path.exists():
                print(f"Deleting existing {subdir} directory...")
                shutil.rmtree(subdir_path)
    
    train_tar = input_dir / "ILSVRC2012_img_train.tar"
    val_tar = input_dir / "ILSVRC2012_img_val.tar"
    devkit_tar = input_dir / "ILSVRC2012_devkit_t12.tar.gz"
    
    if not all(f.exists() for f in [train_tar, val_tar, devkit_tar]):
        print("Missing required files!")
        sys.exit(1)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    extract_train(train_tar, output_dir)
    extract_val(val_tar, devkit_tar, output_dir)
    
    print("Done! Dataset extracted to:")
    print(f"  Training: {output_dir}/train")
    print(f"  Validation: {output_dir}/val")