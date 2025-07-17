#!/usr/bin/env python3
"""
Script to check handshake dataset and diagnose why training metrics show 0s.

This script will:
1. Load your handshake dataset directly
2. Check if handshake data is properly recorded
3. Show the structure of the dataset
4. Verify the data format matches what the training script expects
"""

import logging
import torch
from pathlib import Path
from pprint import pformat

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def check_dataset_structure(dataset):
    """Check the structure of the dataset and print key information."""
    print("=" * 60)
    print("DATASET STRUCTURE ANALYSIS")
    print("=" * 60)
    
    print(f"Dataset features:")
    for key, feature in dataset.features.items():
        print(f"  {key}: {feature}")
    
    print(f"\nDataset metadata:")
    print(f"  FPS: {dataset.fps}")
    print(f"  Number of episodes: {dataset.num_episodes}")
    print(f"  Number of frames: {dataset.num_frames}")
    
    # Check if handshake features exist
    handshake_features = [k for k in dataset.features.keys() if 'handshake' in k.lower()]
    print(f"\nHandshake-related features found: {handshake_features}")
    
    return handshake_features


def check_sample_data(dataset, num_samples=5):
    """Check sample data from the dataset."""
    print("\n" + "=" * 60)
    print("SAMPLE DATA ANALYSIS")
    print("=" * 60)
    
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        print(f"\nSample {i}:")
        
        # Check for handshake data
        handshake_keys = [k for k in sample.keys() if 'handshake' in k.lower()]
        print(f"  Handshake keys: {handshake_keys}")
        
        for key in handshake_keys:
            if key in sample:
                data = sample[key]
                print(f"  {key}: shape={data.shape}, dtype={data.dtype}")
                if hasattr(data, 'tolist'):
                    print(f"    Values: {data.tolist()}")
                else:
                    print(f"    Values: {data}")
        
        # Check observation structure
        obs_keys = [k for k in sample.keys() if k.startswith('observation')]
        print(f"  Observation keys: {obs_keys}")
        
        # Check if we have the expected observation.handshake structure
        if 'observation.handshake' in sample:
            handshake_data = sample['observation.handshake']
            print(f"  observation.handshake: shape={handshake_data.shape}, dtype={handshake_data.dtype}")
            print(f"    Values: {handshake_data.tolist()}")
        else:
            print("  ❌ observation.handshake NOT FOUND")
            
            # Check for individual handshake fields
            individual_fields = ['handshake_ready', 'handshake_confidence', 'hand_position_x', 'hand_position_y']
            found_fields = []
            for field in individual_fields:
                if field in sample:
                    found_fields.append(field)
                    print(f"    {field}: {sample[field]}")
            
            if found_fields:
                print(f"  Found individual handshake fields: {found_fields}")
            else:
                print("  ❌ No handshake fields found at all")


def check_training_compatibility(dataset):
    """Check if the dataset is compatible with the training script."""
    print("\n" + "=" * 60)
    print("TRAINING COMPATIBILITY CHECK")
    print("=" * 60)
    
    # Check if observation.handshake exists
    if 'observation.handshake' in dataset.features:
        print("✅ observation.handshake feature found in dataset")
        feature = dataset.features['observation.handshake']
        print(f"   Shape: {feature['shape']}")
        print(f"   Names: {feature['names']}")
        
        # Check if the shape matches what training script expects
        if feature['shape'] == (4,):
            print("✅ Shape matches expected (4,) for [ready, confidence, pos_x, pos_y]")
        else:
            print(f"❌ Shape mismatch: expected (4,), got {feature['shape']}")
    else:
        print("❌ observation.handshake feature NOT found in dataset")
        
        # Check for individual fields that might need to be combined
        individual_fields = ['handshake_ready', 'handshake_confidence', 'hand_position_x', 'hand_position_y']
        found_individual = []
        for field in individual_fields:
            if field in dataset.features:
                found_individual.append(field)
        
        if found_individual:
            print(f"   Found individual fields: {found_individual}")
            print("   These need to be combined into observation.handshake for training")
        else:
            print("   No handshake fields found at all")


def simulate_training_batch(dataset, batch_size=8):
    """Simulate what the training script sees in a batch."""
    print("\n" + "=" * 60)
    print("TRAINING BATCH SIMULATION")
    print("=" * 60)
    
    # Create a simple batch
    batch = {}
    
    # Simulate the training script's data loading
    if 'observation.handshake' in dataset.features:
        # Collect handshake data from multiple samples
        handshake_data = []
        for i in range(min(batch_size, len(dataset))):
            sample = dataset[i]
            if 'observation.handshake' in sample:
                handshake_data.append(sample['observation.handshake'])
        
        if handshake_data:
            batch['observation.handshake'] = torch.stack(handshake_data)
            print(f"✅ Created batch with observation.handshake: shape={batch['observation.handshake'].shape}")
            
            # Simulate the training metrics computation
            handshake_data = batch['observation.handshake']
            hand_position_x = handshake_data[:, 2]  # Target X position
            hand_position_y = handshake_data[:, 3]  # Target Y position
            
            # Filter valid hand positions (detection succeeded)
            valid_positions = (hand_position_x >= 0) & (hand_position_y >= 0)
            valid_rate = valid_positions.float().mean().item() * 100
            
            print(f"   Valid hand position rate: {valid_rate:.1f}%")
            print(f"   Hand position X range: {hand_position_x.min():.2f} to {hand_position_x.max():.2f}")
            print(f"   Hand position Y range: {hand_position_y.min():.2f} to {hand_position_y.max():.2f}")
            
            if valid_positions.any():
                valid_x = hand_position_x[valid_positions]
                valid_y = hand_position_y[valid_positions]
                print(f"   Valid positions - X mean: {valid_x.mean():.2f}, Y mean: {valid_y.mean():.2f}")
            else:
                print("   ❌ No valid hand positions found - this explains the 0s in training!")
        else:
            print("❌ No observation.handshake data found in samples")
    else:
        print("❌ observation.handshake feature not available for batch simulation")


def main():
    """Main function to check the handshake dataset."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Check handshake dataset structure")
    parser.add_argument("--repo_id", required=True, help="Dataset repo ID (e.g., username/handshake_dataset)")
    parser.add_argument("--root", help="Local dataset root path (optional)")
    
    args = parser.parse_args()
    
    print("🔍 HANDSHAKE DATASET DIAGNOSTIC TOOL")
    print("=" * 60)
    
    # Load dataset directly
    print("Loading dataset...")
    try:
        dataset = LeRobotDataset(args.repo_id, root=args.root)
        print(f"✅ Successfully loaded dataset: {args.repo_id}")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return
    
    # Check structure
    handshake_features = check_dataset_structure(dataset)
    
    # Check sample data
    check_sample_data(dataset)
    
    # Check training compatibility
    check_training_compatibility(dataset)
    
    # Simulate training batch
    simulate_training_batch(dataset)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if 'observation.handshake' in dataset.features:
        print("✅ Dataset has observation.handshake feature")
        print("   If training still shows 0s, check if:")
        print("   1. Handshake detection is working during recording")
        print("   2. Hand positions are being detected (not -1 values)")
        print("   3. Dataset has enough valid handshake episodes")
    else:
        print("❌ Dataset missing observation.handshake feature")
        print("   This is why training shows 0s!")
        print("   The dataset was likely recorded without handshake detection")
        print("   or the handshake data wasn't properly integrated")
    
    print("\nTo fix this issue:")
    print("1. Re-record your dataset using record_handshake.py")
    print("2. Ensure handshake detection is working during recording")
    print("3. Verify that hand positions are being detected (not -1 values)")


if __name__ == "__main__":
    main() 