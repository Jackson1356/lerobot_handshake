#!/usr/bin/env python3
"""
Test script to verify what the training script sees in a batch.
This will help diagnose why training metrics show 0s despite valid dataset data.
"""

import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.scripts.train_handshake import compute_handshake_metrics


def test_training_batch():
    """Test what the training script sees in a batch."""
    print("🧪 TESTING TRAINING BATCH PROCESSING")
    print("=" * 60)
    
    # Load dataset
    dataset = LeRobotDataset("HS_01/handshake_dataset")
    
    # Create a batch like the training script would
    batch_size = 8
    batch = {}
    
    # Collect handshake data from multiple samples
    handshake_data = []
    for i in range(batch_size):
        sample = dataset[i]
        if 'observation.handshake' in sample:
            handshake_data.append(sample['observation.handshake'])
    
    if handshake_data:
        batch['observation.handshake'] = torch.stack(handshake_data)
        print(f"✅ Created batch with observation.handshake: shape={batch['observation.handshake'].shape}")
        print(f"   Batch data: {batch['observation.handshake']}")
        
        # Test the exact same computation as training script
        metrics = compute_handshake_metrics(batch)
        print(f"\n📊 Training script computed metrics:")
        for key, value in metrics.items():
            print(f"   {key}: {value}")
        
        # Additional debugging
        handshake_data = batch['observation.handshake']
        hand_position_x = handshake_data[:, 2]
        hand_position_y = handshake_data[:, 3]
        
        print(f"\n🔍 Detailed analysis:")
        print(f"   Hand position X: {hand_position_x.tolist()}")
        print(f"   Hand position Y: {hand_position_y.tolist()}")
        
        valid_positions = (hand_position_x >= 0) & (hand_position_y >= 0)
        print(f"   Valid positions mask: {valid_positions.tolist()}")
        print(f"   Number of valid positions: {valid_positions.sum().item()}")
        
        if valid_positions.any():
            valid_x = hand_position_x[valid_positions]
            valid_y = hand_position_y[valid_positions]
            print(f"   Valid X positions: {valid_x.tolist()}")
            print(f"   Valid Y positions: {valid_y.tolist()}")
            print(f"   X mean: {valid_x.mean():.2f}")
            print(f"   Y mean: {valid_y.mean():.2f}")
            print(f"   X variance: {valid_x.var():.2f}")
            print(f"   Y variance: {valid_y.var():.2f}")
            print(f"   X range: {valid_x.max() - valid_x.min():.2f}")
            print(f"   Y range: {valid_y.max() - valid_y.min():.2f}")
        else:
            print("   ❌ No valid positions found!")
    else:
        print("❌ No observation.handshake data found in samples")


def test_dataloader_batch():
    """Test what the actual dataloader produces."""
    print("\n" + "=" * 60)
    print("TESTING DATALOADER BATCH")
    print("=" * 60)
    
    from torch.utils.data import DataLoader
    
    # Load dataset
    dataset = LeRobotDataset("HS_01/handshake_dataset")
    
    # Create dataloader like training script
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,  # Use 0 for debugging
    )
    
    # Get first batch
    batch = next(iter(dataloader))
    
    print(f"Batch keys: {list(batch.keys())}")
    
    if 'observation.handshake' in batch:
        handshake_data = batch['observation.handshake']
        print(f"✅ observation.handshake in batch: shape={handshake_data.shape}")
        print(f"   Data: {handshake_data}")
        
        # Test metrics computation
        metrics = compute_handshake_metrics(batch)
        print(f"\n📊 Metrics from dataloader batch:")
        for key, value in metrics.items():
            print(f"   {key}: {value}")
    else:
        print("❌ observation.handshake NOT in dataloader batch!")
        print(f"Available keys: {list(batch.keys())}")


if __name__ == "__main__":
    test_training_batch()
    test_dataloader_batch() 