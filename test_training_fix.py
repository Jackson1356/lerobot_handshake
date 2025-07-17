#!/usr/bin/env python3
"""
Quick test to verify the training fix works.
"""

import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.scripts.train_handshake import compute_handshake_metrics
from torch.utils.data import DataLoader


def test_training_fix():
    """Test that the training fix works."""
    print("🧪 TESTING TRAINING FIX")
    print("=" * 60)
    
    # Load dataset directly (like the fix)
    dataset = LeRobotDataset("HS_01/handshake_dataset")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
    )
    
    # Test a few batches
    for i, batch in enumerate(dataloader):
        print(f"\nBatch {i}:")
        print(f"   Batch keys: {list(batch.keys())}")
        
        if 'observation.handshake' in batch:
            handshake_data = batch['observation.handshake']
            print(f"   observation.handshake: shape={handshake_data.shape}")
            
            # Test metrics computation
            metrics = compute_handshake_metrics(batch)
            print(f"   Metrics: {metrics}")
            
            # Check if metrics are non-zero
            if metrics['valid_hand_position_rate'] > 0:
                print("   ✅ SUCCESS: Non-zero handshake metrics!")
                break
            else:
                print("   ❌ Still getting zero metrics")
        else:
            print("   ❌ observation.handshake missing from batch")
        
        if i >= 2:  # Test first 3 batches
            break
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("If you see 'SUCCESS: Non-zero handshake metrics!' above,")
    print("then the fix works and you should re-run your training.")


if __name__ == "__main__":
    test_training_fix() 