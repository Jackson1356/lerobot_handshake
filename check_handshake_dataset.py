#!/usr/bin/env python3
"""
Check handshake dataset to see what data is actually saved.

Example usage:
python -m check_handshake_dataset --dataset.repo_id=your-username/handshake_dataset
"""

import torch
from lerobot.common.datasets.factory import make_dataset
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig

def check_handshake_dataset(cfg: TrainPipelineConfig):
    """Check what handshake data is actually in the dataset."""
    
    print(f"Checking dataset: {cfg.dataset.repo_id}")
    
    try:
        # Load dataset
        dataset = make_dataset(cfg)
        print(f"✓ Dataset loaded successfully")
        print(f"  - Episodes: {dataset.num_episodes}")
        print(f"  - Frames: {dataset.num_frames}")
        
        # Check all features
        print(f"\nAll dataset features:")
        for key, feature in dataset.features.items():
            print(f"  - {key}: {feature}")
        
        # Check if handshake feature exists
        handshake_key = "observation.handshake"
        if handshake_key not in dataset.features:
            print(f"\n❌ Handshake feature '{handshake_key}' not found!")
            print("Available observation features:")
            for key in dataset.features:
                if key.startswith("observation"):
                    print(f"  - {key}: {dataset.features[key]}")
            return
        
        print(f"\n✓ Handshake feature found: {dataset.features[handshake_key]}")
        
        # Load a batch and analyze
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=8,
            shuffle=False,
            num_workers=0,
        )
        
        batch = next(iter(dataloader))
        print(f"\nBatch keys: {list(batch.keys())}")
        
        if handshake_key in batch:
            handshake_data = batch[handshake_key]
            print(f"\nHandshake data analysis:")
            print(f"  - Shape: {handshake_data.shape}")
            print(f"  - Data type: {handshake_data.dtype}")
            
            # Analyze the 4 components: [ready, confidence, pos_x, pos_y]
            ready_values = handshake_data[:, 0]
            confidence_values = handshake_data[:, 1]
            pos_x_values = handshake_data[:, 2]
            pos_y_values = handshake_data[:, 3]
            
            print(f"\nComponent analysis:")
            print(f"  - Ready (0/1): min={ready_values.min():.3f}, max={ready_values.max():.3f}, mean={ready_values.mean():.3f}")
            print(f"  - Confidence: min={confidence_values.min():.3f}, max={confidence_values.max():.3f}, mean={confidence_values.mean():.3f}")
            print(f"  - Position X: min={pos_x_values.min():.3f}, max={pos_x_values.max():.3f}, mean={pos_x_values.mean():.3f}")
            print(f"  - Position Y: min={pos_y_values.min():.3f}, max={pos_y_values.max():.3f}, mean={pos_y_values.mean():.3f}")
            
            # Check for valid positions (non-negative)
            valid_positions = (pos_x_values >= 0) & (pos_y_values >= 0)
            num_valid = valid_positions.sum().item()
            total_samples = len(valid_positions)
            
            print(f"\nValid handshake positions:")
            print(f"  - Valid: {num_valid}/{total_samples} ({num_valid/total_samples*100:.1f}%)")
            
            if num_valid > 0:
                valid_data = handshake_data[valid_positions]
                print(f"  - Valid ready rate: {valid_data[:, 0].mean():.3f}")
                print(f"  - Valid confidence: {valid_data[:, 1].mean():.3f}")
                print(f"  - Valid pos_x range: {valid_data[:, 2].min():.1f} to {valid_data[:, 2].max():.1f}")
                print(f"  - Valid pos_y range: {valid_data[:, 3].min():.1f} to {valid_data[:, 3].max():.1f}")
                
                # Show a few sample values
                print(f"\nSample valid handshake data:")
                for i in range(min(3, len(valid_data))):
                    print(f"  Sample {i}: ready={valid_data[i, 0]:.1f}, conf={valid_data[i, 1]:.3f}, pos=({valid_data[i, 2]:.1f}, {valid_data[i, 3]:.1f})")
            else:
                print("  - No valid handshake positions found!")
                print("  - This suggests an issue with the recording process")
                
        else:
            print(f"\n❌ Handshake key '{handshake_key}' not found in batch!")
            print("Available keys:", list(batch.keys()))
            
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("Make sure the dataset path is correct and the dataset exists.")

@parser.wrap()
def main(cfg: TrainPipelineConfig):
    """Main function to check handshake dataset."""
    check_handshake_dataset(cfg)

if __name__ == "__main__":
    main() 