#!/usr/bin/env python3
"""
Example usage of handshake training visualization script.

This shows different ways to visualize your training results.
"""

import subprocess
import sys
from pathlib import Path


def visualize_from_log_file():
    """Example: Visualize from local log file."""
    cmd = [
        sys.executable, "-m", "lerobot.scripts.visualize_handshake_training",
        "--log_file", "./outputs/train.log",
        "--output_dir", "./training_plots"
    ]
    
    print("Visualizing from log file...")
    subprocess.run(cmd)


def visualize_from_checkpoint_dir():
    """Example: Visualize from checkpoint directory."""
    cmd = [
        sys.executable, "-m", "lerobot.scripts.visualize_handshake_training",
        "--checkpoint_dir", "./outputs/checkpoints/",
        "--output_dir", "./training_plots",
        "--plot_format", "pdf",  # Save as PDF
        "--plot_dpi", "300"
    ]
    
    print("Visualizing from checkpoint directory...")
    subprocess.run(cmd)


def quick_analysis(training_output_dir):
    """Quick analysis of a training run."""
    output_dir = Path(training_output_dir)
    
    # Try to find training data automatically
    if (output_dir / "train.log").exists():
        log_file = output_dir / "train.log"
        print(f"Found log file: {log_file}")
        
        cmd = [
            sys.executable, "-m", "lerobot.scripts.visualize_handshake_training",
            "--log_file", str(log_file),
            "--output_dir", str(output_dir / "plots")
        ]
        
    elif list(output_dir.glob("checkpoints*")):
        checkpoint_dir = list(output_dir.glob("checkpoints*"))[0]
        print(f"Found checkpoint dir: {checkpoint_dir}")
        
        cmd = [
            sys.executable, "-m", "lerobot.scripts.visualize_handshake_training", 
            "--checkpoint_dir", str(checkpoint_dir),
            "--output_dir", str(output_dir / "plots")
        ]
    else:
        print(f"No training data found in {output_dir}")
        return
    
    print("Running analysis...")
    subprocess.run(cmd)
    print(f"Results saved to {output_dir / 'plots'}")


if __name__ == "__main__":
    # Example usage
    if len(sys.argv) > 1:
        # Quick analysis of provided directory
        quick_analysis(sys.argv[1])
    else:
        print("Usage examples:")
        print("python example_visualize_training.py /path/to/training/output")
        print("")
        print("Or edit this script to use specific visualization functions:")
        print("- visualize_from_log_file()")
        print("- visualize_from_checkpoint_dir()") 