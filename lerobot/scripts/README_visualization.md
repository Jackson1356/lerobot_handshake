# Handshake Training Visualization Guide

This guide explains how to visualize your handshake training results using the provided visualization tools.

## Quick Start

### Option 1: Auto-detect training data
```bash
python lerobot/scripts/example_visualize_training.py /path/to/training/output
```

### Option 2: From log file
```bash
python -m lerobot.scripts.visualize_handshake_training \
    --log_file ./outputs/train.log \
    --output_dir ./training_plots
```

### Option 3: From checkpoint directory
```bash
python -m lerobot.scripts.visualize_handshake_training \
    --checkpoint_dir ./outputs/checkpoints \
    --output_dir ./training_plots
```

## Generated Visualizations

### 1. Training Overview Dashboard (`training_overview.png`)
A comprehensive dashboard showing:
- **Training Loss**: Loss curve over training steps
- **Hand Position Distribution**: Scatter plot of where people extend hands
- **Hand Position Variance**: How much hand positions vary over time
- **Hand Position Range**: Training diversity (wider range = more robust training)
- **Learning Rate**: LR schedule over time
- **Gradient Norm**: Gradient magnitude tracking
- **Training Timing**: Update and data loading times

### 2. Handshake Analysis (`handshake_analysis.png`)
Detailed handshake-specific analysis:
- **Hand Position Heatmap**: 2D density map of hand positions
- **Detection Quality**: % of valid hand detections over time
- **Training Diversity**: X/Y position ranges over time
- **Position Statistics**: Box plots of hand position distributions

### 3. Training Report (`training_report.json`)
JSON summary with key metrics:
```json
{
  "final_loss": 0.0245,
  "min_loss": 0.0201,
  "loss_improvement": 0.1234,
  "avg_hand_x": 320.5,
  "hand_x_std": 45.2,
  "avg_hand_y": 240.1,
  "hand_y_std": 38.7,
  "avg_detection_quality": 94.8,
  "avg_x_diversity": 180.3,
  "avg_y_diversity": 160.7
}
```

## Understanding the Metrics

### Hand Position Metrics (Key for Handshake Success)
| Metric | What It Means | Good Values |
|--------|---------------|-------------|
| `avg_target_hand_x/y` | Where people typically extend hands | Near image center (320, 240 for 640x480) |
| `hand_position_variance_x/y` | Consistency of hand positions | Moderate values (not too low/high) |
| `hand_x/y_range` | Training diversity | Wide ranges (>100 pixels) for robustness |
| `valid_hand_position_rate` | Detection quality | >90% for good data quality |

### Training Health Indicators
| Metric | What It Means | Good Signs |
|--------|---------------|------------|
| Loss curve | Learning progress | Steady decrease, converging |
| Gradient norm | Training stability | Stable, not exploding |
| Learning rate | Optimization schedule | Following expected schedule |

## Tips for Analysis

### 🟢 Good Training Signs
- **Steadily decreasing loss** without major oscillations
- **High detection quality** (>90% valid positions)
- **Wide hand position ranges** (diverse training data)
- **Stable gradient norms** (not exploding)
- **Hand positions distributed around image center**

### 🟡 Warning Signs
- **Loss plateauing early** → May need more training or different LR
- **Low hand position diversity** → Need more varied demonstrations
- **Poor detection quality** → Check handshake detection system
- **Hand positions at image edges** → People extending hands too far

### 🔴 Problem Signs
- **Loss increasing or oscillating wildly** → Learning rate too high
- **Gradient norm exploding** → Gradient clipping needed
- **Very low detection rates** → Handshake detection broken
- **No hand position diversity** → Data collection issues

## Command Line Options

```bash
python -m lerobot.scripts.visualize_handshake_training \
    --log_file ./train.log \              # Path to training log
    --output_dir ./plots \                # Where to save plots
    --plot_format png \                   # png, pdf, svg
    --plot_dpi 300 \                      # Resolution
    --style whitegrid \                   # Plot style
    --figsize_large 15,10                 # Figure size for overview
```

## Integration with Training

Add this to your training script to auto-generate plots:

```bash
# Run training
python -m lerobot.scripts.train_handshake \
    --dataset.repo_id=your_username/handshake_dataset \
    --policy.type=act \
    --output_dir=./outputs

# Auto-visualize results
python lerobot/scripts/example_visualize_training.py ./outputs
```

## Dependencies

Make sure you have installed:
```bash
pip install matplotlib seaborn pandas numpy
``` 