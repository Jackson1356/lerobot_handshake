"""
Visualization script for handshake training results.

This script creates comprehensive plots for training progress including:
- Training loss and metrics over time
- Hand position distribution analysis
- Training data quality metrics
- Learning progress visualization

Example usage:

```bash
# From local logs  
python -m lerobot.scripts.visualize_handshake_training \
    --log_file ./outputs/train_log.txt \
    --output_dir ./training_plots

# From checkpoint directory
python -m lerobot.scripts.visualize_handshake_training \
    --checkpoint_dir ./outputs/checkpoints \
    --output_dir ./training_plots
```
"""

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec

from lerobot.common.utils.utils import init_logging
from lerobot.configs import parser


@dataclass
class TrainingVisualizationConfig:
    # Data source (choose one)
    log_file: Optional[str] = None
    checkpoint_dir: Optional[str] = None
    
    # Output settings
    output_dir: str = "./training_plots"
    plot_format: str = "png"
    plot_dpi: int = 300
    
    # Plot settings
    figsize_large: Tuple[int, int] = (15, 10)
    figsize_medium: Tuple[int, int] = (12, 8)
    figsize_small: Tuple[int, int] = (8, 6)
    
    # Style settings
    style: str = "whitegrid"
    palette: str = "Set2"


class HandshakeTrainingVisualizer:
    """Visualizer for handshake training results."""
    
    def __init__(self, config: TrainingVisualizationConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_style(config.style)
        sns.set_palette(config.palette)
        
        self.training_data = None
        
    def load_data(self) -> pd.DataFrame:
        """Load training data from specified source."""
        if self.config.log_file:
            return self._load_from_log_file()
        elif self.config.checkpoint_dir:
            return self._load_from_checkpoint_dir()
        else:
            raise ValueError("Must specify either log_file or checkpoint_dir")
    
    def _load_from_log_file(self) -> pd.DataFrame:
        """Load data from log file."""
        logging.info(f"Loading data from log file: {self.config.log_file}")
        
        log_path = Path(self.config.log_file)
        if not log_path.exists():
            raise FileNotFoundError(f"Log file not found: {log_path}")
        
        # Parse log file for training metrics
        training_data = []
        
        with open(log_path, 'r') as f:
            for line in f:
                # Look for training metric lines
                # Expected format: step info with metrics
                if 'loss:' in line and 'step' in line:
                    metrics = self._parse_log_line(line)
                    if metrics:
                        training_data.append(metrics)
        
        if not training_data:
            raise ValueError("No training data found in log file")
        
        return pd.DataFrame(training_data)
    
    def _parse_log_line(self, line: str) -> Optional[Dict]:
        """Parse a single log line for metrics."""
        try:
            # Extract step number
            step_match = re.search(r'step[:\s]+(\d+)', line)
            if not step_match:
                return None
            
            step = int(step_match.group(1))
            metrics = {'step': step}
            
            # Extract numeric metrics using regex
            metric_patterns = {
                'loss': r'loss[:\s]+([\d\.]+)',
                'grad_norm': r'grdn[:\s]+([\d\.]+)',
                'lr': r'lr[:\s]+([\d\.e\-\+]+)',
                'update_s': r'updt_s[:\s]+([\d\.]+)',
                'dataloading_s': r'data_s[:\s]+([\d\.]+)',
                'valid_hand_position_rate': r'valid_pos[:\s]+([\d\.]+)',
                'avg_target_hand_x': r'tgt_x[:\s]+([\d\.]+)',
                'avg_target_hand_y': r'tgt_y[:\s]+([\d\.]+)',
                'hand_position_variance_x': r'var_x[:\s]+([\d\.]+)',
                'hand_position_variance_y': r'var_y[:\s]+([\d\.]+)',
                'hand_x_range': r'rng_x[:\s]+([\d\.]+)',
                'hand_y_range': r'rng_y[:\s]+([\d\.]+)',
            }
            
            for metric_name, pattern in metric_patterns.items():
                match = re.search(pattern, line)
                if match:
                    metrics[metric_name] = float(match.group(1))
            
            return metrics if len(metrics) > 1 else None
            
        except Exception as e:
            logging.warning(f"Failed to parse log line: {e}")
            return None
    
    def _load_from_checkpoint_dir(self) -> pd.DataFrame:
        """Load data from checkpoint directory (if training logs are saved there)."""
        logging.info(f"Loading data from checkpoint directory: {self.config.checkpoint_dir}")
        
        # Look for training logs in checkpoint directory
        checkpoint_path = Path(self.config.checkpoint_dir)
        log_files = list(checkpoint_path.glob("*.log")) + list(checkpoint_path.glob("**/train.log"))
        
        if not log_files:
            raise FileNotFoundError(f"No log files found in {checkpoint_path}")
        
        # Use the most recent log file
        log_file = max(log_files, key=lambda x: x.stat().st_mtime)
        logging.info(f"Using log file: {log_file}")
        
        # Parse the log file
        return self._load_from_log_file_path(log_file)
    
    def _load_from_log_file_path(self, log_path: Path) -> pd.DataFrame:
        """Helper to load from a specific log file path."""
        old_log_file = self.config.log_file
        self.config.log_file = str(log_path)
        try:
            return self._load_from_log_file()
        finally:
            self.config.log_file = old_log_file
    
    def create_training_overview(self) -> None:
        """Create comprehensive training overview plot."""
        if self.training_data is None:
            raise ValueError("No training data loaded. Call load_data() first.")
        
        fig = plt.figure(figsize=self.config.figsize_large)
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Main loss plot
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_loss_curve(ax1)
        
        # Hand position metrics
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_hand_position_distribution(ax2)
        
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_hand_position_variance(ax3)
        
        ax4 = fig.add_subplot(gs[1, 2])
        self._plot_hand_position_range(ax4)
        
        # Training dynamics
        ax5 = fig.add_subplot(gs[2, 0])
        self._plot_learning_rate(ax5)
        
        ax6 = fig.add_subplot(gs[2, 1])
        self._plot_gradient_norm(ax6)
        
        ax7 = fig.add_subplot(gs[2, 2])
        self._plot_training_timing(ax7)
        
        plt.suptitle('Handshake Training Overview', fontsize=16, fontweight='bold')
        
        # Save plot
        output_file = self.output_dir / f"training_overview.{self.config.plot_format}"
        plt.savefig(output_file, dpi=self.config.plot_dpi, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Saved training overview to {output_file}")
    
    def _plot_loss_curve(self, ax):
        """Plot training loss curve."""
        if 'loss' in self.training_data.columns:
            ax.plot(self.training_data['step'], self.training_data['loss'], 
                   linewidth=2, label='Training Loss')
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Loss')
            ax.set_title('Training Loss Over Time')
            ax.grid(True, alpha=0.3)
            ax.legend()
    
    def _plot_hand_position_distribution(self, ax):
        """Plot hand position center distribution."""
        if 'avg_target_hand_x' in self.training_data.columns and 'avg_target_hand_y' in self.training_data.columns:
            x_data = self.training_data['avg_target_hand_x'].dropna()
            y_data = self.training_data['avg_target_hand_y'].dropna()
            
            if len(x_data) > 0 and len(y_data) > 0:
                ax.scatter(x_data, y_data, alpha=0.6, s=30)
                ax.set_xlabel('Hand X Position (pixels)')
                ax.set_ylabel('Hand Y Position (pixels)')
                ax.set_title('Hand Position Distribution')
                ax.grid(True, alpha=0.3)
    
    def _plot_hand_position_variance(self, ax):
        """Plot hand position variance over time."""
        variance_cols = ['hand_position_variance_x', 'hand_position_variance_y']
        available_cols = [col for col in variance_cols if col in self.training_data.columns]
        
        for col in available_cols:
            data = self.training_data[col].dropna()
            if len(data) > 0:
                label = 'X Variance' if 'x' in col else 'Y Variance'
                ax.plot(self.training_data['step'], data, label=label, linewidth=2)
        
        if available_cols:
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Position Variance')
            ax.set_title('Hand Position Variance')
            ax.grid(True, alpha=0.3)
            ax.legend()
    
    def _plot_hand_position_range(self, ax):
        """Plot hand position range over time."""
        range_cols = ['hand_x_range', 'hand_y_range']
        available_cols = [col for col in range_cols if col in self.training_data.columns]
        
        for col in available_cols:
            data = self.training_data[col].dropna()
            if len(data) > 0:
                label = 'X Range' if 'x' in col else 'Y Range'
                ax.plot(self.training_data['step'], data, label=label, linewidth=2)
        
        if available_cols:
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Position Range (pixels)')
            ax.set_title('Hand Position Range (Training Diversity)')
            ax.grid(True, alpha=0.3)
            ax.legend()
    
    def _plot_learning_rate(self, ax):
        """Plot learning rate over time."""
        if 'lr' in self.training_data.columns:
            lr_data = self.training_data['lr'].dropna()
            if len(lr_data) > 0:
                ax.semilogy(self.training_data['step'], lr_data, linewidth=2, color='orange')
                ax.set_xlabel('Training Step')
                ax.set_ylabel('Learning Rate')
                ax.set_title('Learning Rate Schedule')
                ax.grid(True, alpha=0.3)
    
    def _plot_gradient_norm(self, ax):
        """Plot gradient norm over time."""
        if 'grad_norm' in self.training_data.columns:
            grad_data = self.training_data['grad_norm'].dropna()
            if len(grad_data) > 0:
                ax.plot(self.training_data['step'], grad_data, linewidth=2, color='red')
                ax.set_xlabel('Training Step')
                ax.set_ylabel('Gradient Norm')
                ax.set_title('Gradient Norm')
                ax.grid(True, alpha=0.3)
    
    def _plot_training_timing(self, ax):
        """Plot training timing metrics."""
        timing_cols = ['update_s', 'dataloading_s']
        available_cols = [col for col in timing_cols if col in self.training_data.columns]
        
        for col in available_cols:
            data = self.training_data[col].dropna()
            if len(data) > 0:
                label = 'Update Time' if 'update' in col else 'Data Loading Time'
                ax.plot(self.training_data['step'], data, label=label, linewidth=2)
        
        if available_cols:
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Time (seconds)')
            ax.set_title('Training Timing')
            ax.grid(True, alpha=0.3)
            ax.legend()
    
    def create_handshake_analysis(self) -> None:
        """Create detailed handshake-specific analysis plots."""
        fig, axes = plt.subplots(2, 2, figsize=self.config.figsize_medium)
        fig.suptitle('Handshake Data Analysis', fontsize=14, fontweight='bold')
        
        # Hand position heatmap
        self._plot_hand_position_heatmap(axes[0, 0])
        
        # Hand position quality over time
        self._plot_hand_position_quality(axes[0, 1])
        
        # Training data diversity
        self._plot_training_diversity(axes[1, 0])
        
        # Hand position statistics summary
        self._plot_position_statistics(axes[1, 1])
        
        plt.tight_layout()
        
        # Save plot
        output_file = self.output_dir / f"handshake_analysis.{self.config.plot_format}"
        plt.savefig(output_file, dpi=self.config.plot_dpi, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Saved handshake analysis to {output_file}")
    
    def _plot_hand_position_heatmap(self, ax):
        """Plot 2D heatmap of hand positions."""
        if 'avg_target_hand_x' in self.training_data.columns and 'avg_target_hand_y' in self.training_data.columns:
            x_data = self.training_data['avg_target_hand_x'].dropna()
            y_data = self.training_data['avg_target_hand_y'].dropna()
            
            if len(x_data) > 10 and len(y_data) > 10:
                # Create 2D histogram
                hist, xedges, yedges = np.histogram2d(x_data, y_data, bins=20)
                extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
                
                im = ax.imshow(hist.T, extent=extent, origin='lower', cmap='viridis', alpha=0.7)
                ax.set_xlabel('Hand X Position (pixels)')
                ax.set_ylabel('Hand Y Position (pixels)')
                ax.set_title('Hand Position Heatmap')
                plt.colorbar(im, ax=ax, label='Frequency')
    
    def _plot_hand_position_quality(self, ax):
        """Plot hand position detection quality over time."""
        if 'valid_hand_position_rate' in self.training_data.columns:
            quality_data = self.training_data['valid_hand_position_rate'].dropna()
            if len(quality_data) > 0:
                ax.plot(self.training_data['step'], quality_data, linewidth=2, color='green')
                ax.set_xlabel('Training Step')
                ax.set_ylabel('Valid Position Rate (%)')
                ax.set_title('Hand Detection Quality')
                ax.grid(True, alpha=0.3)
                ax.set_ylim(0, 100)
    
    def _plot_training_diversity(self, ax):
        """Plot training data diversity metrics."""
        range_cols = ['hand_x_range', 'hand_y_range']
        colors = ['blue', 'red']
        
        for i, col in enumerate(range_cols):
            if col in self.training_data.columns:
                data = self.training_data[col].dropna()
                if len(data) > 0:
                    label = 'X Range' if 'x' in col else 'Y Range'
                    ax.plot(self.training_data['step'], data, 
                           label=label, linewidth=2, color=colors[i])
        
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Position Range (pixels)')
        ax.set_title('Training Data Diversity')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def _plot_position_statistics(self, ax):
        """Plot summary statistics of hand positions."""
        stats_data = []
        labels = []
        
        for col_prefix, label in [('avg_target_hand_x', 'X Position'), 
                                 ('avg_target_hand_y', 'Y Position')]:
            if col_prefix in self.training_data.columns:
                data = self.training_data[col_prefix].dropna()
                if len(data) > 0:
                    stats_data.append(data)
                    labels.append(label)
        
        if stats_data:
            bp = ax.boxplot(stats_data, labels=labels, patch_artist=True)
            
            # Color the boxes
            colors = ['lightblue', 'lightcoral']
            for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                patch.set_facecolor(color)
            
            ax.set_ylabel('Position (pixels)')
            ax.set_title('Hand Position Statistics')
            ax.grid(True, alpha=0.3)
    
    def generate_training_report(self) -> None:
        """Generate a comprehensive training report."""
        if self.training_data is None:
            raise ValueError("No training data loaded. Call load_data() first.")
        
        # Create summary statistics
        report_data = {}
        
        # Training progress
        if 'loss' in self.training_data.columns:
            loss_data = self.training_data['loss'].dropna()
            if len(loss_data) > 0:
                report_data['final_loss'] = loss_data.iloc[-1]
                report_data['min_loss'] = loss_data.min()
                report_data['loss_improvement'] = loss_data.iloc[0] - loss_data.iloc[-1] if len(loss_data) > 1 else 0
        
        # Hand position analysis
        if 'avg_target_hand_x' in self.training_data.columns:
            x_data = self.training_data['avg_target_hand_x'].dropna()
            if len(x_data) > 0:
                report_data['avg_hand_x'] = x_data.mean()
                report_data['hand_x_std'] = x_data.std()
        
        if 'avg_target_hand_y' in self.training_data.columns:
            y_data = self.training_data['avg_target_hand_y'].dropna()
            if len(y_data) > 0:
                report_data['avg_hand_y'] = y_data.mean()
                report_data['hand_y_std'] = y_data.std()
        
        # Data quality
        if 'valid_hand_position_rate' in self.training_data.columns:
            quality_data = self.training_data['valid_hand_position_rate'].dropna()
            if len(quality_data) > 0:
                report_data['avg_detection_quality'] = quality_data.mean()
        
        # Training diversity
        if 'hand_x_range' in self.training_data.columns and 'hand_y_range' in self.training_data.columns:
            x_range = self.training_data['hand_x_range'].dropna()
            y_range = self.training_data['hand_y_range'].dropna()
            if len(x_range) > 0 and len(y_range) > 0:
                report_data['avg_x_diversity'] = x_range.mean()
                report_data['avg_y_diversity'] = y_range.mean()
        
        # Save report
        report_file = self.output_dir / "training_report.json"
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logging.info(f"Saved training report to {report_file}")
        
        # Print summary
        self._print_training_summary(report_data)
    
    def _print_training_summary(self, report_data: Dict):
        """Print training summary to console."""
        print("\n" + "="*50)
        print("HANDSHAKE TRAINING SUMMARY")
        print("="*50)
        
        if 'final_loss' in report_data:
            print(f"Final Loss: {report_data['final_loss']:.4f}")
            print(f"Minimum Loss: {report_data['min_loss']:.4f}")
            print(f"Loss Improvement: {report_data['loss_improvement']:.4f}")
        
        print("\nHand Position Analysis:")
        if 'avg_hand_x' in report_data:
            print(f"  Average Hand X: {report_data['avg_hand_x']:.1f} ± {report_data.get('hand_x_std', 0):.1f} pixels")
        if 'avg_hand_y' in report_data:
            print(f"  Average Hand Y: {report_data['avg_hand_y']:.1f} ± {report_data.get('hand_y_std', 0):.1f} pixels")
        
        if 'avg_detection_quality' in report_data:
            print(f"\nData Quality:")
            print(f"  Average Detection Rate: {report_data['avg_detection_quality']:.1f}%")
        
        if 'avg_x_diversity' in report_data:
            print(f"\nTraining Diversity:")
            print(f"  X Position Range: {report_data['avg_x_diversity']:.1f} pixels")
            print(f"  Y Position Range: {report_data['avg_y_diversity']:.1f} pixels")
        
        print("="*50)
    
    def run_full_analysis(self) -> None:
        """Run complete training analysis."""
        logging.info("Starting handshake training analysis...")
        
        # Load data
        self.training_data = self.load_data()
        logging.info(f"Loaded {len(self.training_data)} training data points")
        
        # Create visualizations
        self.create_training_overview()
        self.create_handshake_analysis()
        
        # Generate report
        self.generate_training_report()
        
        logging.info(f"Analysis complete. Results saved to {self.output_dir}")


@parser.wrap()
def visualize_handshake_training(cfg: TrainingVisualizationConfig):
    """Main function for handshake training visualization."""
    init_logging()
    
    visualizer = HandshakeTrainingVisualizer(cfg)
    visualizer.run_full_analysis()


if __name__ == "__main__":
    visualize_handshake_training() 