"""
Visualization utilities for explainability analysis.
Creates publication-ready figures and visual comparisons.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import torch
from PIL import Image
import pandas as pd

from . import config


def setup_style():
    """Set up matplotlib style for publication-quality figures."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14
    })


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert normalized tensor back to displayable image."""
    mean = torch.tensor(config.IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(config.IMAGENET_STD).view(3, 1, 1)
    
    img = tensor.cpu() * std + mean
    img = img.clamp(0, 1)
    img = img.permute(1, 2, 0).numpy()
    
    return img


def create_comparison_figure(
    image: np.ndarray,
    attributions: Dict[str, np.ndarray],
    part_mask: np.ndarray,
    class_name: str = "",
    predicted_class: str = "",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 8)
) -> None:
    """
    Create a comprehensive comparison figure showing all methods.
    
    Args:
        image: Original image (H, W, C)
        attributions: Dictionary of attribution maps
        part_mask: Ground truth part mask
        class_name: True class name
        predicted_class: Predicted class name
        save_path: Path to save figure
        figsize: Figure size
    """
    setup_style()
    
    n_methods = len(attributions)
    n_cols = n_methods + 2  # +2 for original and ground truth
    
    fig, axes = plt.subplots(2, n_cols, figsize=figsize)
    
    # Title
    title = f"Explainability Comparison"
    if class_name:
        title += f"\nTrue: {class_name}"
    if predicted_class:
        title += f" | Predicted: {predicted_class}"
    fig.suptitle(title, fontsize=14, y=1.02)
    
    # Row 1: Raw visualizations
    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')
    
    # Ground truth parts
    axes[0, 1].imshow(part_mask, cmap='Greens')
    axes[0, 1].set_title("Ground Truth Parts")
    axes[0, 1].axis('off')
    
    # Attribution heatmaps
    for idx, (method_name, attr_map) in enumerate(attributions.items()):
        ax = axes[0, idx + 2]
        if attr_map is not None:
            # Normalize for display
            attr_norm = (attr_map - attr_map.min()) / (attr_map.max() - attr_map.min() + 1e-8)
            im = ax.imshow(attr_norm, cmap='hot')
            ax.set_title(method_name.replace('_', ' ').title())
        else:
            ax.set_title(f"{method_name}\n(failed)")
        ax.axis('off')
    
    # Row 2: Overlays
    axes[1, 0].imshow(image)
    axes[1, 0].set_title("Original")
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(image)
    axes[1, 1].imshow(part_mask, cmap='Greens', alpha=0.5)
    axes[1, 1].set_title("GT Overlay")
    axes[1, 1].axis('off')
    
    for idx, (method_name, attr_map) in enumerate(attributions.items()):
        ax = axes[1, idx + 2]
        ax.imshow(image)
        if attr_map is not None:
            attr_norm = (attr_map - attr_map.min()) / (attr_map.max() - attr_map.min() + 1e-8)
            ax.imshow(attr_norm, cmap='hot', alpha=0.5)
        ax.set_title(f"{method_name.replace('_', ' ').title()} Overlay")
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_metrics_radar_chart(
    summary_df: pd.DataFrame,
    metrics: List[str] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Create radar chart comparing methods across metrics.
    
    Args:
        summary_df: Summary DataFrame with mean metrics
        metrics: List of metrics to include
        save_path: Path to save figure
    """
    setup_style()
    
    if metrics is None:
        metrics = ['pointing_game', 'ebpg', 'auc_roc', 'average_precision', 'spearman_correlation']
    
    # Get mean columns
    mean_cols = [f'{m}_mean' for m in metrics]
    available_cols = [c for c in mean_cols if c in summary_df.columns]
    
    if not available_cols:
        print("No metrics available for radar chart")
        return
    
    methods = summary_df['method'].values
    n_methods = len(methods)
    n_metrics = len(available_cols)
    
    # Set up angles for radar chart
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    colors = plt.cm.Set2(np.linspace(0, 1, n_methods))
    
    for idx, method in enumerate(methods):
        values = summary_df[summary_df['method'] == method][available_cols].values.flatten().tolist()
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=method, color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])
    
    # Set labels
    metric_labels = [m.replace('_mean', '').replace('_', '\n').title() for m in available_cols]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels)
    
    ax.set_ylim(0, 1)
    ax.set_title("Method Comparison Across Metrics", size=14, y=1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_top_k_analysis(
    detailed_df: pd.DataFrame,
    save_path: Optional[str] = None
) -> None:
    """
    Create analysis of IoU and Mass Accuracy at different top-k thresholds.
    
    Args:
        detailed_df: Detailed results DataFrame
        save_path: Path to save figure
    """
    setup_style()
    
    # Get IoU and mass accuracy columns
    iou_cols = [c for c in detailed_df.columns if c.startswith('iou_top')]
    mass_cols = [c for c in detailed_df.columns if c.startswith('mass_accuracy_top')]
    
    if not iou_cols or not mass_cols:
        print("No top-k metrics available")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    methods = detailed_df['method'].unique()
    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))
    
    # IoU plot
    for idx, method in enumerate(methods):
        method_data = detailed_df[detailed_df['method'] == method]
        means = [method_data[col].mean() for col in sorted(iou_cols)]
        stds = [method_data[col].std() for col in sorted(iou_cols)]
        
        k_values = [int(col.split('top')[1]) for col in sorted(iou_cols)]
        
        ax1.errorbar(k_values, means, yerr=stds, marker='o', label=method, 
                    color=colors[idx], capsize=3)
    
    ax1.set_xlabel('Top-k Percentage')
    ax1.set_ylabel('IoU Score')
    ax1.set_title('IoU at Different Thresholds')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Mass Accuracy plot
    for idx, method in enumerate(methods):
        method_data = detailed_df[detailed_df['method'] == method]
        means = [method_data[col].mean() for col in sorted(mass_cols)]
        stds = [method_data[col].std() for col in sorted(mass_cols)]
        
        k_values = [int(col.split('top')[1]) for col in sorted(mass_cols)]
        
        ax2.errorbar(k_values, means, yerr=stds, marker='o', label=method,
                    color=colors[idx], capsize=3)
    
    ax2.set_xlabel('Top-k Percentage')
    ax2.set_ylabel('Mass Accuracy')
    ax2.set_title('Mass Accuracy at Different Thresholds')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_per_part_analysis(
    part_results: Dict[str, Dict[int, float]],
    part_names: Dict[int, str],
    save_path: Optional[str] = None
) -> None:
    """
    Create analysis of attribution quality per bird part.
    
    Args:
        part_results: Dictionary mapping method -> part_id -> metric_value
        part_names: Dictionary mapping part_id -> part_name
        save_path: Path to save figure
    """
    setup_style()
    
    methods = list(part_results.keys())
    parts = sorted(part_names.keys())
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(parts))
    width = 0.8 / len(methods)
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))
    
    for idx, method in enumerate(methods):
        values = [part_results[method].get(p, 0) for p in parts]
        offset = (idx - len(methods) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=method, color=colors[idx])
    
    ax.set_xlabel('Bird Part')
    ax.set_ylabel('EBPG Score')
    ax.set_title('Explanation Quality by Bird Part')
    ax.set_xticks(x)
    ax.set_xticklabels([part_names[p] for p in parts], rotation=45, ha='right')
    ax.legend(title='Method')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_model_vs_method_analysis(
    detailed_df: pd.DataFrame,
    metric: str = 'ebpg',
    save_path: Optional[str] = None
) -> None:
    """
    Create visualization analyzing whether results are due to model or method.
    
    Args:
        detailed_df: Detailed results DataFrame
        metric: Metric to analyze
        save_path: Path to save figure
    """
    setup_style()
    
    if 'correct' not in detailed_df.columns or metric not in detailed_df.columns:
        print("Required columns not available")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    methods = detailed_df['method'].unique()
    
    # 1. Box plot: Correct vs Incorrect
    ax1 = axes[0]
    colors = {'True': '#2ecc71', 'False': '#e74c3c'}
    
    # Convert boolean to string for better display
    plot_df = detailed_df.copy()
    plot_df['Correct Prediction'] = plot_df['correct'].map({True: 'Correct', False: 'Incorrect'})
    
    sns.boxplot(data=plot_df, x='method', y=metric, hue='Correct Prediction', ax=ax1,
               palette=['#2ecc71', '#e74c3c'])
    ax1.set_xticklabels([t.get_text().replace('_', '\n') for t in ax1.get_xticklabels()])
    ax1.set_title('Explanation Quality:\nCorrect vs Incorrect Predictions')
    ax1.set_ylabel(metric.upper())
    
    # 2. Difference plot
    ax2 = axes[1]
    
    correct_means = plot_df[plot_df['correct'] == True].groupby('method')[metric].mean()
    incorrect_means = plot_df[plot_df['correct'] == False].groupby('method')[metric].mean()
    
    diff = correct_means - incorrect_means
    colors_diff = ['#2ecc71' if d > 0 else '#e74c3c' for d in diff.values]
    
    bars = ax2.bar(range(len(diff)), diff.values, color=colors_diff)
    ax2.set_xticks(range(len(diff)))
    ax2.set_xticklabels([m.replace('_', '\n') for m in diff.index])
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_title('Difference in Explanation Quality\n(Correct - Incorrect)')
    ax2.set_ylabel(f'Δ {metric.upper()}')
    
    # Add significance annotation
    ax2.text(0.5, 0.95, 
            "Positive = Better explanations for correct predictions\n(suggests faithful explanations)",
            transform=ax2.transAxes, ha='center', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 3. Correlation with confidence
    ax3 = axes[2]
    
    method_colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))
    
    for idx, method in enumerate(methods):
        method_data = plot_df[plot_df['method'] == method]
        correct_rate = method_data['correct'].mean()
        mean_metric = method_data[metric].mean()
        
        ax3.scatter(correct_rate, mean_metric, s=200, label=method, 
                   color=method_colors[idx], edgecolors='black', linewidth=1)
    
    ax3.set_xlabel('Classification Accuracy')
    ax3.set_ylabel(f'Mean {metric.upper()}')
    ax3.set_title('Method Performance vs Model Accuracy')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_summary_report(
    summary_df: pd.DataFrame,
    detailed_df: pd.DataFrame,
    output_dir: str
) -> None:
    """
    Create all visualizations for a complete analysis report.
    
    Args:
        summary_df: Summary results DataFrame
        detailed_df: Detailed results DataFrame
        output_dir: Directory to save all figures
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("Creating summary report visualizations...")
    
    # 1. Metrics comparison bar chart
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        metrics = ['pointing_game', 'ebpg', 'auc_roc', 'average_precision']
        titles = ['Pointing Game', 'Energy-Based Pointing Game', 'AUC-ROC', 'Average Precision']
        
        for ax, metric, title in zip(axes.flatten(), metrics, titles):
            mean_col = f'{metric}_mean'
            std_col = f'{metric}_std'
            
            if mean_col in summary_df.columns:
                methods = summary_df['method'].values
                means = summary_df[mean_col].values
                stds = summary_df.get(std_col, np.zeros_like(means))
                
                colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))
                bars = ax.bar(range(len(methods)), means, yerr=stds, capsize=5, color=colors)
                ax.set_xticks(range(len(methods)))
                ax.set_xticklabels([m.replace('_', '\n') for m in methods])
                ax.set_ylabel('Score')
                ax.set_title(title)
                ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/1_metrics_comparison.png", dpi=150)
        plt.close()
        print("  ✓ Metrics comparison")
    except Exception as e:
        print(f"  ✗ Metrics comparison: {e}")
    
    # 2. Radar chart
    try:
        create_metrics_radar_chart(summary_df, save_path=f"{output_dir}/2_radar_chart.png")
        print("  ✓ Radar chart")
    except Exception as e:
        print(f"  ✗ Radar chart: {e}")
    
    # 3. Top-k analysis
    try:
        create_top_k_analysis(detailed_df, save_path=f"{output_dir}/3_topk_analysis.png")
        print("  ✓ Top-k analysis")
    except Exception as e:
        print(f"  ✗ Top-k analysis: {e}")
    
    # 4. Model vs Method analysis
    try:
        create_model_vs_method_analysis(detailed_df, save_path=f"{output_dir}/4_model_vs_method.png")
        print("  ✓ Model vs Method analysis")
    except Exception as e:
        print(f"  ✗ Model vs Method analysis: {e}")
    
    # 5. Distribution plots
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        metrics = ['pointing_game', 'ebpg', 'auc_roc', 'average_precision']
        
        for ax, metric in zip(axes.flatten(), metrics):
            if metric in detailed_df.columns:
                sns.violinplot(data=detailed_df, x='method', y=metric, ax=ax)
                ax.set_xticklabels([t.get_text().replace('_', '\n') for t in ax.get_xticklabels()])
                ax.set_title(metric.replace('_', ' ').title())
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/5_distributions.png", dpi=150)
        plt.close()
        print("  ✓ Distribution plots")
    except Exception as e:
        print(f"  ✗ Distribution plots: {e}")
    
    print(f"\nAll visualizations saved to {output_dir}")


if __name__ == "__main__":
    # Test visualizations with dummy data
    print("Testing visualization utilities...")
    
    # Create dummy data
    np.random.seed(42)
    
    methods = ['integrated_gradients', 'input_gradients', 'lime', 'kernel_shap']
    
    # Dummy summary DataFrame
    summary_data = []
    for method in methods:
        row = {
            'method': method,
            'pointing_game_mean': np.random.uniform(0.3, 0.8),
            'pointing_game_std': np.random.uniform(0.1, 0.2),
            'ebpg_mean': np.random.uniform(0.2, 0.6),
            'ebpg_std': np.random.uniform(0.1, 0.15),
            'auc_roc_mean': np.random.uniform(0.5, 0.8),
            'auc_roc_std': np.random.uniform(0.05, 0.1),
            'average_precision_mean': np.random.uniform(0.3, 0.7),
            'average_precision_std': np.random.uniform(0.1, 0.2)
        }
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    
    print("Summary DataFrame created for testing")
    print(summary_df)
    
    # Test radar chart
    create_metrics_radar_chart(summary_df, save_path=str(config.RESULTS_DIR / 'test_radar.png'))
    print(f"Test radar chart saved to {config.RESULTS_DIR / 'test_radar.png'}")
