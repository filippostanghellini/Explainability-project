"""
Visualization utilities for explainability analysis.
Creates publication-ready figures following XAI literature best practices.

Based on:
- Adebayo et al. (2018): Sanity Checks for Saliency Maps
- Samek et al. (2016): Evaluating the visualization of what a DNN has learned
- Zhang et al.: Pointing Game and EBPG metrics
- Montavon et al. (2019): Layer-wise relevance propagation

All visualizations use:
- Color-blind friendly palettes (Okabe-Ito)
- High DPI for publication quality
- Consistent styling across all figures
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
import torch.nn.functional as F

from . import config


def setup_style(style: str = 'publication', dpi: int = 150):
    """
    Set up matplotlib style for publication-quality figures.
    
    Args:
        style: 'publication' for journal-ready figures, 'presentation' for larger fonts
        dpi: DPI for saved figures (default 150, use 300+ for publication)
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Color-blind friendly color cycle (based on Okabe-Ito palette)
    colorblind_colors = [
        '#0077BB',  # Blue
        '#EE7733',  # Orange
        '#009988',  # Teal
        '#CC3311',  # Red
        '#33BBEE',  # Cyan
        '#EE3377',  # Magenta
        '#BBBBBB',  # Grey
        '#000000',  # Black
    ]
    
    base_settings = {
        'figure.dpi': dpi,
        'savefig.dpi': dpi,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'axes.prop_cycle': plt.cycler(color=colorblind_colors),
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif'],
        'image.cmap': 'viridis',  # Color-blind friendly default
    }
    
    if style == 'publication':
        base_settings.update({
            'font.size': 10,
            'axes.titlesize': 11,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.titlesize': 12,
            'lines.linewidth': 1.5,
            'axes.linewidth': 0.8,
        })
    elif style == 'presentation':
        base_settings.update({
            'font.size': 14,
            'axes.titlesize': 16,
            'axes.labelsize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.titlesize': 18,
            'lines.linewidth': 2.5,
            'axes.linewidth': 1.2,
        })
    else:  # default
        base_settings.update({
            'font.size': 11,
            'axes.titlesize': 12,
            'axes.labelsize': 11,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 14,
        })
    
    plt.rcParams.update(base_settings)


# Color-blind friendly colormaps for attributions (Okabe-Ito palette)
SEQUENTIAL_CMAP = 'inferno'  # Better than 'hot' for colorblind accessibility
DIVERGENT_CMAP = 'RdBu_r'    # Red-Blue divergent (still accessible)

# Methods that preserve sign (use divergent colormap)
DIVERGENT_METHODS = {'lime', 'kernel_shap'}


# =============================================================================
# CORE VISUALIZATION FUNCTIONS - XAI Literature Best Practices
# =============================================================================

def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert normalized tensor back to displayable image."""
    mean = torch.tensor(config.IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(config.IMAGENET_STD).view(3, 1, 1)
    
    img = tensor.cpu() * std + mean
    img = img.clamp(0, 1)
    img = img.permute(1, 2, 0).numpy()
    
    return img


def visualize_all_methods(
    input_image: np.ndarray,
    attributions: Dict[str, np.ndarray],
    part_mask: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Visualize attributions from all methods in a single figure.
    
    Uses divergent colormap (RdBu_r) for perturbation-based methods (LIME, SHAP)
    and sequential colormap (hot) for gradient-based methods.
    
    Args:
        input_image: Original image as numpy array (H, W, C)
        attributions: Dictionary of attribution maps
        part_mask: Optional ground truth part mask
        save_path: Path to save the figure
        show: Whether to display the figure
    """
    n_methods = len(attributions) + (1 if part_mask is not None else 0) + 1
    n_cols = 3
    n_rows = (n_methods + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()
    
    # Original image
    axes[0].imshow(input_image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Ground truth part mask if available
    idx = 1
    if part_mask is not None:
        axes[idx].imshow(input_image)
        axes[idx].imshow(part_mask, cmap='Greens', alpha=0.5)
        axes[idx].set_title("Ground Truth Parts")
        axes[idx].axis('off')
        idx += 1
    
    # Attribution maps
    for method_name, attr_map in attributions.items():
        if attr_map is not None:
            # Choose colormap based on method type
            if method_name in DIVERGENT_METHODS:
                # Perturbation-based: divergent colormap centered on 0
                # Red = increases probability, Blue = decreases probability
                # Do NOT take absolute value - preserve sign!
                vmax = max(abs(attr_map.min()), abs(attr_map.max()))
                vmin = -vmax  # Symmetric around 0
                cmap = DIVERGENT_CMAP  # 'RdBu_r': Red (positive) - White (zero) - Blue (negative)
                
                axes[idx].imshow(input_image)
                im = axes[idx].imshow(attr_map, cmap=cmap, alpha=0.7, vmin=vmin, vmax=vmax)
                plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
            else:
                # Gradient-based: sequential colormap (already absolute value)
                attr_norm = (attr_map - attr_map.min()) / (attr_map.max() - attr_map.min() + 1e-8)
                cmap = SEQUENTIAL_CMAP  # 'inferno'
                
                axes[idx].imshow(input_image)
                im = axes[idx].imshow(attr_norm, cmap=cmap, alpha=0.6)
                plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
            
            axes[idx].set_title(method_name.replace('_', ' ').title())
            axes[idx].axis('off')
        else:
            axes[idx].set_title(f"{method_name} (failed)")
            axes[idx].axis('off')
        idx += 1
    
    # Hide remaining axes
    for i in range(idx, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


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
        metrics = ['pointing_game', 'ebpg', 'auc_roc', 'average_precision']
    
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


def create_model_vs_method_analysis(
    detailed_df: pd.DataFrame,
    metric: str = 'ebpg',
    save_path: Optional[str] = None
) -> None:
    """
    Create visualization analyzing whether results are due to model or method.
    Fixed to avoid UserWarning about FixedLocator.
    """
    setup_style()
    
    if 'correct' not in detailed_df.columns or metric not in detailed_df.columns:
        print("Required columns not available")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    methods = detailed_df['method'].unique()
    
    # ---------------------------------------------------------
    # 1. Box plot: Correct vs Incorrect
    # ---------------------------------------------------------
    ax1 = axes[0]
    
    # Convert boolean to string for better display
    plot_df = detailed_df.copy()
    plot_df['Correct Prediction'] = plot_df['correct'].map({True: 'Correct', False: 'Incorrect'})
    
    sns.boxplot(data=plot_df, x='method', y=metric, hue='Correct Prediction', ax=ax1,
               palette=['#2ecc71', '#e74c3c'])
    
    # ### FIX WARNING 1 (BOX PLOT) ###
    # 1. Recuperiamo le posizioni (locators) attuali decise da Seaborn
    locs = ax1.get_xticks()
    # 2. Recuperiamo le etichette di testo attuali
    labels = [item.get_text() for item in ax1.get_xticklabels()]
    
    # 3. Fissiamo PRIMA le posizioni (questo silenzia il warning)
    ax1.set_xticks(locs)
    # 4. SOLO ORA impostiamo le nuove etichette formattate
    ax1.set_xticklabels([l.replace('_', '\n') for l in labels])
    # ###############################
    
    ax1.set_title('Explanation Quality:\nCorrect vs Incorrect Predictions')
    ax1.set_ylabel(metric.upper())
    
    # ---------------------------------------------------------
    # 2. Difference plot
    # ---------------------------------------------------------
    ax2 = axes[1]
    
    correct_means = plot_df[plot_df['correct'] == True].groupby('method')[metric].mean()
    incorrect_means = plot_df[plot_df['correct'] == False].groupby('method')[metric].mean()
    
    diff = correct_means - incorrect_means
    colors_diff = ['#2ecc71' if d > 0 else '#e74c3c' for d in diff.values]
    
    # Qui usiamo range(len) quindi le posizioni sono già esplicite
    bars = ax2.bar(range(len(diff)), diff.values, color=colors_diff)
    
    # ### FIX WARNING 2 (BAR PLOT) ###
    # Fissiamo le posizioni esplicitamente
    ax2.set_xticks(range(len(diff)))
    # Impostiamo le etichette
    ax2.set_xticklabels([m.replace('_', '\n') for m in diff.index])
    # ##############################
    
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_title('Difference in Explanation Quality\n(Correct - Incorrect)')
    ax2.set_ylabel(f'Δ {metric.upper()}')
    
    ax2.text(0.5, 0.95, 
            "Positive = Better explanations for correct predictions\n(suggests faithful explanations)",
            transform=ax2.transAxes, ha='center', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # ---------------------------------------------------------
    # 3. Correlation with confidence
    # ---------------------------------------------------------
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


# =============================================================================
# XAI LITERATURE-BASED VISUALIZATIONS
# =============================================================================

def create_roc_curve_comparison(
    attributions: Dict[str, np.ndarray],
    part_mask: np.ndarray,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Create ROC curves comparing all attribution methods.
    
    Based on standard practice in XAI evaluation (Adebayo et al., NeurIPS 2018).
    
    Args:
        attributions: Dictionary of method_name -> attribution_map
        part_mask: Binary ground truth mask
        save_path: Path to save figure
        show: Whether to display
    """
    from sklearn.metrics import roc_curve, auc
    
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 8))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(attributions)))
    gt_flat = part_mask.flatten().astype(int)
    
    for idx, (method, attr_map) in enumerate(attributions.items()):
        if attr_map is None:
            continue
            
        # Normalize attribution to [0, 1]
        attr_flat = attr_map.flatten()
        attr_norm = (attr_flat - attr_flat.min()) / (attr_flat.max() - attr_flat.min() + 1e-8)
        
        # Compute ROC curve
        fpr, tpr, _ = roc_curve(gt_flat, attr_norm)
        roc_auc = auc(fpr, tpr)
        
        # Plot
        ax.plot(fpr, tpr, color=colors[idx], lw=2,
                label=f'{method.replace("_", " ").title()} (AUC = {roc_auc:.3f})')
    
    # Reference line
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC = 0.500)')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves: Attribution Methods Comparison')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()


def create_pr_curve_comparison(
    attributions: Dict[str, np.ndarray],
    part_mask: np.ndarray,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Create Precision-Recall curves comparing all attribution methods.
    
    Args:
        attributions: Dictionary of method_name -> attribution_map
        part_mask: Binary ground truth mask
        save_path: Path to save figure
        show: Whether to display
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 8))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(attributions)))
    gt_flat = part_mask.flatten().astype(int)
    
    # Baseline (random classifier)
    baseline_ap = gt_flat.mean()
    
    for idx, (method, attr_map) in enumerate(attributions.items()):
        if attr_map is None:
            continue
            
        attr_flat = attr_map.flatten()
        attr_norm = (attr_flat - attr_flat.min()) / (attr_flat.max() - attr_flat.min() + 1e-8)
        
        precision, recall, _ = precision_recall_curve(gt_flat, attr_norm)
        ap = average_precision_score(gt_flat, attr_norm)
        
        ax.plot(recall, precision, color=colors[idx], lw=2,
                label=f'{method.replace("_", " ").title()} (AP = {ap:.3f})')
    
    # Baseline line
    ax.axhline(y=baseline_ap, linestyle='--', color='gray', lw=1,
               label=f'Random (AP = {baseline_ap:.3f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves: Attribution Methods Comparison')
    ax.legend(loc='upper right', framealpha=0.9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()


def create_iou_threshold_curves(
    attributions: Dict[str, np.ndarray],
    part_mask: np.ndarray,
    thresholds: List[float] = None,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Create IoU vs threshold curves for each method.
    
    Shows how IoU varies with different top-k percentages.
    
    Args:
        attributions: Dictionary of method_name -> attribution_map
        part_mask: Binary ground truth mask
        thresholds: List of top-k percentages to evaluate
        save_path: Path to save figure
        show: Whether to display
    """
    if thresholds is None:
        thresholds = [1, 2, 5, 10, 15, 20, 25, 30, 40, 50]
    
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(attributions)))
    gt_binary = part_mask > 0
    
    for idx, (method, attr_map) in enumerate(attributions.items()):
        if attr_map is None:
            continue
            
        ious = []
        for k in thresholds:
            # Threshold to get top k%
            threshold = np.percentile(attr_map, 100 - k)
            pred_binary = attr_map >= threshold
            
            # Compute IoU
            intersection = np.logical_and(pred_binary, gt_binary).sum()
            union = np.logical_or(pred_binary, gt_binary).sum()
            iou = intersection / (union + 1e-8)
            ious.append(iou)
        
        ax.plot(thresholds, ious, 'o-', color=colors[idx], lw=2, markersize=6,
                label=f'{method.replace("_", " ").title()}')
    
    ax.set_xlabel('Top-k Percentage (%)')
    ax.set_ylabel('IoU Score')
    ax.set_title('IoU at Different Thresholds')
    ax.legend(loc='best', framealpha=0.9)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()


def visualize_attribution_enhanced(
    input_image: np.ndarray,
    attribution_map: np.ndarray,
    part_mask: Optional[np.ndarray] = None,
    method_name: str = "Attribution",
    show_contours: bool = True,
    show_percentiles: bool = True,
    percentiles: List[float] = [5, 10, 20],
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Enhanced attribution visualization with contours and percentile thresholding.
    
    Based on best practices from Montavon et al. (2019).
    
    Args:
        input_image: Original image (H, W, C)
        attribution_map: Attribution map (H, W)
        part_mask: Optional ground truth mask for comparison
        method_name: Name of the attribution method
        show_contours: Whether to show contour lines
        show_percentiles: Whether to show percentile thresholds
        percentiles: Which percentiles to highlight
        save_path: Path to save figure
        show: Whether to display
    """
    setup_style()
    
    n_cols = 3 if part_mask is not None else 2
    if show_percentiles:
        n_cols += len(percentiles)
    
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    
    # Determine colormap based on method
    is_divergent = method_name.lower() in DIVERGENT_METHODS
    if is_divergent:
        vmax = np.abs(attribution_map).max()
        vmin = -vmax
        cmap = DIVERGENT_CMAP
    else:
        attr_norm = (attribution_map - attribution_map.min()) / (attribution_map.max() - attribution_map.min() + 1e-8)
        vmin, vmax = 0, 1
        cmap = SEQUENTIAL_CMAP
    
    idx = 0
    
    # 1. Original image
    axes[idx].imshow(input_image)
    axes[idx].set_title("Original Image", fontweight='bold')
    axes[idx].axis('off')
    idx += 1
    
    # 2. Attribution with contours
    axes[idx].imshow(input_image)
    if is_divergent:
        im = axes[idx].imshow(attribution_map, cmap=cmap, alpha=0.7, vmin=vmin, vmax=vmax)
    else:
        im = axes[idx].imshow(attr_norm, cmap=cmap, alpha=0.7)
    
    if show_contours:
        contour_data = attribution_map if is_divergent else attr_norm
        levels = np.percentile(contour_data, [50, 75, 90, 95])
        axes[idx].contour(contour_data, levels=levels, colors='white', linewidths=0.8, alpha=0.8)
    
    axes[idx].set_title(f"{method_name}\n(with contours)", fontweight='bold')
    axes[idx].axis('off')
    plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
    idx += 1
    
    # 3. Ground truth overlay (if available)
    if part_mask is not None:
        axes[idx].imshow(input_image)
        axes[idx].imshow(part_mask, cmap='Greens', alpha=0.5)
        axes[idx].contour(part_mask, levels=[0.5], colors='lime', linewidths=2)
        axes[idx].set_title("Ground Truth", fontweight='bold')
        axes[idx].axis('off')
        idx += 1
    
    # 4. Percentile thresholded views
    if show_percentiles:
        for p in percentiles:
            threshold = np.percentile(attribution_map, 100 - p)
            mask = attribution_map >= threshold
            
            axes[idx].imshow(input_image)
            axes[idx].imshow(mask, cmap='Reds', alpha=0.6)
            axes[idx].contour(mask, levels=[0.5], colors='red', linewidths=1.5)
            
            if part_mask is not None:
                axes[idx].contour(part_mask, levels=[0.5], colors='lime', linewidths=1.5, linestyles='--')
            
            axes[idx].set_title(f"Top {p}%", fontweight='bold')
            axes[idx].axis('off')
            idx += 1
    
    plt.suptitle(f"Enhanced Attribution Analysis: {method_name}", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()


def create_pointing_game_visualization(
    input_image: np.ndarray,
    attribution_map: np.ndarray,
    part_mask: np.ndarray,
    method_name: str = "Attribution",
    top_k_percent: float = 5.0,
    save_path: Optional[str] = None,
    show: bool = True
) -> Dict:
    """
    Visualize Pointing Game metric - shows whether max attribution hits GT region.
    
    Args:
        input_image: Original image (H, W, C)
        attribution_map: Attribution map (H, W)
        part_mask: Binary ground truth mask
        method_name: Name of the method
        top_k_percent: Percentage for top-k pointing game variant
        save_path: Path to save figure
        show: Whether to display
        
    Returns:
        Dictionary with pointing game results
    """
    setup_style()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Find max attribution location
    max_idx = np.unravel_index(np.argmax(attribution_map), attribution_map.shape)
    max_y, max_x = max_idx
    
    # Check if hit
    is_hit = part_mask[max_y, max_x] > 0
    
    # 1. Attribution map with max point
    axes[0].imshow(input_image)
    attr_norm = (attribution_map - attribution_map.min()) / (attribution_map.max() - attribution_map.min() + 1e-8)
    axes[0].imshow(attr_norm, cmap=SEQUENTIAL_CMAP, alpha=0.6)
    
    # Mark max point
    color = 'lime' if is_hit else 'red'
    marker = '★' if is_hit else '✕'
    axes[0].scatter(max_x, max_y, c=color, s=300, marker='*', edgecolors='white', linewidth=2, zorder=5)
    axes[0].set_title(f"Max Attribution Point\n{'✓ HIT' if is_hit else '✗ MISS'}", 
                      fontweight='bold', color='green' if is_hit else 'red')
    axes[0].axis('off')
    
    # 2. Ground truth with max point
    axes[1].imshow(input_image)
    axes[1].imshow(part_mask, cmap='Greens', alpha=0.5)
    axes[1].contour(part_mask, levels=[0.5], colors='lime', linewidths=2)
    axes[1].scatter(max_x, max_y, c=color, s=300, marker='*', edgecolors='white', linewidth=2, zorder=5)
    axes[1].set_title("Ground Truth Region", fontweight='bold')
    axes[1].axis('off')
    
    # 3. Top-k% with GT overlay
    threshold = np.percentile(attribution_map, 100 - top_k_percent)
    top_k_mask = attribution_map >= threshold
    
    axes[2].imshow(input_image)
    axes[2].imshow(top_k_mask, cmap='Reds', alpha=0.5)
    axes[2].contour(part_mask, levels=[0.5], colors='lime', linewidths=2)
    
    # Check if any top-k point hits
    top_k_hit = np.any(np.logical_and(top_k_mask, part_mask > 0))
    axes[2].set_title(f"Top {top_k_percent}% Attribution\n{'✓ HIT' if top_k_hit else '✗ MISS'}", 
                      fontweight='bold', color='green' if top_k_hit else 'red')
    axes[2].axis('off')
    
    plt.suptitle(f"Pointing Game Analysis: {method_name}", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()
    
    return {
        'max_point': (max_x, max_y),
        'is_hit': is_hit,
        'top_k_hit': top_k_hit,
        'top_k_percent': top_k_percent
    }


def create_ebpg_visualization(
    attribution_map: np.ndarray,
    part_mask: np.ndarray,
    method_name: str = "Attribution",
    save_path: Optional[str] = None,
    show: bool = True
) -> Dict:
    """
    Visualize Energy-Based Pointing Game - shows energy distribution inside/outside GT.
    
    Args:
        attribution_map: Attribution map (H, W)
        part_mask: Binary ground truth mask
        method_name: Name of the method
        save_path: Path to save figure
        show: Whether to display
        
    Returns:
        Dictionary with EBPG results
    """
    setup_style()
    
    fig = plt.figure(figsize=(14, 5))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 1.2])
    
    # Use absolute values for energy calculation
    attr_abs = np.abs(attribution_map)
    total_energy = attr_abs.sum()
    
    gt_mask = part_mask > 0
    energy_in_gt = attr_abs[gt_mask].sum()
    energy_outside_gt = attr_abs[~gt_mask].sum()
    
    ebpg_score = energy_in_gt / (total_energy + 1e-8)
    
    # 1. Attribution map split by GT
    ax1 = fig.add_subplot(gs[0])
    attr_in = np.where(gt_mask, attr_abs, np.nan)
    attr_out = np.where(~gt_mask, attr_abs, np.nan)
    
    ax1.imshow(attr_out, cmap='Blues', alpha=0.8, vmin=0, vmax=attr_abs.max())
    im = ax1.imshow(attr_in, cmap='Reds', alpha=0.8, vmin=0, vmax=attr_abs.max())
    ax1.contour(part_mask, levels=[0.5], colors='black', linewidths=2)
    ax1.set_title("Energy Distribution\n(Red=Inside GT, Blue=Outside)", fontweight='bold')
    ax1.axis('off')
    
    # 2. Pie chart
    ax2 = fig.add_subplot(gs[1])
    colors_pie = ['#2ecc71', '#e74c3c']  # Green for inside, red for outside
    sizes = [ebpg_score * 100, (1 - ebpg_score) * 100]
    labels = [f'Inside GT\n({ebpg_score:.1%})', f'Outside GT\n({1-ebpg_score:.1%})']
    
    wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors_pie, autopct='',
                                        startangle=90, explode=(0.05, 0))
    ax2.set_title(f"EBPG Score: {ebpg_score:.3f}", fontweight='bold', fontsize=12)
    
    # 3. Cumulative energy curve
    ax3 = fig.add_subplot(gs[2])
    
    # Sort pixels by attribution value
    flat_attr = attr_abs.flatten()
    flat_gt = gt_mask.flatten()
    sorted_indices = np.argsort(flat_attr)[::-1]  # Descending
    
    sorted_attr = flat_attr[sorted_indices]
    sorted_gt = flat_gt[sorted_indices]
    
    # Cumulative energy
    cumulative_energy = np.cumsum(sorted_attr) / total_energy
    cumulative_gt_energy = np.cumsum(sorted_attr * sorted_gt) / (energy_in_gt + 1e-8)
    
    # X-axis as percentage of pixels
    x = np.arange(len(flat_attr)) / len(flat_attr) * 100
    
    ax3.plot(x, cumulative_energy, 'b-', lw=2, label='Total Energy')
    ax3.plot(x, cumulative_gt_energy, 'g-', lw=2, label='GT Energy (normalized)')
    ax3.axhline(y=0.5, linestyle='--', color='gray', alpha=0.5)
    ax3.axhline(y=0.9, linestyle='--', color='gray', alpha=0.5)
    
    ax3.set_xlabel('Percentage of Pixels (sorted by attribution)')
    ax3.set_ylabel('Cumulative Energy')
    ax3.set_title('Cumulative Energy Distribution', fontweight='bold')
    ax3.legend(loc='lower right')
    ax3.set_xlim(0, 100)
    ax3.set_ylim(0, 1.05)
    ax3.grid(True, alpha=0.3)
    
    plt.suptitle(f"Energy-Based Pointing Game Analysis: {method_name}", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()
    
    return {
        'ebpg_score': ebpg_score,
        'energy_in_gt': energy_in_gt,
        'energy_outside_gt': energy_outside_gt,
        'total_energy': total_energy
    }


def create_methods_ranking_heatmap(
    summary_df: pd.DataFrame,
    metrics: List[str] = None,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Create a heatmap showing method rankings across metrics.
    
    Args:
        summary_df: Summary DataFrame with mean metrics per method
        metrics: List of metrics to include
        save_path: Path to save figure
        show: Whether to display
    """
    setup_style()
    
    if metrics is None:
        metrics = ['pointing_game', 'ebpg', 'auc_roc', 'average_precision']
    
    # Get available metrics
    mean_cols = [f'{m}_mean' for m in metrics]
    available_cols = [c for c in mean_cols if c in summary_df.columns]
    
    if not available_cols:
        print("No metrics available for heatmap")
        return
    
    # Create ranking matrix
    methods = summary_df['method'].values
    data = summary_df[available_cols].values
    
    # Rank within each column (higher is better, so rank descending)
    ranks = np.zeros_like(data)
    for j in range(data.shape[1]):
        ranks[:, j] = len(methods) - np.argsort(np.argsort(data[:, j]))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Values heatmap
    metric_labels = [m.replace('_mean', '').replace('_', ' ').title() for m in available_cols]
    
    sns.heatmap(data, annot=True, fmt='.3f', cmap='YlGnBu', ax=axes[0],
                xticklabels=metric_labels, yticklabels=[m.replace('_', ' ').title() for m in methods],
                cbar_kws={'label': 'Score'})
    axes[0].set_title('Metric Values', fontweight='bold')
    axes[0].set_xlabel('Metric')
    axes[0].set_ylabel('Method')
    
    # 2. Rankings heatmap
    sns.heatmap(ranks, annot=True, fmt='.0f', cmap='RdYlGn_r', ax=axes[1],
                xticklabels=metric_labels, yticklabels=[m.replace('_', ' ').title() for m in methods],
                cbar_kws={'label': 'Rank (1 = best)'}, vmin=1, vmax=len(methods))
    axes[1].set_title('Method Rankings', fontweight='bold')
    axes[1].set_xlabel('Metric')
    axes[1].set_ylabel('Method')
    
    plt.suptitle('Methods Comparison: Values and Rankings', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()


def create_sanity_check_comparison(
    original_attributions: Dict[str, np.ndarray],
    randomized_attributions: Dict[str, np.ndarray],
    input_image: np.ndarray,
    save_path: Optional[str] = None,
    show: bool = True
) -> Dict:
    """
    Visualize sanity check results by comparing original and randomized model attributions.
    
    Based on Adebayo et al. (2018) "Sanity Checks for Saliency Maps".
    
    Args:
        original_attributions: Attributions from trained model
        randomized_attributions: Attributions from randomized model
        input_image: Original image for reference
        save_path: Path to save figure
        show: Whether to display
        
    Returns:
        Dictionary with similarity scores
    """
    from scipy.stats import spearmanr
    
    setup_style()
    
    methods = list(original_attributions.keys())
    n_methods = len(methods)
    
    fig, axes = plt.subplots(n_methods, 4, figsize=(16, 4 * n_methods))
    if n_methods == 1:
        axes = axes.reshape(1, -1)
    
    results = {}
    
    for i, method in enumerate(methods):
        orig = original_attributions.get(method)
        rand = randomized_attributions.get(method)
        
        if orig is None or rand is None:
            continue
        
        # Normalize for visualization
        orig_norm = (orig - orig.min()) / (orig.max() - orig.min() + 1e-8)
        rand_norm = (rand - rand.min()) / (rand.max() - rand.min() + 1e-8)
        
        # Compute similarity
        correlation, _ = spearmanr(orig.flatten(), rand.flatten())
        ssim = 1 - np.mean((orig_norm - rand_norm) ** 2)  # Simple similarity
        
        results[method] = {
            'spearman_correlation': correlation,
            'similarity': ssim,
            'passes_sanity_check': abs(correlation) < 0.5  # Low correlation = passes
        }
        
        # Plot
        axes[i, 0].imshow(input_image)
        axes[i, 0].set_title("Original Image" if i == 0 else "")
        axes[i, 0].set_ylabel(method.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(orig_norm, cmap=SEQUENTIAL_CMAP)
        axes[i, 1].set_title("Trained Model" if i == 0 else "")
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(rand_norm, cmap=SEQUENTIAL_CMAP)
        axes[i, 2].set_title("Randomized Model" if i == 0 else "")
        axes[i, 2].axis('off')
        
        # Difference
        diff = np.abs(orig_norm - rand_norm)
        im = axes[i, 3].imshow(diff, cmap='Reds')
        axes[i, 3].set_title("Difference" if i == 0 else "")
        axes[i, 3].axis('off')
        
        # Add correlation annotation
        color = 'green' if results[method]['passes_sanity_check'] else 'red'
        status = '✓ PASS' if results[method]['passes_sanity_check'] else '✗ FAIL'
        axes[i, 3].text(1.05, 0.5, f"ρ = {correlation:.2f}\n{status}", 
                        transform=axes[i, 3].transAxes, fontsize=10, fontweight='bold',
                        color=color, va='center')
    
    plt.suptitle('Sanity Check: Trained vs Randomized Model Attributions', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()
    
    return results


def create_cascade_randomization_plot(
    cascade_results: Dict[str, Dict[str, List[float]]],
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Create visualization for cascade randomization test (Adebayo et al., 2018).
    
    Shows how explanations change as layers are progressively randomized
    from top (logit) to bottom (conv1). Faithful methods should show
    decreasing correlation as more layers (especially top layers) are randomized.
    
    Args:
        cascade_results: Results from cascade_randomization_test()
                        {method: {layer_stage: [correlations]}}
        save_path: Path to save figure
        show: Whether to display
    """
    setup_style()
    
    # Extract layer stages and methods
    methods = list(cascade_results.keys())
    if not methods:
        print("No methods to plot")
        return
    
    # Get layer stages (assuming all methods have same stages)
    stages = list(cascade_results[methods[0]].keys())
    
    # Compute mean and std for each method at each stage
    data = {}
    for method in methods:
        means = []
        stds = []
        for stage in stages:
            correlations = cascade_results[method].get(stage, [])
            if correlations:
                means.append(np.mean(correlations))
                stds.append(np.std(correlations))
            else:
                means.append(np.nan)
                stds.append(0.0)
        data[method] = {'means': means, 'stds': stds}
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))
    x_positions = np.arange(len(stages))
    
    for idx, method in enumerate(methods):
        means = data[method]['means']
        stds = data[method]['stds']
        
        ax.plot(x_positions, means, marker='o', linewidth=2, markersize=8,
                label=method.replace('_', ' ').title(), color=colors[idx])
        ax.fill_between(x_positions, 
                        np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds),
                        alpha=0.2, color=colors[idx])
    
    # Styling
    ax.set_xticks(x_positions)
    ax.set_xticklabels([s.capitalize() for s in stages], rotation=0)
    ax.set_xlabel('Randomized Layers (Cumulative: Logit → Conv1)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Spearman Correlation (ρ)', fontweight='bold', fontsize=12)
    ax.set_title('Cascade Randomization Test: Model Parameter Dependence\n' + 
                 'Lower correlation = More dependent on learned weights = Better',
                 fontweight='bold', fontsize=14)
    
    # Add reference lines
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5, 
               label='Threshold (ρ=0.5)')
    ax.axhline(y=0.0, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Add interpretation zones
    ax.axhspan(0, 0.3, alpha=0.1, color='green', label='Reliable (ρ < 0.3)')
    ax.axhspan(0.3, 0.6, alpha=0.1, color='orange')
    ax.axhspan(0.6, 1.0, alpha=0.1, color='red', label='Unreliable (ρ > 0.6)')
    
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='upper right', framealpha=0.9, fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add interpretation text box
    interpretation_text = (
        "Expected behavior:\n"
        "• Correlation should DECREASE as layers are randomized\n"
        "• Sharp drop at 'logit' or 'layer4' = Method depends on high-level features\n"
        "• High correlation throughout = Method may be edge detector (unreliable)"
    )
    ax.text(0.02, 0.98, interpretation_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', 
            facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()

