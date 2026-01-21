"""
Main evaluation script for comparing explainability methods.
Runs the complete pipeline: load model, compute explanations, evaluate plausibility.
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime

from src import config
from src.data_loader import CUB200Dataset, get_transforms, load_class_names
from src.model import load_model, create_model
from src.explainability import ExplainabilityMethods, visualize_all_methods
from src.evaluation import PlausibilityEvaluator


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert normalized tensor back to displayable image."""
    mean = torch.tensor(config.IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(config.IMAGENET_STD).view(3, 1, 1)
    
    img = tensor.cpu() * std + mean
    img = img.clamp(0, 1)
    img = img.permute(1, 2, 0).numpy()
    
    return img


def run_evaluation(
    model_path: str = None,
    num_samples: int = 100,
    methods: List[str] = None,
    save_visualizations: bool = True,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, PlausibilityEvaluator]:
    """
    Run the complete evaluation pipeline.
    
    Args:
        model_path: Path to trained model checkpoint
        num_samples: Number of test images to evaluate
        methods: List of explainability methods to use
        save_visualizations: Whether to save visualization images
        random_seed: Random seed for reproducibility
        
    Returns:
        Summary DataFrame and PlausibilityEvaluator object
    """
    # Setup
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if methods is None:
        methods = config.EXPLAINABILITY_METHODS
    
    # Load model
    print("\nLoading model...")
    if model_path and os.path.exists(model_path):
        model = load_model(model_path, num_classes=config.NUM_CLASSES, device=device)
    else:
        print("No trained model found. Using pretrained model for demonstration.")
        model = create_model(num_classes=config.NUM_CLASSES, pretrained=True, device=device)
    
    model.eval()
    
    # Load test dataset with part annotations
    print("\nLoading dataset...")
    test_dataset = CUB200Dataset(
        is_train=False,
        transform=get_transforms(is_train=False),
        load_parts=True
    )
    
    class_names = load_class_names()
    
    # Select random subset for evaluation
    if num_samples < len(test_dataset):
        indices = np.random.choice(len(test_dataset), num_samples, replace=False)
        test_subset = Subset(test_dataset, indices)
    else:
        test_subset = test_dataset
        num_samples = len(test_dataset)
    
    print(f"Evaluating on {num_samples} test images")
    
    # Initialize explainability methods
    print("\nInitializing explainability methods...")
    explainer = ExplainabilityMethods(model, device)
    
    # Initialize evaluator
    evaluator = PlausibilityEvaluator()
    
    # Create visualization directory
    vis_dir = config.VISUALIZATIONS_DIR / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if save_visualizations:
        vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluation loop
    print("\nRunning evaluation...")
    for idx in tqdm(range(len(test_subset))):
        sample = test_subset[idx]
        
        image_tensor = sample['image'].unsqueeze(0).to(device)
        true_label = sample['label']
        image_id = sample['image_id']
        part_mask = sample['part_mask'].numpy()
        
        # Get model prediction
        with torch.no_grad():
            output = model(image_tensor)
            predicted_class = output.argmax(dim=1).item()
        
        # Compute attributions for all methods
        try:
            attributions = explainer.get_all_attributions(
                image_tensor,
                predicted_class,  # Use predicted class for attribution
                methods=methods
            )
        except Exception as e:
            print(f"\nError computing attributions for image {image_id}: {e}")
            continue
        
        # Evaluate
        evaluator.add_result(
            image_id=image_id,
            attributions=attributions,
            part_mask=part_mask,
            predicted_class=predicted_class,
            true_class=true_label
        )
        
        # Save visualization for first 10 images
        if save_visualizations and idx < 10:
            img_np = denormalize_image(sample['image'])
            vis_path = vis_dir / f"sample_{idx}_id{image_id}.png"
            
            try:
                visualize_all_methods(
                    img_np,
                    attributions,
                    part_mask=part_mask,
                    save_path=str(vis_path),
                    show=False
                )
            except Exception as e:
                print(f"\nVisualization error: {e}")
    
    # Get summary results
    print("\nComputing summary statistics...")
    summary_df = evaluator.get_summary()
    
    # Save results
    results_path = config.RESULTS_DIR / f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    summary_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")
    
    # Save detailed results
    detailed_df = evaluator.get_detailed_results()
    detailed_path = config.RESULTS_DIR / f"detailed_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    detailed_df.to_csv(detailed_path, index=False)
    
    return summary_df, evaluator


def analyze_results(evaluator: PlausibilityEvaluator) -> None:
    """
    Analyze and visualize evaluation results.
    
    This function helps answer: "Are the results due to the model or the explanation method?"
    """
    print("\n" + "=" * 60)
    print("ANALYSIS OF RESULTS")
    print("=" * 60)
    
    # Overall summary
    summary = evaluator.get_summary()
    print("\n1. OVERALL PERFORMANCE BY METHOD")
    print("-" * 40)
    
    # Display key metrics
    key_metrics = ['pointing_game_mean', 'ebpg_mean', 'auc_roc_mean', 'average_precision_mean']
    
    for _, row in summary.iterrows():
        print(f"\n{row['method'].upper()}")
        for metric in key_metrics:
            if metric in row:
                std_metric = metric.replace('_mean', '_std')
                print(f"  {metric.replace('_mean', '')}: {row[metric]:.4f} ± {row.get(std_metric, 0):.4f}")
    
    # Compare correct vs incorrect predictions
    print("\n\n2. ANALYSIS: MODEL vs EXPLANATION METHOD")
    print("-" * 40)
    print("Comparing explanations for correct vs incorrect predictions...")
    
    correct_df, incorrect_df = evaluator.get_summary_by_correctness()
    
    if not correct_df.empty and not incorrect_df.empty:
        print("\nCorrect Predictions (EBPG):")
        for _, row in correct_df.iterrows():
            print(f"  {row['method']}: {row.get('ebpg_mean', 'N/A'):.4f}")
        
        print("\nIncorrect Predictions (EBPG):")
        for _, row in incorrect_df.iterrows():
            print(f"  {row['method']}: {row.get('ebpg_mean', 'N/A'):.4f}")
        
        print("\n** INTERPRETATION **")
        print("If explanations are better for correct predictions, this suggests")
        print("the explanation quality is tied to model behavior (faithful explanations).")
        print("If there's no difference, the explanation method may not be capturing")
        print("what the model actually uses for prediction.")
    
    # Statistical comparison between methods
    print("\n\n3. STATISTICAL COMPARISON BETWEEN METHODS")
    print("-" * 40)
    
    comparison = evaluator.compare_methods(metric='ebpg')
    if not comparison.empty:
        print("\nPairwise comparisons (Wilcoxon signed-rank test on EBPG):")
        for _, row in comparison.iterrows():
            sig = "***" if row['wilcoxon_pvalue'] < 0.001 else "**" if row['wilcoxon_pvalue'] < 0.01 else "*" if row['wilcoxon_pvalue'] < 0.05 else ""
            print(f"  {row['method1']} vs {row['method2']}: diff={row['mean_diff']:.4f}, p={row['wilcoxon_pvalue']:.4f} {sig}")


def create_visualizations(evaluator: PlausibilityEvaluator) -> None:
    """Create summary visualizations."""
    summary = evaluator.get_summary()
    detailed = evaluator.get_detailed_results()
    
    if summary.empty:
        print("No results to visualize.")
        return
    
    # 1. Bar plot of key metrics by method
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics_to_plot = ['pointing_game', 'ebpg', 'auc_roc', 'average_precision']
    titles = ['Pointing Game', 'Energy-Based Pointing Game', 'AUC-ROC', 'Average Precision']
    
    for ax, metric, title in zip(axes.flatten(), metrics_to_plot, titles):
        mean_col = f'{metric}_mean'
        std_col = f'{metric}_std'
        
        if mean_col in summary.columns:
            methods = summary['method'].values
            means = summary[mean_col].values
            stds = summary.get(std_col, np.zeros_like(means))
            
            bars = ax.bar(range(len(methods)), means, yerr=stds, capsize=5)
            ax.set_xticks(range(len(methods)))
            ax.set_xticklabels([m.replace('_', '\n') for m in methods], rotation=0)
            ax.set_ylabel('Score')
            ax.set_title(title)
            ax.set_ylim(0, 1)
            
            # Add value labels
            for bar, mean in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{mean:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(config.RESULTS_DIR / 'metrics_comparison.png', dpi=150)
    plt.close()
    
    # 2. Box plot of metrics distribution
    if not detailed.empty:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        for ax, metric, title in zip(axes.flatten(), metrics_to_plot, titles):
            if metric in detailed.columns:
                sns.boxplot(data=detailed, x='method', y=metric, ax=ax)
                ax.set_xticklabels([t.get_text().replace('_', '\n') for t in ax.get_xticklabels()])
                ax.set_title(title)
                ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(config.RESULTS_DIR / 'metrics_distribution.png', dpi=150)
        plt.close()
    
    # 3. Correct vs Incorrect predictions comparison
    if 'correct' in detailed.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.barplot(
            data=detailed,
            x='method',
            y='ebpg',
            hue='correct',
            ax=ax
        )
        ax.set_xticklabels([t.get_text().replace('_', '\n') for t in ax.get_xticklabels()])
        ax.set_title('EBPG: Correct vs Incorrect Predictions')
        ax.set_ylabel('Energy-Based Pointing Game')
        ax.legend(title='Correct Prediction')
        
        plt.tight_layout()
        plt.savefig(config.RESULTS_DIR / 'correct_vs_incorrect.png', dpi=150)
        plt.close()
    
    print(f"Visualizations saved to {config.RESULTS_DIR}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate explainability methods on CUB-200-2011')
    parser.add_argument('--model_path', type=str, default=None, help='Path to trained model')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of test samples to evaluate')
    parser.add_argument('--methods', nargs='+', default=None, help='Explainability methods to use')
    parser.add_argument('--no_visualizations', action='store_true', help='Skip saving visualizations')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Run evaluation
    summary_df, evaluator = run_evaluation(
        model_path=args.model_path,
        num_samples=args.num_samples,
        methods=args.methods,
        save_visualizations=not args.no_visualizations,
        random_seed=args.seed
    )
    
    # Analyze results
    analyze_results(evaluator)
    
    # Create visualizations
    create_visualizations(evaluator)
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"\nResults saved to: {config.RESULTS_DIR}")
    print(f"Visualizations saved to: {config.VISUALIZATIONS_DIR}")


if __name__ == "__main__":
    main()
