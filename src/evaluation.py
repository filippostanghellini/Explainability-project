"""
Evaluation metrics for measuring explanation plausibility.
Compares model explanations with ground-truth part annotations.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score
import pandas as pd
from tqdm import tqdm

from . import config


def normalize_attribution(attr_map: np.ndarray) -> np.ndarray:
    """Normalize attribution map to [0, 1] range."""
    attr_min = attr_map.min()
    attr_max = attr_map.max()
    
    if attr_max - attr_min > 1e-8:
        return (attr_map - attr_min) / (attr_max - attr_min)
    else:
        return np.zeros_like(attr_map)


def compute_pointing_game(
    attribution_map: np.ndarray,
    part_mask: np.ndarray,
    top_k_percent: float = 5.0
) -> float:
    """
    Compute Pointing Game metric.
    
    Checks if the maximum attribution falls within the ground truth region.
    
    Args:
        attribution_map: Attribution map (H, W)
        part_mask: Binary ground truth mask (H, W)
        top_k_percent: Not used in classic pointing game (uses max)
        
    Returns:
        1.0 if max attribution is within part mask, 0.0 otherwise
    """
    # Find the location of maximum attribution
    max_idx = np.unravel_index(np.argmax(attribution_map), attribution_map.shape)
    
    # Check if this location is within the ground truth mask
    if part_mask[max_idx] > 0:
        return 1.0
    else:
        return 0.0


def compute_energy_based_pointing_game(
    attribution_map: np.ndarray,
    part_mask: np.ndarray
) -> float:
    """
    Compute Energy-based Pointing Game (EBPG).
    
    Measures the proportion of total attribution energy that falls
    within the ground truth region.
    
    Args:
        attribution_map: Attribution map (H, W)
        part_mask: Binary ground truth mask (H, W)
        
    Returns:
        Ratio of attribution energy in GT region to total energy
    """
    # Take absolute values
    attr_abs = np.abs(attribution_map)
    
    # Total energy
    total_energy = attr_abs.sum()
    
    if total_energy < 1e-8:
        return 0.0
    
    # Energy within ground truth region
    gt_energy = (attr_abs * part_mask).sum()
    
    return gt_energy / total_energy


def compute_iou_at_threshold(
    attribution_map: np.ndarray,
    part_mask: np.ndarray,
    top_k_percent: float = 10.0
) -> float:
    """
    Compute Intersection over Union (IoU) at a given threshold.
    
    Binarizes the attribution map by taking top-k% pixels and computes
    IoU with the ground truth mask.
    
    Args:
        attribution_map: Attribution map (H, W)
        part_mask: Binary ground truth mask (H, W)
        top_k_percent: Percentage of top pixels to consider
        
    Returns:
        IoU score
    """
    # Normalize attribution map
    attr_norm = normalize_attribution(attribution_map)
    
    # Get threshold for top k%
    threshold = np.percentile(attr_norm.flatten(), 100 - top_k_percent)
    
    # Binarize attribution map
    attr_binary = (attr_norm >= threshold).astype(np.float32)
    
    # Compute IoU
    intersection = (attr_binary * part_mask).sum()
    union = np.maximum(attr_binary, part_mask).sum()
    
    if union < 1e-8:
        return 0.0
    
    return intersection / union


def compute_average_precision(
    attribution_map: np.ndarray,
    part_mask: np.ndarray
) -> float:
    """
    Compute Average Precision (AP) treating attribution as prediction scores.
    
    Args:
        attribution_map: Attribution map (H, W)
        part_mask: Binary ground truth mask (H, W)
        
    Returns:
        Average Precision score
    """
    # Flatten arrays
    attr_flat = normalize_attribution(attribution_map).flatten()
    mask_flat = part_mask.flatten()
    
    # Check if there are any positive labels
    if mask_flat.sum() == 0:
        return 0.0
    
    try:
        ap = average_precision_score(mask_flat, attr_flat)
        return ap
    except:
        return 0.0


def compute_auc_roc(
    attribution_map: np.ndarray,
    part_mask: np.ndarray
) -> float:
    """
    Compute Area Under ROC Curve.
    
    Args:
        attribution_map: Attribution map (H, W)
        part_mask: Binary ground truth mask (H, W)
        
    Returns:
        AUC-ROC score
    """
    # Flatten arrays
    attr_flat = normalize_attribution(attribution_map).flatten()
    mask_flat = part_mask.flatten()
    
    # Check if there are both positive and negative samples
    if mask_flat.sum() == 0 or mask_flat.sum() == len(mask_flat):
        return 0.5
    
    try:
        auc = roc_auc_score(mask_flat, attr_flat)
        return auc
    except:
        return 0.5


def compute_spearman_correlation(
    attribution_map: np.ndarray,
    part_mask: np.ndarray
) -> float:
    """
    Compute Spearman rank correlation between attribution and mask.
    
    Args:
        attribution_map: Attribution map (H, W)
        part_mask: Binary ground truth mask (H, W)
        
    Returns:
        Spearman correlation coefficient
    """
    attr_flat = attribution_map.flatten()
    mask_flat = part_mask.flatten()
    
    try:
        corr, _ = spearmanr(attr_flat, mask_flat)
        return corr if not np.isnan(corr) else 0.0
    except:
        return 0.0


def compute_mass_accuracy(
    attribution_map: np.ndarray,
    part_mask: np.ndarray,
    top_k_percent: float = 10.0
) -> float:
    """
    Compute Mass Accuracy - fraction of top-k% attribution inside GT region.
    
    Args:
        attribution_map: Attribution map (H, W)
        part_mask: Binary ground truth mask (H, W)
        top_k_percent: Percentage of top pixels to consider
        
    Returns:
        Mass accuracy score
    """
    # Normalize and get top-k threshold
    attr_norm = normalize_attribution(attribution_map)
    threshold = np.percentile(attr_norm.flatten(), 100 - top_k_percent)
    
    # Get indices of top-k pixels
    top_k_mask = attr_norm >= threshold
    
    # Count how many are in GT region
    top_k_in_gt = (top_k_mask * part_mask).sum()
    top_k_total = top_k_mask.sum()
    
    if top_k_total == 0:
        return 0.0
    
    return top_k_in_gt / top_k_total


def compute_relevance_rank_accuracy(
    attribution_map: np.ndarray,
    part_mask: np.ndarray,
    top_k_percent: float = 10.0
) -> float:
    """
    Compute Relevance Rank Accuracy.
    
    Measures whether high-attribution pixels tend to be in GT regions.
    
    Args:
        attribution_map: Attribution map (H, W)
        part_mask: Binary ground truth mask (H, W)
        top_k_percent: Percentage of top pixels to consider
        
    Returns:
        Relevance rank accuracy
    """
    attr_flat = attribution_map.flatten()
    mask_flat = part_mask.flatten()
    
    # Get number of GT pixels
    n_gt = int(mask_flat.sum())
    if n_gt == 0:
        return 0.0
    
    # Get indices sorted by attribution (descending)
    sorted_indices = np.argsort(attr_flat)[::-1]
    
    # Count how many of top-n_gt are in GT
    top_n_indices = sorted_indices[:n_gt]
    hits = mask_flat[top_n_indices].sum()
    
    return hits / n_gt


def compute_all_metrics(
    attribution_map: np.ndarray,
    part_mask: np.ndarray,
    top_k_percentages: List[float] = None
) -> Dict[str, float]:
    """
    Compute all plausibility metrics.
    
    Args:
        attribution_map: Attribution map (H, W)
        part_mask: Binary ground truth mask (H, W)
        top_k_percentages: List of top-k percentages for IoU and Mass Accuracy
        
    Returns:
        Dictionary of metric names to values
    """
    if top_k_percentages is None:
        top_k_percentages = config.TOP_K_PERCENT
    
    metrics = {}
    
    # Core metrics
    metrics['pointing_game'] = compute_pointing_game(attribution_map, part_mask)
    metrics['ebpg'] = compute_energy_based_pointing_game(attribution_map, part_mask)
    metrics['auc_roc'] = compute_auc_roc(attribution_map, part_mask)
    metrics['average_precision'] = compute_average_precision(attribution_map, part_mask)
    metrics['spearman_correlation'] = compute_spearman_correlation(attribution_map, part_mask)
    metrics['relevance_rank_accuracy'] = compute_relevance_rank_accuracy(attribution_map, part_mask)
    
    # Threshold-dependent metrics
    for k in top_k_percentages:
        metrics[f'iou_top{k}'] = compute_iou_at_threshold(attribution_map, part_mask, k)
        metrics[f'mass_accuracy_top{k}'] = compute_mass_accuracy(attribution_map, part_mask, k)
    
    return metrics


def evaluate_single_image(
    attributions: Dict[str, np.ndarray],
    part_mask: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate all attribution methods on a single image.
    
    Args:
        attributions: Dictionary mapping method names to attribution maps
        part_mask: Ground truth part mask
        
    Returns:
        Nested dictionary: method -> metric -> value
    """
    results = {}
    
    for method_name, attr_map in attributions.items():
        if attr_map is not None:
            results[method_name] = compute_all_metrics(attr_map, part_mask)
        else:
            results[method_name] = None
    
    return results


def aggregate_results(
    all_results: List[Dict[str, Dict[str, float]]]
) -> pd.DataFrame:
    """
    Aggregate evaluation results across multiple images.
    
    Args:
        all_results: List of results from evaluate_single_image
        
    Returns:
        DataFrame with mean and std for each method and metric
    """
    # Collect all metrics per method
    method_metrics = {}
    
    for result in all_results:
        for method_name, metrics in result.items():
            if metrics is None:
                continue
            
            if method_name not in method_metrics:
                method_metrics[method_name] = {k: [] for k in metrics.keys()}
            
            for metric_name, value in metrics.items():
                method_metrics[method_name][metric_name].append(value)
    
    # Compute mean and std
    rows = []
    for method_name, metrics in method_metrics.items():
        row = {'method': method_name}
        for metric_name, values in metrics.items():
            row[f'{metric_name}_mean'] = np.mean(values)
            row[f'{metric_name}_std'] = np.std(values)
        rows.append(row)
    
    return pd.DataFrame(rows)


def compare_methods_statistical(
    all_results: List[Dict[str, Dict[str, float]]],
    metric_name: str = 'ebpg'
) -> pd.DataFrame:
    """
    Perform statistical comparison between methods.
    
    Args:
        all_results: List of results from evaluate_single_image
        metric_name: Metric to use for comparison
        
    Returns:
        DataFrame with pairwise comparison results
    """
    from scipy.stats import wilcoxon, ttest_rel
    
    # Collect metric values per method
    method_values = {}
    
    for result in all_results:
        for method_name, metrics in result.items():
            if metrics is None:
                continue
            
            if method_name not in method_values:
                method_values[method_name] = []
            
            if metric_name in metrics:
                method_values[method_name].append(metrics[metric_name])
    
    # Pairwise comparisons
    methods = list(method_values.keys())
    comparisons = []
    
    for i, method1 in enumerate(methods):
        for j, method2 in enumerate(methods):
            if i >= j:
                continue
            
            values1 = np.array(method_values[method1])
            values2 = np.array(method_values[method2])
            
            # Ensure same length (paired test)
            min_len = min(len(values1), len(values2))
            values1 = values1[:min_len]
            values2 = values2[:min_len]
            
            try:
                t_stat, t_pval = ttest_rel(values1, values2)
                w_stat, w_pval = wilcoxon(values1, values2)
            except:
                t_stat, t_pval, w_stat, w_pval = np.nan, np.nan, np.nan, np.nan
            
            comparisons.append({
                'method1': method1,
                'method2': method2,
                'mean_diff': np.mean(values1) - np.mean(values2),
                't_statistic': t_stat,
                't_pvalue': t_pval,
                'wilcoxon_statistic': w_stat,
                'wilcoxon_pvalue': w_pval
            })
    
    return pd.DataFrame(comparisons)


class PlausibilityEvaluator:
    """
    Class for systematic plausibility evaluation.
    """
    
    def __init__(self, top_k_percentages: List[float] = None):
        """
        Initialize evaluator.
        
        Args:
            top_k_percentages: List of top-k percentages for metrics
        """
        self.top_k_percentages = top_k_percentages or config.TOP_K_PERCENT
        self.results = []
    
    def add_result(
        self,
        image_id: int,
        attributions: Dict[str, np.ndarray],
        part_mask: np.ndarray,
        predicted_class: int = None,
        true_class: int = None
    ):
        """Add evaluation result for one image."""
        eval_result = evaluate_single_image(attributions, part_mask)
        
        self.results.append({
            'image_id': image_id,
            'predicted_class': predicted_class,
            'true_class': true_class,
            'correct_prediction': predicted_class == true_class if predicted_class is not None else None,
            'metrics': eval_result
        })
    
    def get_summary(self) -> pd.DataFrame:
        """Get aggregated summary of all results."""
        all_metrics = [r['metrics'] for r in self.results]
        return aggregate_results(all_metrics)
    
    def get_summary_by_correctness(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get separate summaries for correct and incorrect predictions.
        
        This helps analyze: Are the results due to the model or the explanation method?
        """
        correct_metrics = [
            r['metrics'] for r in self.results 
            if r['correct_prediction'] == True
        ]
        incorrect_metrics = [
            r['metrics'] for r in self.results 
            if r['correct_prediction'] == False
        ]
        
        correct_df = aggregate_results(correct_metrics) if correct_metrics else pd.DataFrame()
        incorrect_df = aggregate_results(incorrect_metrics) if incorrect_metrics else pd.DataFrame()
        
        return correct_df, incorrect_df
    
    def get_detailed_results(self) -> pd.DataFrame:
        """Get detailed results for each image."""
        rows = []
        
        for result in self.results:
            for method_name, metrics in result['metrics'].items():
                if metrics is None:
                    continue
                
                row = {
                    'image_id': result['image_id'],
                    'predicted_class': result['predicted_class'],
                    'true_class': result['true_class'],
                    'correct': result['correct_prediction'],
                    'method': method_name
                }
                row.update(metrics)
                rows.append(row)
        
        return pd.DataFrame(rows)
    
    def compare_methods(self, metric: str = 'ebpg') -> pd.DataFrame:
        """Statistical comparison between methods."""
        all_metrics = [r['metrics'] for r in self.results]
        return compare_methods_statistical(all_metrics, metric)


def sanity_check_explanations(
    model: torch.nn.Module,
    explainer,
    input_tensor: torch.Tensor,
    target_class: int,
    methods: List[str] = None,
    n_random_seeds: int = 5
) -> Dict[str, List[float]]:
    """
    Sanity check for explanations.
    
    Randomizes model weights and recomputes explanations.
    The attributions should change significantly with random weights.
    
    Args:
        model: Trained PyTorch model
        explainer: ExplainabilityMethods instance
        input_tensor: Input image tensor (1, C, H, W)
        target_class: Target class for attribution
        methods: List of explanation methods to test
        n_random_seeds: Number of random initializations to test
        
    Returns:
        Dictionary mapping method names to lists of correlation changes
    """
    if methods is None:
        methods = config.EXPLAINABILITY_METHODS
    
    device = next(model.parameters()).device
    original_state = {name: param.detach().clone() for name, param in model.named_parameters()}
    original_training = model.training

    # Ensure input is on the correct device
    if input_tensor.device != device:
        input_tensor = input_tensor.to(device)

    # Ensure model is in eval mode
    model.eval()

    # Get original attributions (use explainer as provided)
    original_attrs = explainer.get_all_attributions(input_tensor, target_class, methods)
    
    correlation_changes = {method: [] for method in methods}
    
    print(f"Running sanity check with {n_random_seeds} random initializations...")
    
    for seed in tqdm(range(n_random_seeds)):
        # Randomize model weights
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        with torch.no_grad():
            for param in model.parameters():
                if len(param.shape) > 1:  # Skip biases
                    torch.nn.init.kaiming_uniform_(param, a=0, mode='fan_in', nonlinearity='relu')
                else:
                    torch.nn.init.uniform_(param, 0, 1)
        
        # CRITICAL: Ensure model is in eval mode after weight modification
        model.eval()
        
        # Re-create explainer to clear cached hooks/states
        from src.explainability import ExplainabilityMethods
        random_explainer = ExplainabilityMethods(model, device)
        
        # Compute attributions with random weights using NEW explainer
        random_attrs = random_explainer.get_all_attributions(input_tensor, target_class, methods)
        
        # Compute correlation between original and random attributions
        for method in methods:
            if original_attrs[method] is not None and random_attrs[method] is not None:
                orig_flat = original_attrs[method].flatten()
                rand_flat = random_attrs[method].flatten()
                
                # Compute Spearman correlation
                try:
                    corr, _ = spearmanr(orig_flat, rand_flat)
                    corr = corr if not np.isnan(corr) else 0.0
                except:
                    corr = 0.0
                
                correlation_changes[method].append(corr)
    
    # Restore original weights
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in original_state:
                param.copy_(original_state[name])
    
    # Restore model to original training/eval mode
    model.train(original_training)
    
    return correlation_changes


def print_sanity_check_results(
    sanity_results: Dict[str, List[float]]
) -> None:
    """
    Print and analyze sanity check results.
    
    Expected behavior:
    - Correlations should be LOW (close to 0)
    - This indicates explanations are NOT just random noise
    
    Args:
        sanity_results: Results from sanity_check_explanations
    """
    print("\n" + "="*60)
    print("SANITY CHECK RESULTS")
    print("="*60)
    print("\nExpected: Correlations should be LOW (close to 0)")
    print("If correlations are HIGH, explanations may be artifacts\n")
    
    for method, correlations in sanity_results.items():
        if not correlations:
            print(f"{method:20s}: NO DATA")
            continue
        
        mean_corr = np.mean(correlations)
        std_corr = np.std(correlations)
        min_corr = np.min(correlations)
        max_corr = np.max(correlations)
        
        # Interpretation
        if mean_corr < 0.1:
            status = "✓ PASS (good - explanations are not random)"
        elif mean_corr < 0.3:
            status = "⚠ WARNING (correlations are higher than expected)"
        else:
            status = "✗ FAIL (explanations may be artifacts)"
        
        print(f"{method:20s}")
        print(f"  Mean correlation:  {mean_corr:7.4f} ± {std_corr:.4f}")
        print(f"  Range:             [{min_corr:.4f}, {max_corr:.4f}]")
        print(f"  Status:            {status}\n")


class SanityCheckEvaluator:
    """
    Systematic sanity check evaluator for multiple images.
    """
    
    def __init__(self, model: torch.nn.Module, explainer, n_random_seeds: int = 5):
        """
        Initialize sanity check evaluator.
        
        Args:
            model: Trained PyTorch model
            explainer: ExplainabilityMethods instance
            n_random_seeds: Number of random initializations per image
        """
        self.model = model
        self.explainer = explainer
        self.n_random_seeds = n_random_seeds
        self.results = []
    
    def evaluate_batch(
        self,
        input_tensors: List[torch.Tensor],
        target_classes: List[int],
        methods: List[str] = None
    ):
        """
        Run sanity check on a batch of images.
        
        Args:
            input_tensors: List of input image tensors
            target_classes: List of target classes
            methods: List of explanation methods to test
        """
        for i, (input_tensor, target_class) in enumerate(zip(input_tensors, target_classes)):
            result = sanity_check_explanations(
                self.model,
                self.explainer,
                input_tensor,
                target_class,
                methods,
                self.n_random_seeds
            )
            
            self.results.append({
                'image_idx': i,
                'target_class': target_class,
                'correlations': result
            })
    
    def get_summary(self) -> pd.DataFrame:
        """Get aggregated summary of sanity check results."""
        rows = []
        
        for result in self.results:
            for method, correlations in result['correlations'].items():
                if correlations:
                    rows.append({
                        'image_idx': result['image_idx'],
                        'method': method,
                        'mean_correlation': np.mean(correlations),
                        'std_correlation': np.std(correlations),
                        'min_correlation': np.min(correlations),
                        'max_correlation': np.max(correlations)
                    })
        
        return pd.DataFrame(rows)


def cascade_randomization_test(
    model: torch.nn.Module,
    explainer,
    input_tensor: torch.Tensor,
    target_class: int,
    methods: List[str] = None,
    n_random_seeds: int = 3
) -> Dict[str, Dict[str, List[float]]]:
    """
    Cascade randomization test (Adebayo et al., 2018).
    
    Progressively randomizes layers from top (logit) to bottom (input),
    measuring how explanations change. A faithful method should show
    decreasing correlation as higher layers are randomized.
    
    This tests whether the explanation method is sensitive to the learned
    parameters at different depths of the network.
    
    Args:
        model: Trained PyTorch model (ResNet-50)
        explainer: ExplainabilityMethods instance
        input_tensor: Input image tensor (1, C, H, W)
        target_class: Target class for attribution
        methods: List of explanation methods to test
        n_random_seeds: Number of random seeds per layer configuration
        
    Returns:
        Dictionary: {method: {layer_name: [correlations]}}
    """
    if methods is None:
        methods = config.EXPLAINABILITY_METHODS
    
    device = next(model.parameters()).device
    original_training = model.training

    # Ensure input is on the correct device
    if input_tensor.device != device:
        input_tensor = input_tensor.to(device)
    
    # Save original state
    original_state = {name: param.clone() for name, param in model.named_parameters()}
    
    # Get original attributions with fully trained model
    print("Computing attributions with original trained model...")
    original_attrs = explainer.get_all_attributions(input_tensor, target_class, methods)
    
    # Define layer groups for ResNet-50 (from logit to input)
    # We'll randomize progressively: fc -> layer4 -> layer3 -> layer2 -> layer1 -> conv1
    layer_groups = [
        ('logit', ['fc']),
        ('layer4', ['layer4']),
        ('layer3', ['layer3']),
        ('layer2', ['layer2']),
        ('layer1', ['layer1']),
        ('conv1', ['conv1', 'bn1'])
    ]
    
    # Store results: {method: {layer_stage: [correlations]}}
    results = {method: {} for method in methods}
    
    # Cumulative randomization: each stage includes previous stages
    randomized_layers = []
    
    for stage_name, layer_names in layer_groups:
        randomized_layers.extend(layer_names)
        
        print(f"\nRandomizing up to {stage_name} (layers: {', '.join(randomized_layers)})")
        
        # Initialize correlation storage for this stage
        for method in methods:
            results[method][stage_name] = []
        
        # Run multiple seeds for robustness
        for seed in range(n_random_seeds):
            # Restore original weights
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in original_state:
                        param.copy_(original_state[name])
            
            # Randomize specific layers
            torch.manual_seed(seed)
            with torch.no_grad():
                for name, param in model.named_parameters():
                    # Check if this parameter belongs to a layer we want to randomize
                    should_randomize = any(layer in name for layer in randomized_layers)
                    
                    if should_randomize:
                        if len(param.shape) > 1:  # Weights
                            torch.nn.init.kaiming_uniform_(param, a=0, mode='fan_in', nonlinearity='relu')
                        else:  # Biases
                            torch.nn.init.uniform_(param, 0, 1)
            
            # Compute attributions with partially randomized model
            try:
                # CRITICAL: Ensure model is in eval mode after weight modification
                model.eval()
                
                # Re-create explainer to clear cached hooks/states
                from src.explainability import ExplainabilityMethods
                random_explainer = ExplainabilityMethods(model, device)
                
                random_attrs = random_explainer.get_all_attributions(input_tensor, target_class, methods)
                
                # Compute correlation for each method
                for method in methods:
                    if original_attrs[method] is not None and random_attrs[method] is not None:
                        orig_flat = original_attrs[method].flatten()
                        rand_flat = random_attrs[method].flatten()
                        
                        try:
                            corr, _ = spearmanr(orig_flat, rand_flat)
                            corr = corr if not np.isnan(corr) else 0.0
                        except:
                            corr = 0.0
                        
                        results[method][stage_name].append(corr)
            except Exception as e:
                print(f"  Error at {stage_name}, seed {seed}: {e}")
    
    # Restore original weights
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in original_state:
                param.copy_(original_state[name])
    
    # Restore model to original training/eval mode
    model.train(original_training)
    
    return results
