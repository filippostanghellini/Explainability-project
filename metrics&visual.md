# Explainability Metrics & Visualizations Guide

This document provides a comprehensive overview of all metrics and visualizations used in the XAI evaluation framework for the CUB-200-2011 bird classification project.

---

## Table of Contents

1. [Metrics Overview](#metrics-overview)
2. [Core Plausibility Metrics](#core-plausibility-metrics)
3. [Threshold-Based Metrics](#threshold-based-metrics)
4. [Visualization Functions](#visualization-functions)
5. [Analysis Types](#analysis-types)
6. [Literature References](#literature-references)

---

## Metrics Overview

All metrics measure **plausibility** by comparing explanation attribution maps with ground-truth part annotations from the CUB-200-2011 dataset. These metrics answer: *"Does the explanation focus on the bird parts annotated by humans?"*

### Metric Categories

1. **Location-Based**: Where does the explanation point? (Pointing Game)
2. **Energy-Based**: How much explanation energy is on the bird? (EBPG)
3. **Ranking-Based**: How well does attribution rank pixels? (AUC-ROC, Average Precision)
4. **Overlap-Based**: How much do explanation regions overlap with bird parts? (IoU@k%)

---

## Core Plausibility Metrics

### 1. Pointing Game (PG)

**Purpose**: Tests if the maximum attribution point falls within ground-truth regions.

**Computation**:
```python
max_point = argmax(attribution_map)
score = 1.0 if ground_truth_mask[max_point] > 0 else 0.0
```

**Range**: `{0.0, 1.0}` (binary)

**Interpretation**:
- `1.0`: The most important pixel identified by the explanation is on a bird part ✓
- `0.0`: The most important pixel is outside bird parts ✗

**Literature**: Zhang et al. (2018) - "Top-Down Neural Attention by Excitation Backprop"

**Why this metric?**: 
- Simple and interpretable
- Tests if the method identifies the "most important" location correctly
- Aligns with human intuition: "Where is the model looking?"

**Visualization**: 
- `create_pointing_game_visualization()` shows the max point overlaid on the image and ground truth

---

### 2. Energy-Based Pointing Game (EBPG)

**Purpose**: Measures the proportion of total attribution energy that falls within ground-truth regions.

**Computation**:
```python
total_energy = sum(|attribution_map|)
gt_energy = sum(|attribution_map| * ground_truth_mask)
score = gt_energy / total_energy
```

**Range**: `[0.0, 1.0]` (continuous)

**Interpretation**:
- `1.0`: All attribution energy is on bird parts (perfect plausibility) ✓
- `0.5`: Half the energy is on bird parts, half elsewhere
- `0.0`: No attribution energy on bird parts ✗

**Literature**: Wang et al. (2020) - Improved Pointing Game for weakly-supervised localization

**Why this metric?**:
- More fine-grained than binary Pointing Game
- Considers the entire attribution distribution, not just the maximum
- Robust to noisy attributions with multiple peaks
- Better reflects overall explanation quality

**Visualization**:
- `create_ebpg_visualization()` shows energy distribution inside/outside ground truth with pie charts

---

### 3. AUC-ROC (Area Under Receiver Operating Characteristic Curve)

**Purpose**: Measures how well the attribution map ranks pixels (bird part pixels should have higher attribution than background).

**Computation**:
- Treat attribution values as "scores" for binary classification
- Ground truth mask defines positive class (bird parts = 1, background = 0)
- Compute ROC curve and area under it

**Range**: `[0.0, 1.0]`

**Interpretation**:
- `1.0`: Perfect ranking - all bird pixels ranked higher than background ✓
- `0.5`: Random ranking (baseline)
- `< 0.5`: Worse than random (inverted) ✗

**Literature**: Standard machine learning evaluation metric (Fawcett, 2006)

**Why this metric?**:
- Well-established in ML community
- Threshold-agnostic (doesn't require choosing a cutoff)
- Robust to class imbalance (bird parts are typically small % of image)

**Visualization**:
- `create_roc_curve_comparison()` plots ROC curves for all methods on the same axes

---

### 4. Average Precision (AP)

**Purpose**: Summarizes the Precision-Recall curve; measures ranking quality with emphasis on high-precision regions.

**Computation**:
- Similar to AUC-ROC but uses Precision-Recall curve
- Weighted mean of precisions at each threshold
- More sensitive to rare positives (bird parts)

**Range**: `[0.0, 1.0]`

**Interpretation**:
- `1.0`: Perfect precision-recall trade-off ✓
- Random baseline: `proportion of positive pixels` (typically ~0.1-0.3 for bird parts)

**Literature**: Information retrieval standard metric

**Why this metric?**:
- **Better than AUC-ROC for imbalanced data** (bird parts are small)
- Emphasizes performance on the minority class (bird parts)
- More interpretable for detection/localization tasks

**Visualization**:
- `create_pr_curve_comparison()` plots Precision-Recall curves for all methods

---

## Threshold-Based Metrics

### 5. Intersection over Union at Top-k% (IoU@k%)

**Purpose**: Measures overlap between top-k% attribution pixels and ground-truth regions.

**Computation**:
```python
threshold = percentile(attribution_map, 100 - k)
predicted_mask = attribution_map >= threshold
intersection = sum(predicted_mask * ground_truth_mask)
union = sum(max(predicted_mask, ground_truth_mask))
score = intersection / union
```

**Evaluated at**: `k = [5%, 10%, 15%, 20%, 25%]` (configurable)

**Range**: `[0.0, 1.0]`

**Interpretation**:
- `1.0`: Perfect overlap between top-k% attributions and bird parts ✓
- `0.0`: No overlap ✗

**Literature**: Standard computer vision metric (Everingham et al., 2010 - Pascal VOC)

**Why this metric?**:
- Directly measures localization quality
- Multiple thresholds (k values) provide robustness analysis
- Shows how "focused" the explanation is at different granularities
- IoU@10% approximates "does the explanation focus on ~10% most important pixels?"

**Visualization**:
- `create_iou_threshold_curves()` shows IoU vs. threshold for each method (robustness analysis)

---

## Visualization Functions

All visualizations follow XAI literature best practices with publication-ready styling.

### 1. `visualize_all_methods()`

**Purpose**: Grid comparison of all explanation methods for a single image

**Shows**:
- Original image
- Ground truth part mask
- Attribution heatmap for each method (9 methods typically)

**Colormap**:
- **Sequential** (Inferno) for gradient-based methods (always positive)
- **Divergent** (RdBu) for perturbation-based methods (LIME, SHAP can be negative)

**Why**: Quick visual comparison of all methods at once

**File**: Core visualization for single-image analysis

---

### 2. `visualize_attribution_enhanced()`

**Purpose**: Detailed multi-view analysis of a single method

**Shows** (3-4 panels):
1. Original image
2. Attribution with contour lines (shows intensity levels)
3. Ground truth overlay (visual comparison)
4. Optional: Percentile-thresholded views (top 5%, 10%, 20%)

**Why**: Deep dive into one method's behavior with multiple perspectives

**Based on**: Montavon et al. (2019) - Layer-wise Relevance Propagation visualization guidelines

---

### 3. `create_roc_curve_comparison()`

**Purpose**: Compare ROC curves across all methods

**Shows**:
- ROC curve for each method
- AUC value in legend
- Random baseline (diagonal line)

**Why**: Standard ML visualization for binary classification performance

**Axes**: False Positive Rate (x) vs True Positive Rate (y)

---

### 4. `create_pr_curve_comparison()`

**Purpose**: Compare Precision-Recall curves across all methods

**Shows**:
- PR curve for each method  
- Average Precision (AP) value in legend
- Random baseline (horizontal line at class frequency)

**Why**: Better than ROC for imbalanced data (bird parts are small)

**Axes**: Recall (x) vs Precision (y)

---

### 5. `create_iou_threshold_curves()`

**Purpose**: Show robustness of IoU metric across different thresholds

**Shows**:
- IoU score vs. top-k% threshold for each method
- Multiple lines (one per method)

**Why**: 
- Reveals if a method is consistently good across thresholds
- Shows at which threshold each method performs best
- Identifies methods that are too "diffuse" (low IoU at all k) or too "focused" (high IoU only at very low k)

**Axes**: Top-k Percentage (x) vs IoU Score (y)

---

### 6. `create_pointing_game_visualization()`

**Purpose**: Visual explanation of Pointing Game metric

**Shows** (3 panels):
1. Max attribution point on heatmap (green star if hit, red X if miss)
2. Max point overlaid on ground truth region
3. Top-k% attribution region vs ground truth

**Why**: Makes the Pointing Game metric intuitive and verifiable

**Based on**: Zhang et al. (2018) visualization

---

### 7. `create_ebpg_visualization()`

**Purpose**: Visual explanation of EBPG metric

**Shows** (3 panels):
1. Attribution split by ground truth region (inside vs outside)
2. Pie chart of energy distribution
3. Bar chart comparing methods

**Why**: Shows how attribution energy is distributed spatially

---

### 8. `create_methods_ranking_heatmap()`

**Purpose**: Heatmap showing relative ranking of methods across all metrics

**Shows**:
- Methods (rows) vs Metrics (columns)
- Color intensity = normalized rank (darker = better)

**Why**: 
- Single view to compare all methods on all metrics
- Identifies consistently good/bad methods
- Reveals metric-specific strengths

**Colormap**: Divergent (RdYlGn) - Red = worse, Green = better

---

### 9. `create_metrics_radar_chart()`

**Purpose**: Spider chart comparing methods across multiple metrics

**Shows**:
- Polygon for each method
- Axes = different metrics (EBPG, AUC, IoU@10%, etc.)
- Larger area = better overall performance

**Why**: 
- Multi-dimensional comparison in 2D
- Visually intuitive "larger = better"
- Shows trade-offs between metrics

---

### 10. `create_model_vs_method_analysis()`

**Purpose**: Answer the key question: *"Are results due to the model or the explanation method?"*

**Shows** (3 panels):
1. **Box plots**: Explanation quality for correct vs incorrect predictions
2. **Difference plot**: (Correct - Incorrect) for each method
3. **Scatter plot**: Method performance vs model accuracy

**Interpretation**:
- **Positive difference** = Explanations are better when model is correct → Suggests explanations are **faithful** to model ✓
- **Zero difference** = Explanations don't change with correctness → Method may be **artifact-driven** ⚠
- **Negative difference** = Explanations worse when correct → Highly suspicious ✗

**Why this analysis?**:
- Critical for understanding if explanations are meaningful
- Separates model quality from explanation quality
- Based on Adebayo et al. (2018) - "Model Parameter Randomization Test"

---

### 11. `create_sanity_check_comparison()`

**Purpose**: Visualize sanity check results (Adebayo et al., 2018)

**Shows**:
- Original attributions (trained model)
- Randomized attributions (random weights)
- Correlation values

**Interpretation**:
- **Low correlation** (< 0.3): Explanation changes with random weights → **RELIABLE** ✓
- **High correlation** (> 0.6): Explanation similar with random weights → **UNRELIABLE** (possible artifact) ✗

**Why**: 
- Tests if explanations actually depend on trained model
- Detects "EdgeDetector" artifacts (methods that just highlight edges regardless of model)
- Essential validation step before trusting any explanation

**Based on**: Adebayo et al. (2018) - "Sanity Checks for Saliency Maps" (NeurIPS)

---

### 12. `create_cascade_randomization_plot()`

**Purpose**: Visualize cascade/layer-wise randomization test (Adebayo et al., 2018)

**Shows**:
- X-axis: Progressive layer randomization (logit → layer4 → layer3 → layer2 → layer1 → conv1)
- Y-axis: Spearman correlation between original and randomized explanations
- Multiple lines: One per explanation method

**Interpretation**:
- **Sharp drop at logit/layer4** (high layers): Method depends on high-level learned features → **RELIABLE** ✓
- **Gradual decrease**: Method uses features from multiple layers → **ACCEPTABLE** ⚠
- **High correlation throughout** (> 0.6): Method may be edge detector, independent of learned features → **UNRELIABLE** ✗

**Why**:
- **More rigorous than simple randomization**: Tests sensitivity at different network depths
- Shows WHICH layers the explanation depends on
- Faithful methods should be most sensitive to high-level layers (where class-specific features are learned)
- Edge detectors show high correlation regardless of which layers are randomized

**Based on**: Adebayo et al. (2018) - "Sanity Checks for Saliency Maps" Section 3.2

**Scientific Rigor**: This is considered the gold standard sanity check in the XAI literature

---

## Analysis Types

### 1. Single-Image Analysis (Section 5)

**Goal**: Understand how methods behave on individual examples

**Visualizations used**:
- `visualize_all_methods()` - Compare all methods
- `visualize_attribution_enhanced()` - Deep dive into one method
- Metrics table - Quantitative comparison

**When to use**: Debugging, qualitative analysis, method selection

---

### 2. Multi-Image Aggregation (Section 7)

**Goal**: Statistical comparison across many images

**Visualizations used**:
- Box plots - Distribution of metrics
- Heatmap - Method ranking
- Radar chart - Multi-metric comparison
- ROC/PR curves - Ranking quality

**Metrics computed**: Mean ± Std for all core + threshold metrics

**When to use**: Publication results, method benchmarking

---

### 3. Model vs Method Analysis (Section 8)

**Goal**: Determine if results are due to model quality or explanation method

**Visualizations used**:
- `create_model_vs_method_analysis()` with 3 sub-analyses

**Key insight**: Do explanations improve when the model is correct?

**When to use**: Understanding faithfulness, debugging suspicious results

---

### 4. Sanity Check (Section 10)

**Goal**: Verify explanations are not artifacts

**Two Tests Implemented**:

#### 4a. Complete Model Randomization (Basic Test)
- Randomize ALL model weights
- Recompute explanations
- Measure correlation with original explanations
- **Pass criteria**: Correlation < 0.3

**Visualizations used**:
- `create_sanity_check_comparison()` - Side-by-side comparison

#### 4b. Cascade Randomization (Advanced Test - NEW!)
- Progressively randomize layers: logit → layer4 → layer3 → layer2 → layer1 → conv1
- At each stage, measure correlation
- **Pass criteria**: 
  - Correlation should DECREASE as layers are randomized
  - Sharp drop at high layers (logit, layer4) indicates dependency on learned features
  - Methods that stay high (> 0.6) throughout are likely edge detectors

**Visualizations used**:
- `create_cascade_randomization_plot()` - Line plot showing correlation vs. layer depth

**Scientific Rigor**: The cascade test is considered more rigorous because:
1. Shows WHERE in the network the explanation is sensitive
2. Distinguishes between high-level (semantic) vs low-level (edge) features
3. Directly tests the claim that explanations reflect learned model behavior

**When to use**:
- Basic test: Quick check, suitable for initial validation
- Cascade test: Publication-quality validation, demonstrates scientific rigor

---

## Metrics Summary Table

| Metric | Range | Type | Best Practice Use | Literature |
|--------|-------|------|-------------------|------------|
| **Pointing Game** | {0, 1} | Location | Quick assessment of max point | Zhang et al. (2018) |
| **EBPG** | [0, 1] | Energy | Overall explanation quality | Wang et al. (2020) |
| **AUC-ROC** | [0, 1] | Ranking | Balanced class performance | Fawcett (2006) |
| **Average Precision** | [0, 1] | Ranking | Imbalanced data (preferred) | IR Standard |
| **IoU@5%** | [0, 1] | Overlap | Very focused explanations | Pascal VOC |
| **IoU@10%** | [0, 1] | Overlap | **Primary IoU metric** | Pascal VOC |
| **IoU@15%** | [0, 1] | Overlap | Moderate focus | Pascal VOC |
| **IoU@20%** | [0, 1] | Overlap | Broader explanations | Pascal VOC |
| **IoU@25%** | [0, 1] | Overlap | Robustness check | Pascal VOC |

---

## When to Use Each Metric

### Primary Metrics (Always Report)

1. **EBPG**: Best overall plausibility measure
   - Continuous (more sensitive than binary Pointing Game)
   - Considers entire attribution distribution
   - Most correlated with human judgment

2. **Average Precision**: Best ranking metric
   - Better than AUC-ROC for imbalanced data
   - Emphasizes performance on bird parts (minority class)

3. **IoU@10%**: Best localization metric
   - Standard in computer vision
   - 10% is typical for "focused" explanations
   - Directly measures spatial overlap

### Supporting Metrics

4. **Pointing Game**: Simple sanity check
   - Easy to interpret
   - Quick assessment of basic functionality

5. **AUC-ROC**: Complementary to Average Precision
   - Threshold-agnostic
   - Well-known in ML community

6. **IoU@{5,15,20,25}%**: Robustness analysis
   - Shows performance across different focus levels
   - Identifies methods that are too diffuse or too focused

---

## Visualization Best Practices

All visualizations follow these principles:

1. **Color-blind friendly**: Okabe-Ito palette (8 distinct colors)
2. **Publication quality**: 150-300 DPI, vector-friendly
3. **Consistent styling**: Same fonts, sizes, grids across all figures
4. **Clear legends**: Method names, metric values, significance indicators
5. **Contextual baselines**: Random baseline shown where applicable
6. **Grid layout**: Multiple views in single figure for comparison

### Colormaps

- **Sequential** (Inferno): Gradient-based methods (IG, Input Gradients, Saliency, Occlusion, Gradient SHAP)
  - Always positive values (absolute value taken)
  - Dark = low attribution, Bright = high attribution
  - Range: [0, max]

- **Divergent** (RdBu_r): Perturbation-based methods (LIME, Kernel SHAP)
  - Can have negative values (suppression vs enhancement)
  - **Blue** = negative attribution (removing feature INCREASES prediction)
  - **White** = zero attribution (neutral)
  - **Red** = positive attribution (removing feature DECREASES prediction)
  - Range: [-max, +max] (symmetric around 0)
  - **Important**: Sign is preserved (NOT taking absolute value) to distinguish suppressive from enhancing features

**Why distinguish positive/negative for LIME and SHAP?**

In perturbation-based methods:
- **Positive attribution** (red): This superpixel/region SUPPORTS the prediction. Removing it would decrease the class probability.
- **Negative attribution** (blue): This superpixel/region CONTRADICTS the prediction. Removing it would actually INCREASE the class probability (it's evidence against this class).

Example: For a "Cardinal" (red bird) prediction:
- Red regions: Actual red feathers (support "Cardinal")
- Blue regions: Green background or non-cardinal features (contradict "Cardinal")

Gradient-based methods typically don't have meaningful negative values after abs() aggregation, so sequential colormap is appropriate.

---


## File Locations

### Metrics Implementation
- `src/evaluation.py`: All metric computation functions
  - `compute_pointing_game()`
  - `compute_energy_based_pointing_game()`
  - `compute_auc_roc()`
  - `compute_average_precision()`
  - `compute_iou_at_threshold()`
  - `compute_all_metrics()` (wrapper)

### Visualization Implementation
- `src/visualizations_utils.py`: All visualization functions (13 functions)
  - Single-image: `visualize_all_methods()`, `visualize_attribution_enhanced()`
  - Curves: `create_roc_curve_comparison()`, `create_pr_curve_comparison()`, `create_iou_threshold_curves()`
  - Detailed: `create_pointing_game_visualization()`, `create_ebpg_visualization()`
  - Aggregation: `create_methods_ranking_heatmap()`, `create_metrics_radar_chart()`
  - Analysis: `create_model_vs_method_analysis()`, `create_sanity_check_comparison()`

### Configuration
- `src/config.py`: Metric settings
  - `TOP_K_PERCENT = [5, 10, 15, 20, 25]`
  - `EXPLAINABILITY_METHODS` - list of all methods

### Results
- `results/`: All saved visualizations (PNG format, 150+ DPI)
  - Individual plots: `roc_curves.png`, `pr_curves.png`, etc.
  - Summary: `metrics_comparison.png`, `methods_ranking_heatmap.png`
  - Sanity checks: `sanity_check_comparison.png`, `cascade_randomization_test.png` (NEW!)
  - CSV exports: `summary_results.csv`, `detailed_results.csv`

---

## Interpretation Guidelines

### What makes a "good" explanation?

1. **High Plausibility** (EBPG > 0.5, IoU@10% > 0.3)
   - Focuses on bird parts, not background
   - Aligns with human intuition

2. **High Ranking Quality** (AP > 0.7, AUC > 0.8)
   - Correctly ranks bird pixels above background
   - Shows the method discriminates relevant features

3. **Passes Sanity Check** (correlation < 0.3 with random weights)
   - Changes significantly when model is randomized
   - Not just an edge detector or input gradient

4. **Faithful to Model** (better scores for correct predictions)
   - Explanations improve when model is correct
   - Suggests method captures model reasoning

### Red Flags

- **Low EBPG** (< 0.3): Most attribution is on background → Poor plausibility
- **AP ≈ Random Baseline**: Method doesn't rank pixels better than chance
- **High Sanity Check Correlation** (> 0.6): Method likely artifact-driven
- **No difference between correct/incorrect**: Method may be model-agnostic in a bad way

---

## Recommended Reporting

### For Publications

**Table 1**: Mean ± Std across all images
- Columns: EBPG, AP, AUC-ROC, IoU@10%
- Rows: Methods
- Bold: Best per column

**Figure 1**: Qualitative Examples (3-5 images)
- `visualize_all_methods()` for each image
- Show diverse cases (correct, incorrect, different species)

**Figure 2**: Quantitative Comparison
- ROC curves + PR curves (side by side)
- Methods ranking heatmap

**Figure 3**: Sanity Check
- `create_sanity_check_comparison()` for 2-3 methods
- Show correlation values

### For Reports/Documentation

- All metrics table (9+ metrics)
- Box plots for distribution analysis
- Radar chart for multi-metric view
- Model vs Method analysis (faithfulness)
- Full sanity check results (all methods)

---

## Future Extensions

### Potential Additional Metrics

1. **Insertion/Deletion** (Fong & Vedaldi, 2017)
   - Measures faithfulness by perturbing image
   - Requires many model evaluations (slow)

2. **Segmentation Metrics** (Dice, F1)
   - Alternative to IoU
   - Better for very imbalanced regions

3. **Human Evaluation**
   - Ground truth = what humans find important
   - Gold standard but expensive

### Potential Additional Visualizations

1. **Attribution Evolution**
   - How attributions change across layers
   - Requires layer-wise methods (GradCAM, LRP)

2. **Failure Analysis**
   - Deep dive into low-scoring images
   - Identify systematic failure modes

3. **Cross-Method Agreement**
   - Correlation matrix between methods
   - Identify redundant/complementary methods

---

## Conclusion

This framework provides:
- ✅ **4 core plausibility metrics** (PG, EBPG, AUC, AP)
- ✅ **5 IoU variants** for robustness (5%, 10%, 15%, 20%, 25%)
- ✅ **11 visualization functions** following XAI literature
- ✅ **3 analysis types** (single-image, aggregation, model-vs-method)
- ✅ **Sanity checks** for reliability (Adebayo et al., 2018)

**Primary metrics for reporting**: EBPG (plausibility), AP (ranking), IoU@10% (localization)

All metrics and visualizations are publication-ready and follow established best practices from the XAI literature.
