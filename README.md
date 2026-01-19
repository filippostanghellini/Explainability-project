# Explainability Methods Comparison on CUB-200-2011

This project compares different explainability approaches from the **Captum** library for a CNN classifier trained on the **CUB-200-2011** bird species dataset. The focus is on measuring **explanation plausibility** by comparing model explanations with ground-truth part annotations.

## 🎯 Project Goals

1. **Train a CNN classifier** on CUB-200-2011 (200 bird species)
2. **Apply multiple explainability methods**: LIME, SHAP, Integrated Gradients, Input Gradients
3. **Evaluate explanation quality** (plausibility) using ground-truth part annotations
4. **Analyze**: Are the results due to the model or to the explanation method?

## 📁 Project Structure

```
Explainability-project/
├── src/                       # Python package
│   ├── __init__.py
│   ├── config.py              # Configuration and hyperparameters
│   ├── data_loader.py         # Dataset loading and preprocessing
│   ├── model.py               # CNN model definition (ResNet-50)
│   ├── explainability.py      # Explainability methods (Captum wrapper)
│   ├── evaluation.py          # Plausibility metrics
│   └── visualizations_utils.py # Visualization utilities
├── train.py                   # Training script
├── run_evaluation.py          # Main evaluation script
├── explainability_analysis.ipynb  # Interactive Jupyter notebook
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── models/                    # Saved model checkpoints
├── results/                   # Evaluation results and metrics
├── visualizations/            # Generated visualizations
└── CUB_200_2011/             # Dataset directory
```

## 🚀 Quick Start

### 1. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare the Dataset

The CUB-200-2011 dataset should be in the `CUB_200_2011/` directory with the following structure:
- `images/` - Bird images organized by species
- `parts/` - Part location annotations
- `attributes/` - Attribute annotations
- Various metadata files (images.txt, classes.txt, etc.)

Download from: https://www.vision.caltech.edu/datasets/cub_200_2011

### 4. Train the Model

```bash
python train.py --epochs 30 --batch_size 32 --lr 0.001
```

Options:
- `--epochs`: Number of training epochs (default: 30)
- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 0.001)
- `--model`: Model type - 'resnet50' or 'vgg16' (default: resnet50)
- `--resume`: Resume from checkpoint

### 5. Run Evaluation

```bash
python run_evaluation.py --num_samples 100 --model_path models/best_resnet50_cub200.pth
```

Options:
- `--num_samples`: Number of test images to evaluate (default: 100)
- `--model_path`: Path to trained model
- `--methods`: Specific methods to use (e.g., `--methods integrated_gradients lime`)

### 6. Interactive Analysis

Open the Jupyter notebook for interactive exploration:

```bash
jupyter notebook explainability_analysis.ipynb
```

## 📊 Explainability Methods

| Method | Type | Description |
|--------|------|-------------|
| **Integrated Gradients** | Gradient-based | Accumulates gradients along a path from baseline to input |
| **Input × Gradient** | Gradient-based | Element-wise product of input and its gradient |
| **Saliency** | Gradient-based | Simple gradient of output w.r.t. input |
| **LIME** | Perturbation-based | Local linear approximation using superpixel perturbations |
| **Kernel SHAP** | Perturbation-based | Shapley values via weighted linear regression |
| **Occlusion** | Perturbation-based | Systematically occludes image regions |

## 📈 Evaluation Metrics

We measure **plausibility** - how well explanations align with human-interpretable features (bird parts):

| Metric | Description |
|--------|-------------|
| **Pointing Game** | Does the maximum attribution fall within a ground-truth part? |
| **EBPG** (Energy-Based Pointing Game) | Proportion of attribution energy within GT parts |
| **AUC-ROC** | Area under ROC curve treating attributions as predictions |
| **Average Precision** | Average precision for detecting GT regions |
| **Mass Accuracy** | Fraction of top-k% attributions in GT regions |
| **IoU** | Intersection over Union of thresholded attributions |
| **Spearman Correlation** | Rank correlation between attributions and GT mask |

## 🔬 Key Analysis: Model vs Explanation Method

The project addresses a fundamental question: **Are the observed results due to the model's reasoning or artifacts of the explanation method?**

We analyze this by comparing explanation quality for:
- **Correct predictions**: Model correctly identified the bird species
- **Incorrect predictions**: Model misclassified the bird

**Interpretation**:
- If explanations are **better for correct predictions** → The explanation method is likely **faithful** (captures what the model actually uses)
- If there's **no difference** → The explanation might not be capturing the model's true reasoning

## 📋 Ground Truth: Part Annotations

CUB-200-2011 provides 15 part annotations per image:
1. Back
2. Beak
3. Belly
4. Breast
5. Crown
6. Forehead
7. Left eye
8. Left leg
9. Left wing
10. Nape
11. Right eye
12. Right leg
13. Right wing
14. Tail
15. Throat

We create binary masks around visible parts (configurable radius) to serve as ground truth for plausibility evaluation.

## 📊 Example Results

After running the evaluation, you'll find:

- **Summary metrics** for each method
- **Statistical comparisons** between methods (Wilcoxon signed-rank test)
- **Visualizations**:
  - Attribution heatmaps overlaid on images
  - Metrics comparison bar charts
  - Distribution box plots
  - Correct vs incorrect prediction analysis

## 🔗 References

- **Captum**: https://captum.ai - PyTorch model interpretability library
- **CUB-200-2011**: https://www.vision.caltech.edu/datasets/cub_200_2011 - Fine-grained bird classification dataset

### Papers:
- Sundararajan et al. (2017) - Axiomatic Attribution for Deep Networks (Integrated Gradients)
- Ribeiro et al. (2016) - LIME: Local Interpretable Model-agnostic Explanations
- Lundberg & Lee (2017) - SHAP: A Unified Approach to Interpreting Model Predictions
- Simonyan et al. (2014) - Deep Inside Convolutional Networks (Saliency Maps)

## 📝 Notes

1. **Plausibility vs Faithfulness**: High plausibility doesn't guarantee the explanation is faithful to the model's actual reasoning. Parts are just one aspect of what defines bird species.

2. **Computational Cost**: LIME and SHAP are slower than gradient-based methods due to perturbation sampling.

3. **Hyperparameters**: Results may vary with different settings (e.g., number of LIME samples, IG steps).
