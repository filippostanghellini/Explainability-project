[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Captum](https://img.shields.io/badge/Captum-XAI-orange?style=for-the-badge&logo=pytorch&logoColor=white)](https://captum.ai/)
<a target="_blank" href="https://lightning.ai/">
  <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg" alt="Open in Studio" />
</a>

# Explainability Methods Comparison on CUB-200-2011

This project compares different explainability approaches from the **Captum** library for a CNN classifier trained on the **CUB-200-2011** bird species dataset. The focus is on measuring **explanation plausibility** by comparing model explanations with ground-truth part annotations.

## Project Goals

1. **Train a CNN classifier** on CUB-200-2011 (200 bird species)
2. **Apply multiple explainability methods**: Integrated Gradients, Saliency, Input×Gradient, LIME, Kernel SHAP, Occlusion, Noise Tunnel (wrapper)
3. **Evaluate explanation quality** (plausibility) using ground-truth part annotations
4. **Analyze**: Are the results due to the model or to the explanation method?

## Project Structure

```
Explainability-project/
├── src/                       # Python package
│   ├── __init__.py
│   ├── config.py              # Configuration and hyperparameters
│   ├── data_loader.py         # Dataset loading and preprocessing
│   ├── model.py               # CNN model definition (ResNet-50)
│   ├── explainability.py      # Explainability methods (Captum)
│   ├── evaluation.py          # Plausibility metrics
│   └── visualizations_utils.py # Visualization utilities
├── train.py                   # Training script
├── model_evaluation.py        # Model evaluation script
├── explainability_analysis.ipynb  # Interactive Jupyter notebook
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── models/                    # Saved model checkpoints
├── results/                   # Evaluation results and metrics
├── visualizations/            # Generated visualizations
├──CUB_200_2011/
  └── CUB_200_2011/
      └── ...

```

## Quick Start

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

### 4. Download Pre-trained Model

The trained ResNet-50 model trained by us is available via **Git LFS** (Large File Storage). To download it:

```bash
# Install Git LFS if not already installed
# macOS
brew install git-lfs

# Ubuntu/Debian
sudo apt-get install git-lfs

# Windows (with Git for Windows)
# Git LFS is included by default

# Initialize Git LFS
git lfs install

# Pull the model file
git lfs pull
```

The model will be downloaded to `models/best_resnet50_cub200.pth`.

> **Note**: If you cloned the repository without Git LFS installed, the model file will be a pointer file. Run `git lfs pull` after installing Git LFS to download the actual model.

### 5. Train the Model (Optional)

If you want to train the model from scratch instead of using the pre-trained one:

```bash
python train.py --epochs 30 --batch_size 32 --lr 0.001
```

Options:

Check 'train.py'

### 6. Run Model Evaluation

```bash
python model_evaluation.py
```

This script evaluates the trained model on the test set and generates:
- Classification metrics (accuracy, precision, recall, F1-score)
- Confusion matrix
- Per-class performance analysis

Results are saved in the `results/` directory.

### 7. Interactive Analysis (MAIN NOTEBOOK)

Open the Jupyter notebook for interactive exploration:

```bash
jupyter notebook explainability_analysis.ipynb
```

## Technical Details

### Hardware
The training and intensive explainability computations were performed using:
* **GPU**: NVIDIA L4 
* **Cloud Platform**: [Lightning.ai](https://lightning.ai/)

## References

- **Captum**: https://captum.ai - PyTorch model interpretability library
- **CUB-200-2011**: https://www.vision.caltech.edu/datasets/cub_200_2011 - Fine-grained bird classification dataset

### Papers:

[1] J. Adebayo, J. Gilmer, M. Muelly, I. Goodfellow, M. Hardt, and B. Kim, "Sanity Checks for Saliency Maps," in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2018.

[2] S. M. Lundberg and S.-I. Lee, "A Unified Approach to Interpreting Model Predictions," in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2017.

[3] M. T. Ribeiro, S. Singh, and C. Guestrin, "'Why Should I Trust You?': Explaining the Predictions of Any Classifier," in Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discovery and Data Mining (KDD), 2016.

[4] K. Simonyan, A. Vedaldi, and A. Zisserman, "Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps," in Proc. ICLR Workshop, 2014.

[5] M. Sundararajan, A. Taly, and Q. Yan, "Axiomatic Attribution for Deep Networks," in Proc. 34th Int. Conf. Machine Learning (ICML), 2017.

[6] C. Wah, S. Branson, P. Welinder, P. Perona, and S. Belongie, "The Caltech-UCSD Birds-200-2011 Dataset," California Institute of Technology, Tech. Rep. CNS-TR-2011-001, 2011.

[7] J. Wei et al., "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models," in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2022.

## Authors

This project was developed by:

- [@filippostanghellini](https://github.com/filippostanghellini)
- [@samueleviola](https://github.com/samueleviola)
