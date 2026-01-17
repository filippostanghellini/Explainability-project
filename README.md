## Explainable AI Benchmark on CUB200-2011

## 📂 Structure

```
project/
├── CUB_200_2011/          # dataset
├── src/
│   ├── data_utils.py      # CUB200 dataset management and metadata parsing
│   ├── model.py           # Architecture definition (ResNet50)
│   ├── explainers.py      # Captum wrappers (IG, LIME, SHAP...)
│   └── train.py           # Training and validation loop
├── exploratory.ipynb      # Notebook for preliminary experiments
└── README.md
```

---

## 🛠 Implemented Modules

### 1. Model (`src/model.py`)

Uses **Transfer Learning** starting from a **ResNet50** pre-trained on **ImageNet**.

- **Modification**: the final *Fully Connected* layer is replaced to adapt to the **200 bird classes** in the CUB dataset.
- **Features**: supports **saving/loading weights** and **feature extraction**.

---

### 2. Data (`src/data_utils.py`)

A custom `CUBDataset` class that manages the complexity of the **CUB200** text metadata files:

- Cross-references `images.txt` and `image_class_labels.txt` to associate **images and labels**.
- Handles the required **transformations** (resize, normalization) for ResNet.

> **Note**: includes a preliminary function `get_part_annotations` to read **anatomical part coordinates** (crucial for the evaluation phase).

---

### 3. Explainers (`src/explainers.py`)

An object-oriented architecture based on **Captum** that standardizes the interface across different explanation methods:

- **Gradient-based (White-box)**: Integrated Gradients, Saliency (Input Gradients).
- **Perturbation-based (Black-box)**: LIME, KernelSHAP.

**Output**: each explainer returns a **normalized heatmap**, ready for visualization or quantitative evaluation.

---

### 4. Training (`src/train.py`)

A fast **fine-tuning** pipeline:

- Uses `CrossEntropyLoss` and the **Adam** optimizer.
- Automatically saves the **model with the best validation accuracy**.

---

## 🚀 How to Run

### Prerequisites

```bash
pip install -r requirements.txt
```

Run the exploratory notebook to test the modules.

---

## 📊 Project Status and Next Steps (TODO)

At the moment, the project is able to:

- [x] Correctly load the dataset and labels
- [x] Train the model with good performance
- [x] Generate visual explanations (heatmaps) using **4 different algorithms**

### Gap Analysis – Missing Requirements for the Exam

The core course requirement is **quantitative evaluation of plausibility**.

- [ ] **Data Engineering**: update `CUBDataset` to return **Ground Truth Masks** (binary masks created from anatomical part coordinates)
- [ ] **Metric**: implement an **Intersection over Union (IoU)** or **Energy Fraction** metric to measure how much the heatmap overlaps with the true part masks
- [ ] **Benchmark**: run a script over the entire test set to obtain final scores (e.g., *"IG achieves 60% plausibility vs LIME 45%"*)
