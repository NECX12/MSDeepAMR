# MSDeepAMR: ONNX Migration & PyTorch Fine-Tuning for Antimicrobial Resistance Prediction

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![ONNX](https://img.shields.io/badge/ONNX-opset_13-green.svg)](https://onnx.ai/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

An end-to-end framework for migrating pre-trained **MSDeepAMR** 1D convolutional neural network (1D-CNN) models from **TensorFlow/Keras** into native **PyTorch** via **ONNX**, enabling cross-framework interoperability and systematic **transfer learning / fine-tuning** on clinical MALDI-TOF mass spectrometry data.

---

## 📌 Background & Context

This project builds upon the foundational research:
> **"MSDeepAMR: antimicrobial resistance prediction based on deep neural networks and transfer learning"**  
> *Frontiers in Microbiology (2024)*.  
> 🔗 **Article**: [10.3389/fmicb.2024.1361795](https://www.frontiersin.org/journals/microbiology/articles/10.3389/fmicb.2024.1361795/full)

### What is MSDeepAMR?
MSDeepAMR is a deep learning methodology designed for direct analysis of raw **Matrix-Assisted Laser Desorption/Ionization Time-of-Flight (MALDI-TOF)** mass spectra to rapidly identify antimicrobial resistance (AMR). Conventional antimicrobial susceptibility testing (AST) requires 24–72 hours of microbial culturing. MSDeepAMR enables rapid, low-cost resistance screening directly from raw spectra in minutes.

The original models target three critical nosocomial pathogens and antibiotic pairs:
- **_Escherichia coli_** — Ceftriaxone (3rd-generation cephalosporin)
- **_Klebsiella pneumoniae_** — Ceftriaxone (high clinical concern)
- **_Staphylococcus aureus_** — Oxacillin (Methicillin resistance / MRSA screening)

### Original Model Architecture
The architecture comprises a 4-block 1D-CNN (Conv1D + BatchNormalization + ReLU + MaxPooling1D) followed by a flattening layer and dense classification head (Dense 256 → Dense 128 → Dense 1 + Sigmoid) totaling **566,336 parameters** per model:

![MSDeepAMR Architecture](https://user-images.githubusercontent.com/43461313/201794393-78c50952-ff42-425b-b5ee-c275481e835d.png)

---

## 🎯 Repository Focus: PyTorch Fine-Tuning via ONNX

While the original models were distributed as legacy Keras `.h5` files trained in TensorFlow 2.3.0 / Keras 2.4.3, this repository focuses on **migrating the pre-trained weights into PyTorch and systematically fine-tuning them** on target clinical isolates using the workflow implemented in [`MSDeepAMR_finetuning_pytorch.ipynb`](MSDeepAMR_finetuning_pytorch.ipynb).

### Why ONNX & `onnx2torch`?
Directly retraining deep neural networks on small clinical datasets leads to severe overfitting. Transfer learning and fine-tuning allow us to reuse the spectral feature extraction learned by the original models. However:
- Traditional **ONNX Runtime** is **inference-only**: it lacks an autograd computation graph, preventing backpropagation and parameter updates.
- **`onnx2torch` (ENOT-AutoDL)** parses the ONNX intermediate representation (opset 13) and reconstructs the computation graph into authentic `torch.nn.Module` objects with live `torch.nn.Parameter` tensors.
- This unlocks PyTorch's native autograd engine, custom loss functions, optimizers, learning rate schedulers, and layer-freezing mechanics.

---

## 🔄 End-to-End Workflow Pipeline

```
┌────────────────────────┐      ┌────────────────────────┐      ┌────────────────────────┐
│  Original Keras Model  │ ──►  │    ONNX Intermediate   │ ──►  │  PyTorch nn.Module     │
│       (.h5 file)       │      │   Graph (Opset 13)     │      │  (Trainable Parameters)│
└────────────────────────┘      └────────────────────────┘      └───────────┬────────────┘
         tf2onnx CLI                     onnx.checker                       │
                                                                            ▼
                                                      ┌─────────────────────────────────────────┐
                                                      │        Fine-Tuning Strategies           │
                                                      ├─────────────────────────────────────────┤
                                                      │ Strategy A: Zero-shot Baseline (Frozen) │
                                                      │ Strategy B: Full Fine-Tune (LR = 1e-5)  │
                                                      │ Strategy C: Partial Unfreeze (LR = 5e-6)│
                                                      └─────────────────────────────────────────┘
```

1. **Model Export**: Load `.h5` files, attach explicit serving signatures `(None, 6000, 1)`, and export to ONNX opset 13 via `tf2onnx`.
2. **Graph Validation**: Structural integrity checked with `onnx.checker` (all 3 models confirmed with 49 valid nodes).
3. **Graph Reconstruction**: Converted into native PyTorch modules via `onnx2torch.convert()`, preserving all 566,336 parameters.
4. **Data Adaptation**: Raw 6,000 m/z spectra formatted into `SpectraDataset` (handling channels-last convention `(N, 6000, 1)` expected by the converted operators).
5. **Fine-Tuning & Evaluation**: Multi-strategy training with class-imbalance mitigation, early stopping, and threshold calibration.

---

## 🔬 Fine-Tuning Strategies

The notebook implements three comparative strategies to evaluate how pre-trained spectral representations adapt to downstream clinical splits:

| Strategy | Description | Trainable Layers | Learning Rate | Purpose |
|---|---|---|---|---|
| **Strategy A: Zero-Shot Baseline** | Model evaluated as-is without further training | None (all layers frozen) | — | Benchmark transferability of pre-trained weights without adaptation |
| **Strategy B: Full Fine-Tune** | All layers unfrozen and trained end-to-end | All 4 Conv blocks + Dense head | $1 \times 10^{-5}$ | Adapt entire feature representation to dataset characteristics |
| **Strategy C: Partial Unfreeze** | Conv blocks 1 & 2 frozen; Conv blocks 3 & 4 and Dense head trainable | Conv blocks 3 & 4 + Dense head | $5 \times 10^{-6}$ | Preserve low-level spectral peak extractors while specializing high-level features |

---

## ⚙️ Key Technical Implementations

The PyTorch pipeline in [`MSDeepAMR_finetuning_pytorch.ipynb`](MSDeepAMR_finetuning_pytorch.ipynb) introduces critical engineering solutions tailored to clinical mass spectrometry:

1. **Class Imbalance Handling with `WeightedRandomSampler`**:
   - Clinical AMR datasets suffer from severe class imbalance: resistant cases typically represent only **6.5% – 10.4%** of samples.
   - Standard mini-batch sampling causes models to collapse into predicting 100% susceptible, producing 0% recall on the resistant class.
   - A `WeightedRandomSampler` dynamically oversamples minority (resistant) isolates during training, balancing gradient updates across classes.

2. **Overfitting Prevention**:
   - **EarlyStopping**: Monitors validation balanced accuracy / loss with a patience of 5 epochs and automatically restores the best model weights.
   - **ReduceLROnPlateau**: Dynamically decays the learning rate by $0.1\times$ (patience = 3 epochs, minimum LR = $1 \times 10^{-7}$).

3. **Post-Hoc Threshold Calibration (Youden's J Statistic)**:
   - Default classification threshold ($\tau = 0.5$) is frequently suboptimal for imbalanced clinical screening.
   - The pipeline computes the optimal decision threshold $\tau^*$ on validation probabilities maximizing Youden's Index ($J = \text{Sensitivity} + \text{Specificity} - 1$), substantially recovering clinical sensitivity.

4. **Rigorous Evaluation Suite**:
   - Computes standard Accuracy, **Balanced Accuracy** (unbiased average of sensitivity and specificity), Precision, Recall, AUROC, AUPRC, and confusion matrices using `scikit-learn`.

---

## 📊 Dataset & Evaluated Splits

Experiments use the **DRIAMS C2018** MALDI-TOF dataset (Dryad DOI: [10.5061/dryad.bzkh1899q](https://doi.org/10.5061/dryad.bzkh1899q)):
- **Spectra Data**: `data/C2018/rawSpectra_data.npy` (9,658 samples $\times$ 6,000 intensity bins).
- **Labels & Metadata**: `data/C2018/combined_long_table.csv` and predefined split indexes in `data/C2018/data_splits.csv`.

| Pathogen | Antibiotic | Train Isolates (Resistant %) | Val Isolates (Resistant %) | Test Isolates (Resistant %) |
|---|---|---|---|---|
| **_Escherichia coli_** | Ceftriaxone | 737 (16.3%) | 92 (20.7%) | 780 (10.4%) |
| **_Klebsiella pneumoniae_** | Ceftriaxone | 297 (15.2%) | 40 (15.0%) | 526 (6.5%) |
| **_Staphylococcus aureus_** | Oxacillin | — | — | 450 (~8.2%) |

---

## 📁 Repository Structure

```text
MSDeepAMR_finetuned/
├── data/
│   └── C2018/
│       ├── rawSpectra_data.npy                 # 9,658 MALDI-TOF raw spectra (6,000 bins)
│       ├── combined_long_table.csv             # Metadata, species, drug, and AMR labels
│       └── data_splits.csv                     # Predefined train/val/test split indices
├── e_coli_ceftriaxone.h5                       # Pre-trained Keras model (E. coli + Ceftriaxone)
├── k_pneumoniae_ceftriaxone.h5                 # Pre-trained Keras model (K. pneumoniae + Ceftriaxone)
├── s_aureus_oxacillin.h5                       # Pre-trained Keras model (S. aureus + Oxacillin)
├── MSDeepAMR_finetuning_pytorch.ipynb          # ★ Main PyTorch ONNX conversion & fine-tuning notebook
├── ONNX_migration_of_tf_model_and_finetuning_of_MSDeepAMR_Model.ipynb # Experimental migration notebook
├── MSDeepAMR_CS197_Focus_Group_Review.pptx     # Comprehensive 18-slide presentation
├── generate_pptx.py                            # Presentation generation script
└── README.md                                   # Project documentation
```

---

## 🚀 Quickstart & Usage

### 1. Environment Setup

Create a virtual environment and install the required dependencies:

```bash
git clone https://github.com/your-username/MSDeepAMR_finetuned.git
cd MSDeepAMR_finetuned

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # or cpu
pip install tf2onnx==1.16.1 onnx2torch==1.5.15 onnx==1.16.2 onnxruntime==1.19.2
pip install scikit-learn matplotlib pandas numpy
```

### 2. Running the Fine-Tuning Pipeline

Open and run [`MSDeepAMR_finetuning_pytorch.ipynb`](MSDeepAMR_finetuning_pytorch.ipynb) in Jupyter Lab, VS Code, or Google Colab:
- **Part 1**: Converts `.h5` models to `.onnx` and checks validity.
- **Part 2**: Reconstructs models as PyTorch `nn.Module`s.
- **Part 3–5**: Configures data loaders, custom `EarlyStopping`, and training loops.
- **Part 6**: Executes fine-tuning across all three pathogen–drug pairs for Strategies A, B, and C.
- **Part 7**: Generates comparative performance summaries, confusion matrices, and calibrated threshold analyses.

### 3. Programmatic Usage in PyTorch

```python
import onnx
import torch
from onnx2torch import convert as onnx2torch_convert

# 1. Load ONNX model and convert to PyTorch nn.Module
onnx_model = onnx.load("e_coli_ceftriaxone.onnx")
pytorch_model = onnx2torch_convert(onnx_model)
pytorch_model.eval()

# 2. Input tensor: (Batch, 6000, 1) float32 spectra
sample_spectra = torch.randn(10, 6000, 1)

# 3. Forward pass
with torch.no_grad():
    resistance_probabilities = pytorch_model(sample_spectra)

print("Predicted probabilities:", resistance_probabilities.squeeze())
```

---

## 👥 Contributors & Acknowledgements

Developed as part of the **ML Community CS197 Focus Group Review**:
- **Nwakpa Ejike Chukwuma**
- **Daniel Bitrus**
- **Elegbede Tobi**
- **Abioye Delight**

Special acknowledgment to the original authors of **MSDeepAMR** and the **DRIAMS** research consortium for making the models and mass spectrometry datasets openly accessible to the scientific community.

---

## 📜 References

1. **MSDeepAMR Paper**:  
   Ferruz-Agudo, C. et al. *MSDeepAMR: antimicrobial resistance prediction based on deep neural networks and transfer learning*. Frontiers in Microbiology, 15:1361795 (2024).  
   DOI: [10.3389/fmicb.2024.1361795](https://doi.org/10.3389/fmicb.2024.1361795)

2. **DRIAMS Dataset**:  
   Weis, C. et al. *Direct antimicrobial resistance prediction from clinical MALDI-TOF mass spectra using machine learning*. Dryad Data Repository.  
   DOI: [10.5061/dryad.bzkh1899q](https://doi.org/10.5061/dryad.bzkh1899q)
