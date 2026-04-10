#  Non-Invasive Anemia Detection from Palpebral Conjunctiva Images
### Using a Hybrid CNN-ViT Architecture Optimized with Grey Wolf Optimization (GWO)



---

##  Overview

This repository implements a **non-invasive, deep-learning-based anemia screening system** that analyzes images of the **palpebral conjunctiva** (inner eyelid) — a clinically established indicator of hemoglobin levels. Instead of a blood draw, the model classifies a patient as **Anemic** or **Non-Anemic** from a photograph.

The core contribution is a **Hybrid CNN-ViT architecture** — combining a CNN backbone (ResNet-50) for extracting local microvascular features with a Vision Transformer (ViT) encoder for capturing global pallor patterns — further enhanced by **Grey Wolf Optimization (GWO)** for automatic hyperparameter tuning.

---



##  Dataset

**EYES-DEFY-ANEMIA** (Kaggle)

| Class | Samples |
|---|---|
| Non-Anemic (Label 0) | 442 |
| Anemic (Label 1) | 358 |
| **Total** | **800** |

The dataset CSV must contain three columns: `image_path`, `mask_path`, and `label`.

> **Kaggle path:** `/kaggle/input/eyes-defy-anemia`

A **synthetic fallback mode** is built in for offline development — it generates realistic conjunctiva-like RGB images (reddish-pink with dense vascular streaks for non-anemic; pale pink with sparse streaks for anemic) so all cells can be run end-to-end without the real dataset.

---

##  Pipeline

```
Input Image [B, 3, 224, 224]
      │
      ▼
CNN Backbone (ResNet-50)         ← local microvascular features
      │  Feature Maps F ∈ R^{H×W×C}
      ▼
Linear Projection + [CLS] + Positional Embedding   ← patch sequence z₀
      │
      ▼
Transformer Encoder (L blocks)  ← MSA captures global pallor patterns
      │  CLS token output
      ▼
MLP Classification Head  →  {Anemic, Non-Anemic}
```



## 🐺 Grey Wolf Optimization (GWO)

GWO mimics the social hunting hierarchy of grey wolves (Alpha > Beta > Delta > Omega) to search for optimal hyperparameters without manual grid search.

**Search Space Ω:**

| Hyperparameter | Range |
|---|---|
| Learning rate η | [10⁻⁵, 10⁻²] (log scale) |
| Dropout rate d | [0.1, 0.5] |
| Attention heads K | {2, 4, 8} |
| Transformer depth L | {2, 4, 6, 8} |



## ⚙️ Configuration

Key flags in **Cell 3** (`Experiment Configuration`):

```python
USE_REAL_DATASET    = False   # Set True for real EYES-DEFY-ANEMIA data
DATASET_PATH        = "/kaggle/input/eyes-defy-anemia"
USE_PRETRAINED_RESNET = True  # ResNet-50 with ImageNet weights
RUN_GWO             = False   # Set True to run hyperparameter search
EMBED_DIM           = 256     # Transformer embedding dimension
EPOCHS              = 30
BATCH_SIZE          = 32
WEIGHT_DECAY        = 1e-4
```

---

## 🚀 Getting Started

### Kaggle (Recommended)

1. Open `anemia_cnn_vit_gwo_kaggle.ipynb` in a Kaggle kernel.
2. Attach the **EYES-DEFY-ANEMIA** dataset.
3. Set `USE_REAL_DATASET = True` and verify `DATASET_PATH`.
4. Enable GPU accelerator (P100 / T4).
5. Run all cells in order.

### Local

```bash
# Clone the repo
git clone <your-repo-url>
cd <repo-name>

# Install dependencies
pip install torch torchvision timm torchcam scikit-learn \
            matplotlib seaborn pillow pandas numpy scipy

# Launch Jupyter
jupyter notebook eyes-anemia.ipynb
```

> Ensure your conda/virtual environment uses Python 3.9+.

---

## 📊 Results

The GWO-optimized Hybrid CNN-ViT is compared against standard baselines trained on the same train/val/test splits:

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|---|---|---|---|---|---|
| **GWO CNN-ViT (Ours)** | — | — | — | — | — |
| ResNet-50 | — | — | — | — | — |
| VGG-16 | — | — | — | — | — |
| DenseNet-121 | — | — | — | — | — |
| EfficientNet-B0 | — | — | — | — | — |

<img width="1525" height="595" alt="image" src="https://github.com/user-attachments/assets/48e30b54-1c2e-48a8-a5c2-50f7ebe044f9" />


<img width="858" height="566" alt="image" src="https://github.com/user-attachments/assets/02985531-c22d-4a1f-bfd1-5566c26f9bfd" />


## 💾 Model Checkpoint

After training, the best model is saved to:

```
outputs/gwo_cnn_vit_anemia_best.pth
```

The checkpoint contains:

```python
{
    "model_state_dict": ...,
    "hyperparams":      BEST_HP,   # GWO-optimal hyperparameters
    "metrics":          metrics,   # Final evaluation metrics
    "embed_dim":        EMBED_DIM,
}
```

-
---

## 📚 References

1. Dosovitskiy, A. et al. (2021). *An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale.* ICLR.
2. Mirjalili, S. (2014). *Grey Wolf Optimizer.* Advances in Engineering Software, 69, 46–61.
3. WHO (2023). *Global Anaemia Estimates.* WHO Press.
4. Bano, S. et al. (2024). *Digital imaging of the palpebral conjunctiva: A systematic review.* Medical Image Analysis, 89.
5. Sehar, N. et al. (2025). *Deep learning model-based detection of anemia from conjunctiva images.* Healthcare Informatics Research, 31(1).

---


---

*Implementation of: "Non-Invasive Anemia Detection from Palpebral Conjunctiva Images Using Vision Transformers Optimized with Grey Wolf Optimization"*
