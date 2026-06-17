[![Python](https://img.shields.io/badge/Python-3.x-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-orange)](https://pytorch.org/)

# Deep Learning with PyTorch — ANN, CNN & RNN Projects

A collection of four deep learning models built from scratch in PyTorch, covering the three core architectures: feedforward networks (ANN/FNN) for classification and regression, a convolutional network (CNN) for image classification, and a recurrent network (RNN) for text sentiment analysis. Each project is a self-contained Jupyter notebook covering data preprocessing, model definition, training, and evaluation.

## Repository Structure

```
.
├── ANN/
│   ├── ANN_classification.ipynb   # Feedforward NN — multi-class classification (Date Fruit dataset)
│   ├── ANN_regression.ipynb       # Feedforward NN — regression (Power Plant energy output)
│   ├── DateFruit_Dataset.csv
│   └── powerplant_data.csv
├── CNN/
│   └── CNN_for_CIFAR10.ipynb      # Convolutional NN — image classification (CIFAR-10)
├── RNN/
│   └── RNN_sentiment-analysis.ipynb  # Recurrent NN — sentiment classification (IMDB reviews)
└── README.md
```
IMDB Dataset of 50K Movie Reviews (used by the RNN notebook, not committed to this repo) — download from Kaggle and place IMDB Dataset.csv in the RNN/ folder.

## Tech Stack

- Python 3.x
- PyTorch & torchvision
- scikit-learn
- pandas / numpy
- matplotlib
- NLTK (text preprocessing for the RNN notebook)

## Setup

```bash
git clone <repo-url>
cd <repo-folder>
pip install torch torchvision pandas numpy scikit-learn matplotlib nltk
```

CIFAR-10 downloads automatically via `torchvision.datasets.CIFAR10` on first run, so the `CNN/` folder needs no dataset of its own. The `ANN/` and `RNN/` folders already contain their datasets alongside the notebooks (`DateFruit_Dataset.csv`, `powerplant_data.csv`, `IMDB Dataset.csv`), so each notebook's relative `pd.read_csv(...)` path works as-is as long as you run it from within its own folder. The RNN notebook also pulls a few NLTK corpora (`punkt`, `punkt_tab`, `stopwords`) on first run, so an internet connection is needed the first time.

Launch any notebook with:

```bash
jupyter notebook
```

---

## 1. ANN — Classification (Date Fruit Varieties)

A feedforward network that classifies date fruits into 7 varieties (BERHI, DEGLET, DOKOL, IRAQI, ROTANA, SAFAVI, SOGAY) from 34 morphological, color, and texture features extracted from fruit images.

**Pipeline:** label encoding of the target, train/test split (80/20), feature standardization with `StandardScaler`, then tensors loaded via `TensorDataset` / `DataLoader`.

**Architecture:**
```
Input (34) → Linear(34, 64) → ReLU → Linear(64, 64) → ReLU → Linear(64, 7)
```

**Training:** Adam optimizer, Cross-Entropy loss, 100 epochs, batch size 32.

**Result:** 93.33% test accuracy, with a full confusion matrix and per-class precision/recall/F1 in the notebook.

## 2. ANN — Regression (Power Plant Energy Output)

A feedforward network that predicts the net hourly electrical energy output (PE) of a combined-cycle power plant from four ambient operating conditions: temperature (AT), exhaust vacuum (V), ambient pressure (AP), and relative humidity (RH).

**Pipeline:** train/test split (80/20), feature standardization, tensors via `TensorDataset` / `DataLoader`.

**Architecture:**
```
Input (4) → Linear(4, 6) → ReLU → Linear(6, 6) → ReLU → Linear(6, 1)
```

**Training:** Adam optimizer, MSE loss, 100 epochs, batch size 32. The notebook checkpoints the model whenever validation loss improves, saving the best weights to `Best_model.pt`.

**Result:** Test MSE ≈ 17.07, R² ≈ 0.94 on held-out data.

## 3. CNN — CIFAR-10 Image Classification

A convolutional network classifying 32×32 color images into CIFAR-10's 10 object categories.

**Pipeline:** images converted to tensors and normalized to [-1, 1]; loaded in batches of 64.

**Architecture:**
```
[Conv2d(3→32, k3) → ReLU → MaxPool(2,2)]
[Conv2d(32→64, k3) → ReLU → MaxPool(2,2)]
[Conv2d(64→128, k3) → ReLU → MaxPool(2,2)]
Flatten → Linear(2048, 256) → ReLU → Linear(256, 10)
```

**Training:** Adam optimizer, Cross-Entropy loss, 11 epochs.

**Result:** 76.1% test accuracy. Validation loss bottoms out around epoch 6 and rises afterward while training loss keeps falling — a sign the model starts overfitting past that point.

## 4. RNN — IMDB Sentiment Analysis

A many-to-one RNN that classifies movie reviews as positive or negative.

**Text preprocessing pipeline:** lowercasing, URL/punctuation/HTML-tag removal, stopword removal, Porter stemming, label encoding of sentiment, and TF-IDF vectorization (top 5,000 terms) — applied to the 50,000-review IMDB dataset (49,582 after de-duplication).

**Architecture:**
```
TF-IDF vector (5000) → nn.RNN(input_size=5000, hidden_size=128, num_layers=1, batch_first=True)
→ take final timestep → Linear(128, 1) → Sigmoid
```

**Training:** Adam optimizer, BCE loss, 10 epochs, batch size 64.

**Result:** 85.53% test accuracy.

---

## Results Summary

| Notebook | Architecture | Task | Metric | Score |
|---|---|---|---|---|
| ANN_classification | Feedforward NN | 7-class date fruit classification | Accuracy | 93.33% |
| ANN_regression | Feedforward NN | Power plant energy output regression | R² | 0.94 |
| CNN_for_CIFAR10 | CNN | 10-class image classification | Accuracy | 76.1% |
| RNN_sentiment-analysis | RNN | Binary sentiment classification | Accuracy | 85.53% |

## Possible Next Steps

A few directions each notebook could be extended in: adding dropout/batch norm and data augmentation to the CNN to address the overfitting visible after epoch 6; swapping the vanilla RNN for an LSTM or GRU and TF-IDF for learned/pretrained embeddings, which usually generalizes better on sentiment tasks; and replacing the manual `sigmoid` + `BCELoss` combination in the RNN notebook with `BCEWithLogitsLoss` for better numerical stability.
