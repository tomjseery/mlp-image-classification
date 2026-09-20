# MLP Image Classification

A data-mining and machine-learning study built around scikit-learn's `MLPClassifier`. It loads a folder-structured scene-image dataset, reduces each image to 32×32 pixels and investigates how preprocessing, architecture and evaluation choices affect classification performance.

## What the project explores

- Image loading, resizing, flattening and label encoding
- Feature standardisation with a target standard deviation of 2.5
- Staged tuning of hidden layers, activation, solver, learning rate and batch size
- Manual epoch-by-epoch training with loss and accuracy histories
- L2 regularisation analysis across a range of `alpha` values
- Precision, recall, F1 and confusion-matrix evaluation
- K-fold versus stratified K-fold cross-validation
- Paired hypothesis testing of cross-validation scores
- Locally Linear Embedding (LLE) for two-dimensional visualisation

## Repository layout

| File | Purpose |
| --- | --- |
| `coc131_cw.py` | Reusable `COC131` implementation for questions 1–6 |
| `dm_ml_cw.ipynb` | Analysis, visualisations, recorded experiments and discussion |

## Dataset

The image dataset is not included. Supply it as one directory per class:

```text
dataset/
├── class-a/
│   ├── image-001.jpg
│   └── image-002.jpg
├── class-b/
│   └── image-003.jpg
└── ...
```

Class labels are derived from the directory names. Images must be readable by Pillow.

## Running the analysis

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install jupyter numpy scipy pillow matplotlib scikit-learn
jupyter notebook dm_ml_cw.ipynb
```

Place the dataset beside the notebook before running it from top to bottom. Some stages train many MLP configurations and may take considerable time.

## Status

This repository preserves a university coursework investigation. The notebook contains the original analysis and outputs, while `coc131_cw.py` keeps the assessed implementation separate from presentation code.
