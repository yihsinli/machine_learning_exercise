# Customer Churn Prediction – ML/DL Models

## Project Overview

This project implements and compares multiple classification models (MLP with ReLU, MLP with LeakyReLU, and Logistic Regression) for predicting customer churn (`Exited`). The pipeline includes:

* **Data preprocessing** (dropping identifiers, one-hot encoding categorical features, normalization).
* **Model training** (custom MLP with weighted sampling to handle class imbalance).
* **Model evaluation** (confusion matrix, classification report, ROC curve).
* **Comparison of pretrained models** (visual and numerical evaluation).
* **Export of predictions** on the test dataset to CSV.

---

## Workflow

### 1. Data Preparation

* Input files:

  * `train.csv` (with target `Exited`)
  * `test.csv` (without target)
* Processing steps:

  * Remove identifiers (`id`, `CustomerId`, `Surname`).
  * One-hot encode categorical features.
  * Normalize numeric features using `StandardScaler`.
  * Split `train.csv` into **train/validation** sets (80/20).

---

### 2. Models Implemented

* **MLP (ReLU)**
* **MLP (LeakyReLU)**
* **Logistic Regression**

Each model outputs logits that are converted into probabilities of class `1` (Exited).

---

### 3. Training (Trained Model)

* Model trained: `MLP_relu`.
* Loss: `CrossEntropyLoss`.
* Optimizer: `Adam`.
* Balanced sampling with `WeightedRandomSampler`.
* Number of epochs: **10**.

Outputs:

1. **Confusion matrix + ROC curve** plots (default threshold=0.9).
   → Saved under `exercise1_plot/Trained_MLP_ReLU_visualization.jpg`.
2. **Trained model parameters** saved to `checkpoints/Trained_MLP_ReLU.pth` if save_model flag is true (see parameter setting below).

---

### 4. Evaluation of Pretrained Models (option, see parameter setting below)

Three pretrained models are evaluated on the validation set:

* `simple_mlp_normalized.pth`
* `simple_mlp_leakyrelu.pth`
* `logitregression.pth`

Outputs:

1. **Confusion matrix + ROC curve** plots for each model.
   → Saved under `exercise1_plot/{model_name}_visualization.jpg`.
2. **ROC comparison plot** across all three models.
   → Saved under `exercise1_plot/ROC_3_model_comparison.jpg`.

---

### 5. Predictions on Test Set

* Each model predicts churn (`Exited`) on `test.csv`.
* Predictions are thresholded at 0.5.
* Outputs:

  * `model_predictions_test.csv` → predictions from all three pretrained models.
  * `Predicted_Exited_from_test.csv` → predictions from the trained model.

---

## Run Instructions

The script can be controlled via command-line arguments:

```bash
python main.py \
    --train_file train.csv \
    --test_file test.csv \
    --model mlp_relu \
    --epochs 10 \
    --lr 0.001 \
    --thres 0.9 \
    --pretrained \
    --save_model
```

### Available Parameters

* `--train_file` (str, default=`train.csv`): Path to training CSV file
* `--test_file` (str, default=`test.csv`): Path to test CSV file
* `--model` (str, default=`mlp_relu`): Model to train and evaluate. Options: `mlp_relu`, `mlp_leakyrelu`, `logreg`
* `--epochs` (int, default=`10`): Number of training epochs
* `--lr` (float, default=`0.001`): Learning rate
* `--thres` (float, default=`0.9`): Threshold for binary classification
* `--pretrained` (flag): If set, evaluates pretrained models instead of training
* `--save_model` (flag): If set, saves the trained model to disk

---

## Final Outputs

### Trained Model

* ✅ `exercise1_plot/Trained_MLP_ReLU_visualization.jpg` (confusion matrix + ROC)
* ✅ `checkpoints/Trained_MLP_ReLU.pth` (model weights)
* ✅ `exercise1_predicted/Predicted_Exited_from_test_Trained_MLP_ReLU.csv` (predictions on test set)

### Pretrained Models

* ✅ `exercise1_plot/{model_name}_visualization.jpg` (confusion matrix + ROC for each model)
* ✅ `exercise1_plot/ROC_3_model_comparison.jpg` (combined ROC comparison)
* ✅ `exercise1_predicted/Predicted_Exited_from_test_{model_name}.csv` (predictions on test set)

---

## Requirements

* Python 3.8+
* Libraries: `torch`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `tqdm`

---


## Discussion & Potential Improvements

1. **Model architecture and features**

   * The current MLP may be limited in capacity. Accuracy could potentially be improved by exploring deeper neural networks or alternative architectures beyond simple MLPs.
   * Feature engineering can also help. For example, numeric features like `Age` could be categorized into levels and one-hot encoded, or all categorical features could be encoded more comprehensively.

2. **Feature selection**

   * Not all features may provide useful information. Some, such as `Gender` or raw `Age`, might be less informative.
   * Dynamic or iterative feature selection could help identify the most relevant subset of features, improving model performance and reducing overfitting.
   * See [Dynamic Feature Selection](https://github.com/iancovert/dynamic-selection?tab=readme-ov-file) for a reference implementation.

3. **Data imbalance**

   * The current approach uses weighted sampling to handle class imbalance.
   * Further improvements could include adaptive weighting of the loss function (e.g., assigning higher weights to minority classes) or using oversampling/undersampling techniques.

4. **Loss function alternatives**

   * Currently, `CrossEntropyLoss` is used for multi-class outputs (or BCE for single output).
   * More robust loss functions could be considered, such as **angular loss** or other specialized classification losses, to improve learning under class imbalance or noisy labels.

---
