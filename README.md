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

1. **Confusion matrix + ROC curve** plots (threshold=0.5 and threshold=0.9).

   * Saved under `exercise1_plot/trained_model.jpg`.
2. **Trained model parameters** saved to `trained_model.pth`.

---

### 4. Evaluation of Pretrained Models

Three pretrained models are evaluated on the validation set:

* `simple_mlp_normalized.pth`
* `simple_mlp_leakyrelu.pth`
* `logitregression.pth`

Outputs:

1. **Confusion matrix + ROC curve** plots for each model.

   * Saved under `exercise1_plot/{model_name}_visualization.jpg`.
2. **ROC comparison plot** across all three models.

   * Saved under `exercise1_plot/ROC_3_model_comparison.jpg`.

---

### 5. Predictions on Test Set

* Each model predicts churn (`Exited`) on `test.csv`.
* Predictions are thresholded at 0.5.
* Outputs:

  * `model_predictions_test.csv` → predictions from all three pretrained models.
  * `Predicted_Exited_from_test.csv` → predictions from the trained model.

---

## Final Outputs

### Trained Model

* ✅ `exercise1_plot/trained_model.jpg` (confusion matrix + ROC)
* ✅ `trained_model.pth` (model weights)
* ✅ `Predicted_Exited_from_test.csv` (predictions on test set)

### Pretrained Models

* ✅ `exercise1_plot/{model_name}_visualization.jpg` (confusion matrix + ROC for each model)
* ✅ `exercise1_plot/ROC_3_model_comparison.jpg` (combined ROC comparison)
* ✅ `model_predictions_test.csv` (predictions on test set)

---

## Requirements

* Python 3.8+
* Libraries: `torch`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `tqdm`

---

## Run Instructions

```bash
python main.py
```

Make sure `train.csv` and `test.csv` are present in the same directory.

---

Absolutely! We can add a **“Discussion & Potential Improvements”** section to your README. Here’s a refined version that integrates your points clearly and professionally:

---

## Discussion & Potential Improvements

1. **Model architecture and features**

   * The current MLP may be limited in capacity. Accuracy could potentially be improved by exploring deeper neural networks or alternative architectures beyond simple MLPs.
   * Feature engineering can also help. For example, numeric features like `Age` could be categorized into levels and one-hot encoded, or all categorical features could be encoded more comprehensively.

2. **Feature selection**

   * Not all features may provide useful information. Some, such as `Gender` or raw `Age`, might be less informative.
   * Dynamic or iterative feature selection could help identify the most relevant subset of features, improving model performance and reducing overfitting.

3. **Data imbalance**

   * The current approach uses weighted sampling to handle class imbalance.
   * Further improvements could include adaptive weighting of the loss function (e.g., assigning higher weights to minority classes) or using oversampling/undersampling techniques.

4. **Loss function alternatives**

   * Currently, `CrossEntropyLoss` is used for multi-class outputs (or BCE for single output).
   * More robust loss functions could be considered, such as **angular loss** or other specialized classification losses, to improve learning under class imbalance or noisy labels.

---