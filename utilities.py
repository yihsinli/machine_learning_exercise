# utilities.py
import os
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns


def encode_categorical(features_train, features_test):
    """One-hot encode categorical columns."""
    categorical_features = [
        col for col in features_train.columns
        if features_train[col].dtype not in ['int64', 'float64']
    ]
    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    encoded_features_train = encoder.fit_transform(features_train[categorical_features])
    encoded_features_df_train = pd.DataFrame(encoded_features_train, columns=encoder.get_feature_names_out(categorical_features))
    encoded_features_test = encoder.fit_transform(features_test[categorical_features])
    encoded_features_df_test = pd.DataFrame(encoded_features_test, columns=encoder.get_feature_names_out(categorical_features))
    print('Done: not numeric data -> onehot feature')


    # 3. Combine onehot features and other numeric features
    features_numeric = features_train.drop(columns=categorical_features)
    processed_features_train = pd.concat([features_numeric, encoded_features_df_train], axis=1)
    features_numeric = features_test.drop(columns=categorical_features)
    processed_features_test = pd.concat([features_numeric, encoded_features_df_test], axis=1)
    print('Done: combine onehot feature and numeric features')

    return processed_features_train, processed_features_test


def split_train_valid(features, target, test_size=0.2, random_state=42):
    """Split dataset into training and validation sets."""
    return train_test_split(features, target, test_size=test_size, random_state=random_state)


def preprocess_data(train_data, test_data):
    """Full preprocessing pipeline: drop cols, encode categorical, split, scale, tensorize."""
    # Drop irrelevant columns
    target = train_data['Exited']
    features_train = train_data.drop(['Exited', 'Surname', 'CustomerId', 'id'], axis=1)
    features_test = test_data.drop(['Surname', 'CustomerId', 'id'], axis=1)

    # Encode categorical
    processed_train, processed_test = encode_categorical(features_train, features_test)

    # Train-valid split
    X_train, X_valid, y_train, y_valid = split_train_valid(processed_train, target)

    # Normalize features
    scaler = StandardScaler()
    print(X_train.shape, X_valid.shape, processed_test.shape)
    X_train_scaled = scaler.fit_transform(X_train.values)
    X_valid_scaled = scaler.transform(X_valid.values)
    X_test_scaled = scaler.transform(processed_test.values)

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_valid_tensor = torch.tensor(X_valid_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).reshape(-1, 1)
    y_valid_tensor = torch.tensor(y_valid.values, dtype=torch.float32).reshape(-1, 1)

    return X_train_tensor, X_valid_tensor, X_test_tensor, y_train_tensor, y_valid_tensor


def create_balanced_loader(X_train_tensor, y_train_tensor, batch_size=64):
    """Create DataLoader with weighted sampling to handle class imbalance."""
    y = y_train_tensor.reshape(-1).long()
    dataset = TensorDataset(X_train_tensor, y)

    # Compute sampling weights (inverse class frequency)
    class_counts = torch.bincount(y)
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[y]
    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples=len(sample_weights),
                                    replacement=True)

    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return loader


def evaluate_model(name, model, X, y_true, threshold=0.5, out_dir="exercise1_plot"):
    """Evaluate a trained model: confusion matrix, ROC, classification report, and save plots."""
    os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        logits = model(X)
        if logits.shape[1] == 1:  # logistic regression
            probs = torch.sigmoid(logits).view(-1)
        else:  # MLP (2 outputs)
            probs = torch.softmax(logits, dim=1)[:, 1]

    preds = (probs >= threshold).long()

    # Classification report
    print(f"\n===== {name} (threshold={threshold}) =====")
    print(classification_report(y_true, preds, digits=3))

    # Confusion Matrix
    cm = confusion_matrix(y_true, preds)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, probs)
    roc_auc = auc(fpr, tpr)

    # Subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Confusion matrix subplot
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Pred 0", "Pred 1"],
                yticklabels=["True 0", "True 1"],
                ax=axes[0])
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    axes[0].set_title(f"{name} Confusion Matrix")

    # ROC curve subplot
    axes[1].plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    axes[1].plot([0, 1], [0, 1], "k--")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title(f"{name} ROC Curve")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{name}_visualization.jpg"))

    return probs.numpy()


# Plot ROC comparison
def plot_roc_comparison(all_probs, y, out_dir='exercise1_plot'):
    os.makedirs(out_dir,exist_ok=True)
    plt.figure(figsize=(6,6))
    for name, probs in all_probs.items():
        fpr, tpr, _ = roc_curve(y, probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")

    plt.plot([0,1], [0,1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,'ROC_3_model_comparison.jpg'))
    
def save_prediction(name, model, X_test, out_dir='exercise1_plot'):
    # 4. Predictions on Test Set

    os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        logits = model(X_test)
        probs_test = torch.softmax(logits, dim=1)[:, 1]
        preds = (probs_test >= 0.5).long()
    df = pd.DataFrame({"Predicted_Exited": preds.numpy()})
    df.to_csv(os.path.join(out_dir,"Predicted_Exited_from_test_{}.csv".format(name)), index=False)
    print("\n✅ Outputs saved: Predicted_Exited_from_test_{}.csv".format(name,name))