import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from tqdm import tqdm
import os
from models import MLP_relu, MLP_leakyrelu, LogisticRegression
from utilities import preprocess_data, create_balanced_loader, evaluate_model,plot_roc_comparison,save_prediction
import argparse

def train_model(model, loader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        with tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for xb, yb in pbar:
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                avg_loss = running_loss / (pbar.n + 1)
                pbar.set_postfix(loss=avg_loss)


def main(args):

    # === Read data ===
    train_data = pd.read_csv(args.train_file)
    test_data = pd.read_csv(args.test_file)
    # === Preprocess ===
    X_train, X_valid, X_test, y_train, y_valid = preprocess_data(train_data,test_data)
    loader = create_balanced_loader(X_train, y_train)

    input_size = X_train.shape[1]
    model = MLP_relu(input_dim=input_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # === Train ===
    train_model(model, loader, criterion, optimizer, epochs=args.epochs)

    # === Evaluate trained model ===
    print("\n======== Evaluate trained model on valid dataset ========\n")
    probs = evaluate_model("Trained_MLP_ReLU", model, X_valid, y_valid, threshold=args.thres)
    if args.save_model:
        torch.save(model.state_dict(), os.path.join('checkpoints',"{}.pth".format('Trained_MLP_ReLU')))

    # === Save outputs ===
    save_prediction("trained_mlp_relu", model, X_test, out_dir='exercise1_predicted')

    # === Evaluate Pretrained model ===
    if args.pretrained:
        print("\n======== Evaluate pretrained model on valid dataset ========\n")
        input_size = X_valid.shape[1]
        # ReLU MLP
        model_relu = MLP_relu(input_dim=input_size)
        model_relu.load_state_dict(torch.load("checkpoints/simple_mlp_normalized.pth"))
        model_relu.eval()
        # LeakyReLU MLP
        model_leaky = MLP_leakyrelu(input_dim=input_size)
        model_leaky.load_state_dict(torch.load("checkpoints/simple_mlp_leakyrelu.pth"))
        model_leaky.eval()
        # Logistic Regression
        model_logreg = LogisticRegression(input_dim=input_size)
        model_logreg.load_state_dict(torch.load("checkpoints/logitregression.pth"))
        model_logreg.eval()

        models = {
            "MLP (ReLU)": model_relu,
            "MLP (LeakyReLU)": model_leaky,
            "Logistic Regression": model_logreg
        }

        all_probs = {}

        for i, (name, model) in enumerate(models.items()):
            probs = evaluate_model(name, model, X_valid, y_valid, threshold=0.9)
            save_prediction(name, model, X_test, out_dir='exercise1_predicted')
            all_probs[name] = probs
        
        plot_roc_comparison(all_probs, y_valid, out_dir='exercise1_plot')
    
    


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run ML/DL training pipeline")
    parser.add_argument("--train_file", type=str, default='train.csv', help="Path to training CSV file")
    parser.add_argument("--test_file", type=str, default='test.csv', help="Path to test CSV file")
    parser.add_argument("--model", type=str, default="mlp_relu",
                        choices=["mlp_relu", "mlp_leakyrelu", "logreg"],
                        help="Model to train and evaluate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--thres", type=float, default=0.9, help="Threshold of binary class")
    parser.add_argument("--pretrained", action="store_true", help="Evaluate Pretrained model")
    parser.add_argument("--save_model", action="store_true", help="Save trained model to disk")

    args = parser.parse_args()
    main(args)