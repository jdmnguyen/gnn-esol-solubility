import os
import sys

# --- Mac OpenMP fix (avoid libomp crash) ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

# --- Make sure we can import from src/ when running from notebooks/ ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import torch
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt

from src.dataset import ESOLDataset
from src.models import SolubilityGNN
from src.utils import split_dataset, compute_metrics


def main():
    # 1. Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 2. Load dataset and create the SAME split as in training
    dataset = ESOLDataset(root="data")
    train_dataset, val_dataset, test_dataset = split_dataset(dataset)  # same seed inside split_dataset

    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 3. Recreate model architecture (must match training)
    sample_data = dataset[0]
    in_channels = sample_data.x.shape[1]

    model = SolubilityGNN(
        in_channels=in_channels,
        hidden_channels=64,
        num_layers=3,
    )
    model.to(device)

    # 4. Load best saved weights
    model_path = os.path.join("results", "models", "best_model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Could not find saved model at {model_path}. "
            "Make sure you have run `python -m src.train` first."
        )

    # For PyTorch >= 2.6, weights_only=True is default; here it's fine because we saved a pure state_dict
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 5. Run predictions on the test set
    ys = []
    preds = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            ys.extend(batch.y.view(-1).cpu().numpy())
            preds.extend(out.cpu().numpy())

    # 6. Compute metrics
    metrics = compute_metrics(ys, preds)
    print("Test metrics from analysis script:", metrics)

    # 7. Make and save plot: True vs Predicted logS
    os.makedirs(os.path.join("results", "figures"), exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.scatter(ys, preds, alpha=0.7)

    plt.xlabel("True logS")
    plt.ylabel("Predicted logS")
    plt.title("ESOL: True vs Predicted logS (Test Set)")

    # Diagonal reference line
    min_val = min(min(ys), min(preds))
    max_val = max(max(ys), max(preds))
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

    # Annotate with metrics
    text_str = f"RMSE = {metrics['rmse']:.3f}\nR² = {metrics['r2']:.3f}"
    plt.text(
        0.05,
        0.95,
        text_str,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    out_path = os.path.join("results", "figures", "pred_vs_true.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {out_path}")

    plt.show()


if __name__ == "__main__":
    main()
