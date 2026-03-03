# src/utils.py
import torch
from torch.utils.data import random_split
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def split_dataset(dataset, train_ratio=0.8, val_ratio=0.1, seed=42):
    """Randomly split into train/val/test."""
    n = len(dataset)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)
    n_test = n - n_train - n_val

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [n_train, n_val, n_test], generator=generator
    )
    return train_dataset, val_dataset, test_dataset

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return {
        "rmse": rmse(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }
