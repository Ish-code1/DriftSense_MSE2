import os
import sys

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

BATCH = int(sys.argv[1]) if len(sys.argv) > 1 else 3   # new data batch
os.makedirs("reports", exist_ok=True)


def load_batch(n, clean=True):
    X, y = [], []
    with open(f"data/batch{n}.dat") as f:
        for line in f:
            if not line.strip():
                continue
            head, *pairs = line.split()
            y.append(int(head.split(";")[0]) - 1)
            row = np.full(128, np.nan)              # NaN = missing feature
            for p in pairs:
                i, v = p.split(":")
                row[int(i) - 1] = float(v)
            X.append(row)
    X = np.array(X)
    return (np.nan_to_num(X) if clean else X), np.array(y)


def scores(model, X, y):
    p = model.predict(X)
    return {
        "accuracy": accuracy_score(y, p),
        "precision": precision_score(y, p, average="macro", zero_division=0),
        "recall": recall_score(y, p, average="macro", zero_division=0),
        "f1": f1_score(y, p, average="macro", zero_division=0),
    }


def old_split():       # same split as colab_train.py (batch 1)
    X, y = load_batch(1)
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def new_split(X, y):   # 70% for retraining, 30% holdout for the gate
    return train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)