import sys, os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

BATCH = int(sys.argv[1]) if len(sys.argv) > 1 else 3  
os.makedirs("reports", exist_ok=True)


def load_batch(n, clean=True):
    X, y = [], []
    for line in open(f"data/batch{n}.dat"):
        if not line.strip():
            continue
        head, *pairs = line.split()
        y.append(int(head.split(";")[0]) - 1)
        row = np.full(128, np.nan)             
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


def old_split():       
    X, y = load_batch(1)
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def new_split(X, y):   
    return train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
