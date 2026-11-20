# Usage: python task_a_validate.py 3
import json
import numpy as np
from common import *

X, y = load_batch(BATCH, clean=False)
labels, counts = np.unique(y, return_counts=True)

report = {
    "batch": BATCH,
    "rows": len(X),
    "features": X.shape[1],
    "missing_cells": int(np.isnan(X).sum()),
    "duplicate_rows": len(X) - len(np.unique(X, axis=0)),
    "target_distribution": dict(zip(labels.tolist(), counts.tolist())),
}

json.dump(report, open("reports/data_validation.json", "w"), indent=2)
print(report)
