import shutil

import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://127.0.0.1:7001")
m = mlflow.sklearn.load_model("models:/driftsense-classifier@champion")
mlflow.sklearn.save_model(
    m,
    "model",
    skops_trusted_types=["sklearn.tree._tree.Tree"],
)
# reference stats from Colab, needed by /drift in main.py
shutil.copy("models/reference_stats.json", "model/reference_stats.json")
print("Exported champion model + reference stats to model/")
