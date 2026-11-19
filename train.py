import json
import joblib
from pathlib import Path
import mlflow
import mlflow.sklearn
# Client API lets us manage registered models and their aliases
from mlflow import MlflowClient

# Configure MLflow tracking
mlflow.set_tracking_uri("http://127.0.0.1:7001")

# Group related training runs under one experiment
mlflow.set_experiment("driftsense-classification")

# Model was trained on Google Colab (colab_train.py) on batch 1
# Here we load it with its saved metrics, then track and register it in MLflow
model_dir = Path("models")
model_path = model_dir / "driftsense.joblib"
metrics_path = model_dir / "metrics.json"

clf = joblib.load(model_path)
with open(metrics_path) as f:
    saved = json.load(f)

print("Model loaded from", model_path)

model_parameters = saved["params"]

# Start a tracked run
with mlflow.start_run() as run:
    # Record the training configuration
    mlflow.log_params(model_parameters)

    accuracy = saved["test_accuracy"]
    precision = saved["test_precision"]
    recall = saved["test_recall"]
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")

    # Model Quality Gate
    # Define the min acceptable test accuracy for now
    MIN_TEST_ACCURACY = 0.9
    if accuracy < MIN_TEST_ACCURACY:
        raise ValueError(
            f"Model rejected: test accuracy: {accuracy:.2f} "
            f"is below the required {MIN_TEST_ACCURACY}"
        )
    print("Model Quality Gate Passed...")

    mlflow.log_metrics({
        "test_accuracy": accuracy,
        "test_precision": precision,
        "test_recall": recall,
    })

    # Drift curve: accuracy of the same model on later batches
    for batch, acc_b in saved["accuracy_on_batch"].items():
        mlflow.log_metric("accuracy_on_batch", acc_b, step=int(batch))
        print(f"batch {batch}: {acc_b:.3f}")

    # Model registration
    registered_model_name = "driftsense-classifier"
    mlflow.sklearn.log_model(
        sk_model=clf,
        name="driftsense_model",
        registered_model_name=registered_model_name,
        skops_trusted_types=[
            "sklearn.tree._tree.Tree"
        ],
    )

    # Assign the champion alias to the newly registered model
    client = MlflowClient()
    # Identify current training run
    # We use this run id to find the model version registered by this particular run
    current_run_id = mlflow.active_run().info.run_id
    # Retrieve all versions registered under this model name
    registered_versions = client.search_model_versions(
        f"name='{registered_model_name}'"
    )
    # Find the version created by the current training run
    current_model_version = next(
        (
            version
            for version in registered_versions
            if version.run_id == current_run_id
        ),
        None,
    )
    client.set_registered_model_alias(
        name=registered_model_name,
        alias="champion",
        version=current_model_version.version,
    )

    print("Registered Model: driftsense-classifier")
