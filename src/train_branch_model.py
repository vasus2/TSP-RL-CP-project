import csv
import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from learning_features import FEATURE_NAMES


def load_training_data(input_path):
    df = pd.read_csv(input_path)
    x = df[FEATURE_NAMES].to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=int)
    return x, y


def evaluate_model(name, model, x_test, y_test):
    scores = model.predict_proba(x_test)[:, 1]
    preds = (scores >= 0.5).astype(int)
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, scores),
        "average_precision": average_precision_score(y_test, scores),
    }


def save_model(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(model, f)
    print(f"Saved model to: {path}")


def train_branch_models(
    input_path="results/branch_data.csv",
    model_dir="models",
    metrics_path="results/branch_model_metrics.csv",
):
    x, y = load_training_data(input_path)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=0,
        stratify=y,
    )

    models = {
        "logistic_branch": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=0),
        ),
        "random_forest_branch": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=0,
            n_jobs=-1,
        ),
    }

    metrics = []
    model_dir = Path(model_dir)
    for name, model in models.items():
        model.fit(x_train, y_train)
        metrics.append(evaluate_model(name, model, x_test, y_test))
        save_model(model, model_dir / f"{name}.pkl")

    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "accuracy", "roc_auc", "average_precision"])
        writer.writeheader()
        writer.writerows(metrics)

    print(f"Saved training metrics to: {metrics_path}")
    return metrics


if __name__ == "__main__":
    train_branch_models()
