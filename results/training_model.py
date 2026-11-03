"""Train a TF-IDF and logistic regression topic classifier."""

import argparse
import hashlib
import json
from pathlib import Path
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, learning_curve
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
TOPICS = {
    "tech": "Tech",
    "sport": "Sport",
    "business": "Business",
    "entertainment": "Entertainment",
    "politics": "Politics",
}


def load_data(path):
    data = pd.read_csv(path)
    required_columns = {"Text", "Category"}
    if not required_columns.issubset(data.columns):
        raise ValueError(f"{path} must contain Text and Category columns")

    data = data[["Text", "Category"]].copy()
    valid_text = data["Text"].map(lambda text: isinstance(text, str) and bool(text.strip()))
    if data.empty or not valid_text.all():
        raise ValueError(f"{path} contains missing or empty article text")

    categories = data["Category"].str.strip().str.lower()
    data["Category"] = categories.map(TOPICS)
    if data["Category"].isna().any():
        raise ValueError(f"{path} contains missing or unsupported categories")
    return data


def text_key(text):
    return " ".join(text.casefold().split())


def prepare_training_data(train, test):
    train = train.copy()
    train["text_key"] = train["Text"].map(text_key)

    labels_per_text = train.groupby("text_key")["Category"].nunique()
    if labels_per_text.gt(1).any():
        raise ValueError("Identical training texts have conflicting topic labels")

    original_count = len(train)
    train = train.drop_duplicates("text_key")
    duplicate_count = original_count - len(train)

    # Remove shared stories from training only; never edit either CSV.
    test_keys = set(test["Text"].map(text_key))
    overlapping_texts = train["text_key"].isin(test_keys)
    overlap_count = int(overlapping_texts.sum())
    train = train.loc[~overlapping_texts, ["Text", "Category"]].reset_index(drop=True)

    available_topics = set(train["Category"])
    smallest_class = train["Category"].value_counts().min()
    if available_topics != set(TOPICS.values()) or smallest_class < 5:
        raise ValueError("Training data must contain at least five distinct stories per topic")

    audit = {
        "original_training_rows": original_count,
        "duplicate_training_rows_removed": duplicate_count,
        "training_texts_overlapping_test_removed": overlap_count,
        "training_rows_used": len(train),
        "test_rows": len(test),
        "test_rows_removed": 0,
    }
    return train, audit


def build_classifier():
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        sublinear_tf=True,
        ngram_range=(1, 2),
        min_df=2,
    )
    classifier = LogisticRegression(C=5.0, max_iter=2000, random_state=42)

    # Saving one pipeline also saves the vocabulary needed for prediction.
    return Pipeline([
        ("tfidf", vectorizer),
        ("logistic_regression", classifier),
    ])


def plot_learning_curves(model, texts, labels, output_dir):
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Each fold learns its own TF-IDF vocabulary, using training texts only.
    sizes, training_scores, validation_scores = learning_curve(
        model,
        texts,
        labels,
        cv=folds,
        scoring="accuracy",
        train_sizes=np.array([0.1, 0.25, 0.5, 0.75, 1.0]),
        shuffle=True,
        random_state=42,
        n_jobs=1,
        error_score="raise",
    )

    figure, axes = plt.subplots(figsize=(8, 5))
    curves = [
        (training_scores, "Training accuracy", "#2563eb"),
        (validation_scores, "Validation accuracy (5-fold CV)", "#059669"),
    ]
    for scores, label, color in curves:
        mean_score = scores.mean(axis=1)
        standard_deviation = scores.std(axis=1)
        lower_bound = np.clip(mean_score - standard_deviation, 0, 1)
        upper_bound = np.clip(mean_score + standard_deviation, 0, 1)

        axes.plot(sizes, mean_score, "o-", label=label, color=color)
        axes.fill_between(sizes, lower_bound, upper_bound, color=color, alpha=0.15)

    axes.axhline(0.95, color="#64748b", linestyle="--", linewidth=1, label="95% target")
    axes.set(
        title="TF-IDF + logistic regression learning curves",
        xlabel="Articles used for training per fold",
        ylabel="Accuracy",
    )
    axes.yaxis.set_major_formatter(PercentFormatter(1.0))
    axes.set_ylim(top=1.01)
    axes.grid(alpha=0.2)
    axes.legend(loc="lower right")
    figure.text(
        0.5, 0.015,
        "Shading: ±1 standard deviation. The held-out test set is not used in these curves.",
        ha="center", fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig(output_dir / "learning_curves.png", dpi=180)
    plt.close(figure)

    return {
        "training_sizes": sizes.tolist(),
        "training_accuracy_by_fold": training_scores.tolist(),
        "validation_accuracy_by_fold": validation_scores.tolist(),
        "final_cv_accuracy_mean": float(validation_scores[-1].mean()),
        "final_cv_accuracy_std": float(validation_scores[-1].std()),
    }


def evaluate_model(model, train, test):
    expected_topics = test["Category"]
    predictions = model.predict(test["Text"])
    test_accuracy = float(accuracy_score(expected_topics, predictions))
    correct_predictions = int((predictions == expected_topics.to_numpy()).sum())

    metrics = {
        "training_accuracy": float(model.score(train["Text"], train["Category"])),
        "test_accuracy": test_accuracy,
        "correct_test_predictions": correct_predictions,
        "requirement_test_accuracy_above_95_percent": test_accuracy > 0.95,
        "topics": model.classes_.tolist(),
        "vocabulary_size": len(model.named_steps["tfidf"].vocabulary_),
        "classification_report": classification_report(
            expected_topics, predictions, output_dict=True, zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            expected_topics, predictions, labels=model.classes_,
        ).tolist(),
    }
    return metrics, predictions


def save_results(model, metrics, output_dir, model_filename):
    with (output_dir / model_filename).open("wb") as model_file:
        pickle.dump(model, model_file, protocol=pickle.HIGHEST_PROTOCOL)

    metrics_file = output_dir / "topic_metrics.json"
    metrics_file.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


def train_model(train_path, test_path, output_dir, model_filename="topic_classifier.pkl"):
    train = load_data(train_path)
    test = load_data(test_path)
    train, audit = prepare_training_data(train, test)

    print(f"Training rows: {audit['training_rows_used']}; test rows: {audit['test_rows']}", flush=True)
    print(
        f"Excluded in memory: {audit['duplicate_training_rows_removed']} duplicate training rows, "
        f"{audit['training_texts_overlapping_test_removed']} training texts shared with test.",
        flush=True,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    model = build_classifier()

    # A single BLAS thread avoids overhead for this small sparse dataset.
    with threadpool_limits(limits=1):
        print("Computing five-fold learning curves...", flush=True)
        curves = plot_learning_curves(model, train["Text"], train["Category"], output_dir)
        print("Fitting the final pipeline on the training data...", flush=True)
        model.fit(train["Text"], train["Category"])
        metrics, predictions = evaluate_model(model, train, test)

    metrics.update({
        "model": "TF-IDF + LogisticRegression",
        "sklearn_version": sklearn.__version__,
        "random_state": 42,
        "data_audit": audit,
        "source_sha256": {
            "training": hashlib.sha256(train_path.read_bytes()).hexdigest(),
            "test": hashlib.sha256(test_path.read_bytes()).hexdigest(),
        },
        "learning_curves": curves,
    })
    save_results(model, metrics, output_dir, model_filename)

    print(classification_report(test["Category"], predictions, digits=4, zero_division=0))
    print(f"Training accuracy: {metrics['training_accuracy']:.2%}")
    print(f"Cross-validation accuracy: {curves['final_cv_accuracy_mean']:.2%}")
    print(
        f"Test accuracy: {metrics['test_accuracy']:.2%} "
        f"({metrics['correct_test_predictions']}/{len(test)})"
    )
    print(f"Saved model, learning curves and metrics in {output_dir}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, default=ROOT / "bbc_news_train.csv")
    parser.add_argument("--test-csv", type=Path, default=ROOT / "bbc_news_tests.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    metrics = train_model(args.train_csv, args.test_csv, args.output_dir)
    if not metrics["requirement_test_accuracy_above_95_percent"]:
        print("The test accuracy requirement (>95%) was not met.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
