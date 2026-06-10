from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

_CACHE_ROOT = Path(tempfile.gettempdir()) / "dl4m-evaluation-cache"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eval_utils import (
    NOISE_TYPES, TRANSFORM_TYPES,
    EmbeddingItem, safe_label, model_labels,
    list_data, load_matrix, per_genre_bar_chart,
    apply_style, SNR_PALETTE, LEVEL_PALETTE, MODEL_PALETTE,
)

apply_style()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a genre classifier on clean GTZAN embeddings and evaluate it "
            "on noisy/augmented GTZAN embeddings."
        )
    )
    parser.add_argument("--embedding-root", action="append", type=Path, required=True,
                        help="Root folder containing Data/genres_original and Data/genres_augmented.")
    parser.add_argument("--model-label", action="append",
                        help="Label for the corresponding --embedding-root.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/genre_classification"),
                        help="Directory for classification tables and plots.")
    parser.add_argument("--results-csv", type=Path, default=None,
                        help="Optional path for per-file predictions. Only written with --write-csv.")
    parser.add_argument("--write-csv", action="store_true",
                        help="Write CSV tables in addition to graph images.")
    parser.add_argument("--c", type=float, default=1.0,
                        help="Inverse regularization strength for logistic regression.")
    parser.add_argument("--seed", type=int, default=24, help="Random seed for the classifier.")
    return parser.parse_args()


def train_classifier(train_items: list[EmbeddingItem], c_value: float, seed: int):
    x_train = load_matrix(train_items)
    y_train = [item.genre for item in train_items]
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c_value, class_weight="balanced", max_iter=5000, random_state=seed),
    )
    classifier.fit(x_train, y_train)
    return classifier, sorted(set(y_train))


def evaluate_model(root: Path, model: str, c_value: float, seed: int) -> pd.DataFrame:
    train_items, test_items = list_data(root)
    classifier, _ = train_classifier(train_items, c_value=c_value, seed=seed)
    x_test = load_matrix(test_items)
    predictions = classifier.predict(x_test)
    probabilities = classifier.predict_proba(x_test)
    classes = classifier.classes_
    max_probabilities = probabilities.max(axis=1)

    rows = []
    for item, predicted, confidence in zip(test_items, predictions, max_probabilities):
        true_probability = probabilities[len(rows), int(np.flatnonzero(classes == item.genre)[0])]
        rows.append({
            "model": model,
            "track_id": item.track_id,
            "degradation_type": item.degradation_type,
            "degradation_value": item.degradation_value,
            "true_genre": item.genre,
            "predicted_genre": predicted,
            "correct": int(predicted == item.genre),
            "prediction_confidence": float(confidence),
            "true_genre_probability": float(true_probability),
            "query_path": item.relative_path,
        })
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> dict[str, pd.DataFrame]:
    metrics = dict(
        num_queries=("correct", "count"),
        accuracy=("correct", "mean"),
        mean_prediction_confidence=("prediction_confidence", "mean"),
        mean_true_genre_probability=("true_genre_probability", "mean"),
    )
    by_snr_genre = (results.groupby(["model", "true_genre", "degradation_value"], as_index=False)
                    .agg(**metrics).sort_values(["model", "true_genre", "degradation_value"]))
    by_noise_snr_genre = (
        results.groupby(["model", "degradation_type", "degradation_value", "true_genre"], as_index=False)
        .agg(**metrics).sort_values(["model", "degradation_type", "degradation_value", "true_genre"]))
    by_snr = (results.groupby(["model", "degradation_value"], as_index=False)
              .agg(**metrics).sort_values(["model", "degradation_value"]))
    by_noise_snr = (
        results.groupby(["model", "degradation_type", "degradation_value"], as_index=False)
        .agg(**metrics).sort_values(["model", "degradation_type", "degradation_value"]))
    by_genre = (results.groupby(["model", "true_genre"], as_index=False)
                .agg(**metrics).sort_values(["model", "true_genre"]))
    confusion = (results.groupby(["model", "true_genre", "predicted_genre"], as_index=False)
                 .size().rename(columns={"size": "count"}))
    totals = (confusion.groupby(["model", "true_genre"], as_index=False)["count"]
              .sum().rename(columns={"count": "true_genre_total"}))
    confusion = confusion.merge(totals, on=["model", "true_genre"])
    confusion["rate"] = confusion["count"] / confusion["true_genre_total"]

    return {
        "by_snr_genre": by_snr_genre,
        "by_noise_snr_genre": by_noise_snr_genre,
        "by_snr": by_snr,
        "by_noise_snr": by_noise_snr,
        "by_genre": by_genre,
        "confusion": confusion,
    }


def plot_per_genre_by_noise_snr(by_noise_snr_genre: pd.DataFrame, output_dir: Path) -> None:
    noise_data = by_noise_snr_genre[by_noise_snr_genre["degradation_type"].isin(NOISE_TYPES)]
    for model in noise_data["model"].drop_duplicates():
        avg = (noise_data[noise_data["model"] == model]
               .groupby(["true_genre", "degradation_value"], as_index=False)
               .agg(accuracy=("accuracy", "mean")))
        per_genre_bar_chart(
            data=avg, metric="accuracy",
            level_order=[20, 10, 0],
            colors=SNR_PALETTE,
            legend_labels={20: "20 dB", 10: "10 dB", 0: "0 dB"},
            title=f"Per-Genre Classification Accuracy by SNR — Noise Augmentations\n{model}",
            xlabel="Top-1 genre accuracy (%) — averaged across crowd, street & white noise",
            legend_title="SNR",
            output_path=output_dir / f"per_genre_accuracy_by_noise_snr_{safe_label(model)}.png",
        )


def plot_per_genre_by_transform_level(by_noise_snr_genre: pd.DataFrame, output_dir: Path) -> None:
    transform_data = by_noise_snr_genre[by_noise_snr_genre["degradation_type"].isin(TRANSFORM_TYPES)]
    if transform_data.empty:
        return
    for model in transform_data["model"].drop_duplicates():
        avg = (transform_data[transform_data["model"] == model]
               .groupby(["true_genre", "degradation_value"], as_index=False)
               .agg(accuracy=("accuracy", "mean")))
        per_genre_bar_chart(
            data=avg, metric="accuracy",
            level_order=[1, 2, 3],
            colors=LEVEL_PALETTE,
            legend_labels={1: "Level 1", 2: "Level 2", 3: "Level 3"},
            title=f"Per-Genre Classification Accuracy by Level — Transform Augmentations\n{model}",
            xlabel="Top-1 genre accuracy (%) — averaged across pitch shift up, pitch shift down & lo-fi",
            legend_title="Severity Level",
            output_path=output_dir / f"per_genre_accuracy_by_transform_level_{safe_label(model)}.png",
        )


def plot_noise_snr_heatmap(by_noise_snr_genre: pd.DataFrame, output_path: Path) -> None:
    def condition_label(aug_type: str, level: int) -> str:
        names = {"crowd_noise": "Crowd", "street_noise": "Street", "white_noise": "White",
                 "pitch_shift_up": "Pitch Up", "pitch_shift_down": "Pitch Down",
                 "lofi": "Lo-Fi", "lofi_filter": "Lo-Fi"}
        return f"{names.get(aug_type, aug_type.replace('_', ' ').title())}\n{level}"

    data = by_noise_snr_genre.copy()
    data["condition"] = [condition_label(r.degradation_type, r.degradation_value)
                         for r in data.itertuples(index=False)]
    condition_order = (data[["degradation_type", "degradation_value", "condition"]]
                       .drop_duplicates().sort_values(["degradation_type", "degradation_value"]))
    conditions = condition_order["condition"].tolist()
    genres = sorted(data["true_genre"].drop_duplicates())
    models = data["model"].drop_duplicates().tolist()

    fig, axes = plt.subplots(len(models), 1, figsize=(13, 5.5 * len(models)), squeeze=False)
    image = None
    for ax, model in zip(axes[:, 0], models):
        pivot = (data[data["model"] == model]
                 .pivot(index="true_genre", columns="condition", values="accuracy")
                 .reindex(index=genres, columns=conditions))
        image = ax.imshow(pivot.to_numpy(), vmin=0, vmax=1, cmap="RdYlGn")
        ax.set_aspect("auto")
        ax.set_title(model, fontsize=12, fontweight="bold")
        ax.set_yticks(range(len(genres)), labels=genres, fontsize=9)
        ax.set_xticks(range(len(conditions)), labels=conditions, rotation=40, ha="right", fontsize=8)
        ax.grid(visible=False)
        for row in range(pivot.shape[0]):
            for col in range(pivot.shape[1]):
                value = pivot.iat[row, col]
                color = "white" if (value < 0.3 or value > 0.75) else "black"
                ax.text(col, row, f"{value:.2f}", ha="center", va="center",
                        color=color, fontsize=8, fontweight="bold")

    fig.suptitle("Genre Accuracy by Noise Type and SNR", fontsize=14, fontweight="bold")
    if image is not None:
        fig.colorbar(image, ax=axes, fraction=0.015, pad=0.02, label="Accuracy")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_overall_by_aug_type(by_noise_snr: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib.ticker as mticker
    specs = [
        (by_noise_snr[by_noise_snr["degradation_type"].isin(NOISE_TYPES)],
         "SNR (dB)",
         "Overall Genre Accuracy by SNR — Noise Augmentations",
         "(averaged across crowd, street & white noise)",
         "overall_accuracy_by_noise_snr.png"),
        (by_noise_snr[by_noise_snr["degradation_type"].isin(TRANSFORM_TYPES)],
         "Severity Level",
         "Overall Genre Accuracy by Level — Transform Augmentations",
         "(averaged across pitch shift up, pitch shift down & lo-fi)",
         "overall_accuracy_by_transform_level.png"),
    ]
    for data, xlabel, title, subtitle, filename in specs:
        if data.empty:
            continue
        avg = (data.groupby(["model", "degradation_value"], as_index=False)
               .agg(accuracy=("accuracy", "mean")))
        models = avg["model"].unique().tolist()
        levels = sorted(avg["degradation_value"].unique())
        x = np.arange(len(levels))
        width = 0.75 / max(len(models), 1)

        fig, ax = plt.subplots(figsize=(8, 5))
        for i, (model, color) in enumerate(zip(models, MODEL_PALETTE)):
            vals = [avg[(avg["model"] == model) & (avg["degradation_value"] == lv)]["accuracy"].mean() * 100
                    for lv in levels]
            offset = (i - (len(models) - 1) / 2) * width
            bars = ax.bar(x + offset, vals, width, label=model, color=color, zorder=3)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.5,
                        f"{val:.0f}%", ha="center", va="bottom", fontsize=8)

        ax.set_title(f"{title}\n{subtitle}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 110)
        ax.set_xticks(x, labels=[str(lv) for lv in levels])
        ax.tick_params(axis="x", rotation=0)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))
        ax.legend(title="Embedding Model")
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_confusion(confusion: pd.DataFrame, output_path: Path) -> None:
    genres = sorted(set(confusion["true_genre"]) | set(confusion["predicted_genre"]))
    models = confusion["model"].drop_duplicates().tolist()

    fig, axes = plt.subplots(len(models), 1, figsize=(10, 7 * len(models)), squeeze=False)
    image = None
    for ax, model in zip(axes[:, 0], models):
        pivot = (confusion[confusion["model"] == model]
                 .pivot(index="true_genre", columns="predicted_genre", values="rate")
                 .reindex(index=genres, columns=genres).fillna(0))
        image = ax.imshow(pivot.to_numpy(), vmin=0, vmax=1, cmap="Blues")
        ax.set_aspect("auto")
        ax.set_title(model, fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted genre")
        ax.set_ylabel("Ground-truth GTZAN genre")
        ax.set_xticks(range(len(genres)), labels=genres, fontsize=9)
        ax.set_yticks(range(len(genres)), labels=genres, fontsize=9)
        ax.tick_params(axis="x", rotation=35)
        ax.grid(visible=False)
        for row in range(pivot.shape[0]):
            for col in range(pivot.shape[1]):
                value = pivot.iat[row, col]
                if value >= 0.03:
                    color = "white" if value >= 0.6 else "black"
                    ax.text(col, row, f"{value:.2f}", ha="center", va="center",
                            color=color, fontsize=9, fontweight="bold")

    fig.suptitle("Genre Classification Confusion Matrix", fontsize=14, fontweight="bold")
    if image is not None:
        fig.colorbar(image, ax=axes, fraction=0.02, pad=0.03, label="Rate within true genre")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def write_outputs(results: pd.DataFrame, output_dir: Path,
                  results_csv: Path | None, write_csv: bool) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = summarize(results)

    if write_csv:
        if results_csv is not None:
            results.to_csv(results_csv, index=False)
        tables["by_snr_genre"].to_csv(output_dir / "accuracy_by_snr_genre.csv", index=False)
        tables["by_noise_snr_genre"].to_csv(output_dir / "accuracy_by_noise_snr_genre.csv", index=False)
        tables["by_snr"].to_csv(output_dir / "accuracy_by_snr.csv", index=False)
        tables["by_noise_snr"].to_csv(output_dir / "accuracy_by_noise_snr.csv", index=False)
        tables["by_genre"].to_csv(output_dir / "accuracy_by_genre.csv", index=False)
        tables["confusion"].to_csv(output_dir / "confusion_matrix.csv", index=False)

    plot_per_genre_by_noise_snr(tables["by_noise_snr_genre"], output_dir)
    plot_per_genre_by_transform_level(tables["by_noise_snr_genre"], output_dir)
    plot_noise_snr_heatmap(tables["by_noise_snr_genre"], output_dir / "accuracy_by_noise_snr_genre.png")
    plot_overall_by_aug_type(tables["by_noise_snr"], output_dir)
    plot_confusion(tables["confusion"], output_dir / "confusion_matrices.png")
    return tables


def main() -> None:
    args = parse_args()
    roots = [root.expanduser().resolve() for root in args.embedding_root]
    labels = model_labels(roots, args.model_label)
    output_dir = args.output_dir
    results_csv = args.results_csv or output_dir / "classification_results.csv"

    frames = []
    for root, label_value in zip(roots, labels):
        print(f"Training clean-original genre classifier for {label_value}")
        print(f"  root: {root}")
        frame = evaluate_model(root, label_value, c_value=args.c, seed=args.seed)
        print(f"  noisy test files: {len(frame):,}")
        print(f"  accuracy: {frame['correct'].mean() * 100:.1f}%")
        frames.append(frame)

    results = pd.concat(frames, ignore_index=True)
    tables = write_outputs(results, output_dir, results_csv, write_csv=args.write_csv)

    print("\nOverall accuracy by SNR:")
    print(tables["by_snr"].pivot(index="degradation_value", columns="model", values="accuracy")
          .mul(100).round(1).to_string())
    if args.write_csv:
        print(f"\nWrote {results_csv.resolve()}")
    print(f"Wrote {output_dir.resolve()}")


if __name__ == "__main__":
    main()
