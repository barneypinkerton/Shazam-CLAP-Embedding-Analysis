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

from eval_utils import (
    NOISE_TYPES, TRANSFORM_TYPES,
    safe_label, model_labels,
    list_data, load_matrix, per_genre_bar_chart,
    apply_style, SNR_PALETTE, LEVEL_PALETTE, MODEL_PALETTE,
)

apply_style()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate exact-track retrieval: noisy GTZAN embeddings query a clean "
            "GTZAN original embedding index."
        )
    )
    parser.add_argument("--embedding-root", action="append", type=Path, required=True)
    parser.add_argument("--model-label", action="append")
    parser.add_argument("--output-dir", type=Path, default=Path("results/exact_song_retrieval"))
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--write-csv", action="store_true",
                        help="Write CSV tables in addition to graph images.")
    return parser.parse_args()


def evaluate_model(root: Path, model: str, batch_size: int) -> pd.DataFrame:
    index_items, query_items = list_data(root)
    index_matrix = load_matrix(index_items)
    query_matrix = load_matrix(query_items)

    index_track_ids = np.array([item.track_id for item in index_items])
    index_genres = np.array([item.genre for item in index_items])
    index_paths = np.array([item.relative_path for item in index_items])

    rows = []
    for start in range(0, len(query_items), batch_size):
        end = min(start + batch_size, len(query_items))
        similarities = query_matrix[start:end] @ index_matrix.T
        top_k = min(5, similarities.shape[1])
        top_indices = np.argpartition(-similarities, kth=top_k - 1, axis=1)[:, :top_k]
        top_scores = np.take_along_axis(similarities, top_indices, axis=1)
        order = np.argsort(-top_scores, axis=1)
        top_indices = np.take_along_axis(top_indices, order, axis=1)
        top_scores = np.take_along_axis(top_scores, order, axis=1)

        for offset, query in enumerate(query_items[start:end]):
            candidate_indices = top_indices[offset]
            candidate_scores = top_scores[offset]
            top1_index = int(candidate_indices[0])
            top1_track = str(index_track_ids[top1_index])
            top1_genre = str(index_genres[top1_index])

            rows.append({
                "model": model,
                "track_id": query.track_id,
                "degradation_type": query.degradation_type,
                "degradation_value": query.degradation_value,
                "true_genre": query.genre,
                "matched_track_id": top1_track,
                "predicted_genre": top1_genre,
                "top_1_correct": int(top1_track == query.track_id),
                "top_5_correct": int(query.track_id in set(index_track_ids[candidate_indices])),
                "genre_correct": int(top1_genre == query.genre),
                "cosine_similarity": float(candidate_scores[0]),
                "query_path": query.relative_path,
                "matched_path": str(index_paths[top1_index]),
            })

    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> dict[str, pd.DataFrame]:
    results = results.copy()
    results["wrong_song_right_genre"] = (
        (results["top_1_correct"] == 0) & (results["genre_correct"] == 1)
    ).astype(int)

    metrics = dict(
        num_queries=("track_id", "count"),
        top_1_accuracy=("top_1_correct", "mean"),
        top_5_accuracy=("top_5_correct", "mean"),
        retrieved_genre_accuracy=("genre_correct", "mean"),
        wrong_song_right_genre_rate=("wrong_song_right_genre", "mean"),
        mean_cosine_similarity=("cosine_similarity", "mean"),
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

    return {
        "by_snr_genre": by_snr_genre,
        "by_noise_snr_genre": by_noise_snr_genre,
        "by_snr": by_snr,
        "by_noise_snr": by_noise_snr,
    }


def plot_per_genre_by_noise_snr(by_noise_snr_genre: pd.DataFrame, output_dir: Path) -> None:
    noise_data = by_noise_snr_genre[by_noise_snr_genre["degradation_type"].isin(NOISE_TYPES)]
    for model in noise_data["model"].drop_duplicates():
        avg = (noise_data[noise_data["model"] == model]
               .groupby(["true_genre", "degradation_value"], as_index=False)
               .agg(top_1_accuracy=("top_1_accuracy", "mean")))
        per_genre_bar_chart(
            data=avg, metric="top_1_accuracy",
            level_order=[20, 10, 0],
            colors=SNR_PALETTE,
            legend_labels={20: "20 dB", 10: "10 dB", 0: "0 dB"},
            title=f"Exact Top-1 Retrieval Accuracy by Genre and SNR — Noise Augmentations\n{model}",
            xlabel="Top-1 same-track retrieval accuracy (%) — averaged across crowd, street & white noise",
            legend_title="SNR",
            output_path=output_dir / f"exact_top1_by_genre_noise_snr_{safe_label(model)}.png",
        )


def plot_per_genre_by_transform_level(by_noise_snr_genre: pd.DataFrame, output_dir: Path) -> None:
    transform_data = by_noise_snr_genre[by_noise_snr_genre["degradation_type"].isin(TRANSFORM_TYPES)]
    if transform_data.empty:
        return
    for model in transform_data["model"].drop_duplicates():
        avg = (transform_data[transform_data["model"] == model]
               .groupby(["true_genre", "degradation_value"], as_index=False)
               .agg(top_1_accuracy=("top_1_accuracy", "mean")))
        per_genre_bar_chart(
            data=avg, metric="top_1_accuracy",
            level_order=[1, 2, 3],
            colors=LEVEL_PALETTE,
            legend_labels={1: "Level 1", 2: "Level 2", 3: "Level 3"},
            title=f"Exact Top-1 Retrieval Accuracy by Genre and Level — Transform Augmentations\n{model}",
            xlabel="Top-1 same-track retrieval accuracy (%) — averaged across pitch shift up, pitch shift down & lo-fi",
            legend_title="Severity Level",
            output_path=output_dir / f"exact_top1_by_genre_transform_level_{safe_label(model)}.png",
        )


def plot_overall_by_aug_type(by_noise_snr: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib.ticker as mticker
    specs = [
        (by_noise_snr[by_noise_snr["degradation_type"].isin(NOISE_TYPES)],
         "SNR (dB)",
         "Overall Exact Top-1 Retrieval Accuracy by SNR — Noise Augmentations",
         "(averaged across crowd, street & white noise)",
         "overall_exact_top1_by_noise_snr.png"),
        (by_noise_snr[by_noise_snr["degradation_type"].isin(TRANSFORM_TYPES)],
         "Severity Level",
         "Overall Exact Top-1 Retrieval Accuracy by Level — Transform Augmentations",
         "(averaged across pitch shift up, pitch shift down & lo-fi)",
         "overall_exact_top1_by_transform_level.png"),
    ]
    for data, xlabel, title, subtitle, filename in specs:
        if data.empty:
            continue
        avg = (data.groupby(["model", "degradation_value"], as_index=False)
               .agg(top_1_accuracy=("top_1_accuracy", "mean")))
        models = avg["model"].unique().tolist()
        levels = sorted(avg["degradation_value"].unique())
        x = np.arange(len(levels))
        width = 0.75 / max(len(models), 1)

        fig, ax = plt.subplots(figsize=(8, 5))
        for i, (model, color) in enumerate(zip(models, MODEL_PALETTE)):
            vals = [avg[(avg["model"] == model) & (avg["degradation_value"] == lv)]["top_1_accuracy"].mean() * 100
                    for lv in levels]
            offset = (i - (len(models) - 1) / 2) * width
            bars = ax.bar(x + offset, vals, width, label=model, color=color, zorder=3)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.5,
                        f"{val:.0f}%", ha="center", va="bottom", fontsize=8)

        ax.set_title(f"{title}\n{subtitle}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Top-1 Accuracy (%)")
        ax.set_ylim(0, 110)
        ax.set_xticks(x, labels=[str(lv) for lv in levels])
        ax.tick_params(axis="x", rotation=0)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))
        ax.legend(title="Embedding Model")
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)


def write_outputs(results: pd.DataFrame, output_dir: Path, write_csv: bool) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = summarize(results)

    if write_csv:
        results.to_csv(output_dir / "retrieval_results.csv", index=False)
        tables["by_snr_genre"].to_csv(output_dir / "exact_retrieval_by_snr_genre.csv", index=False)
        tables["by_noise_snr_genre"].to_csv(output_dir / "exact_retrieval_by_noise_snr_genre.csv", index=False)
        tables["by_snr"].to_csv(output_dir / "exact_retrieval_by_snr.csv", index=False)
        tables["by_noise_snr"].to_csv(output_dir / "exact_retrieval_by_noise_snr.csv", index=False)

    plot_per_genre_by_noise_snr(tables["by_noise_snr_genre"], output_dir)
    plot_per_genre_by_transform_level(tables["by_noise_snr_genre"], output_dir)
    plot_overall_by_aug_type(tables["by_noise_snr"], output_dir)
    return tables


def main() -> None:
    args = parse_args()
    roots = [root.expanduser().resolve() for root in args.embedding_root]
    labels = model_labels(roots, args.model_label)

    frames = []
    for root, label_value in zip(roots, labels):
        print(f"Evaluating exact-song retrieval for {label_value}")
        frame = evaluate_model(root, label_value, batch_size=args.batch_size)
        print(f"  noisy queries: {len(frame):,}")
        print(f"  top-1 exact-song accuracy: {frame['top_1_correct'].mean() * 100:.1f}%")
        frames.append(frame)

    results = pd.concat(frames, ignore_index=True)
    tables = write_outputs(results, args.output_dir, write_csv=args.write_csv)

    print("\nExact-song Top-1 accuracy by SNR:")
    print(tables["by_snr"].pivot(index="degradation_value", columns="model", values="top_1_accuracy")
          .mul(100).round(1).to_string())
    print(f"\nWrote {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
