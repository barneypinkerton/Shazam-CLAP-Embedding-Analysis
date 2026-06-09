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
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize GTZAN embedding data layout.")
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Embedding root containing Data/genres_original and Data/genres_augmented.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/data_overview"),
        help="Directory for overview CSVs and plots.",
    )
    parser.add_argument(
        "--write-csv",
        action="store_true",
        help="Write CSV tables in addition to graph images.",
    )
    return parser.parse_args()


def original_rows(root: Path) -> list[dict[str, object]]:
    rows = []
    for path in sorted((root / "Data" / "genres_original").glob("*/*.npy")):
        rows.append(
            {
                "split": "original",
                "genre": path.parent.name,
                "track_id": path.stem,
                "degradation_type": "clean",
                "degradation_value": 0,
                "path": path.relative_to(root).as_posix(),
            }
        )
    return rows


_NOISE_TYPES = {"crowd_noise", "street_noise", "white_noise"}
_TRANSFORM_TYPES = {"pitch_shift_up", "pitch_shift_down", "lofi_filter", "lofi"}


def augmented_rows(root: Path) -> list[dict[str, object]]:
    rows = []
    aug_root = root / "Data" / "genres_augmented"

    # Additive noise: structure is {noise_type}/{snr}dB/{genre}/*.npy
    for path in sorted(aug_root.glob("*/*dB/*/*.npy")):
        relative = path.relative_to(root)
        _, _, degradation_type, value_name, genre, _ = relative.parts
        rows.append(
            {
                "split": "augmented",
                "aug_category": "noise",
                "genre": genre,
                "track_id": path.stem,
                "degradation_type": degradation_type,
                "degradation_value": int(value_name.removesuffix("dB")),
                "path": relative.as_posix(),
            }
        )

    # Musical transforms: structure is {transform_type}/{level}/{genre}/*.npy
    for path in sorted(aug_root.glob("*/*/*/*.npy")):
        relative = path.relative_to(root)
        _, _, degradation_type, value_name, genre, _ = relative.parts
        if degradation_type not in _TRANSFORM_TYPES:
            continue
        rows.append(
            {
                "split": "augmented",
                "aug_category": "transform",
                "genre": genre,
                "track_id": path.stem,
                "degradation_type": degradation_type,
                "degradation_value": int(value_name),
                "path": relative.as_posix(),
            }
        )

    return rows


def plot_bar(series: pd.Series, output_path: Path, title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    series.plot(kind="bar", ax=ax, color="#386cb0")
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


AUG_TYPE_ORDER = [
    "crowd_noise", "street_noise", "white_noise",
    "lofi_filter", "pitch_shift_up", "pitch_shift_down",
]
AUG_LABEL = {
    "crowd_noise":      "Crowd Noise",
    "street_noise":     "Street Noise",
    "white_noise":      "White Noise",
    "lofi_filter":      "Lo-Fi Filter",
    "pitch_shift_up":   "Pitch Up",
    "pitch_shift_down": "Pitch Down",
}


def _heatmap_panel(ax, pivot, col_labels, title):
    image = ax.imshow(pivot.to_numpy(), cmap="Blues")
    ax.set_xticks(range(len(pivot.columns)), labels=col_labels)
    ax.set_yticks(
        range(len(pivot.index)),
        labels=[AUG_LABEL.get(n, n.replace("_", " ").title()) for n in pivot.index],
    )
    ax.set_title(title)
    for row in range(pivot.shape[0]):
        for col in range(pivot.shape[1]):
            ax.text(col, row, int(pivot.iat[row, col]), ha="center", va="center", color="black")
    return image


def plot_noise_grid(counts: pd.DataFrame, output_path: Path) -> None:
    noise_counts = counts[counts["degradation_type"].isin(_NOISE_TYPES)]
    tx_counts = counts[counts["degradation_type"].isin(_TRANSFORM_TYPES)]

    noise_pivot = noise_counts.pivot_table(
        index="degradation_type", columns="degradation_value",
        values="count", aggfunc="sum", fill_value=0,
    ).reindex([t for t in AUG_TYPE_ORDER if t in _NOISE_TYPES]).sort_index(axis=1)

    tx_pivot = tx_counts.pivot_table(
        index="degradation_type", columns="degradation_value",
        values="count", aggfunc="sum", fill_value=0,
    ).reindex([t for t in AUG_TYPE_ORDER if t in _TRANSFORM_TYPES]).sort_index(axis=1)

    fig, (ax_noise, ax_tx) = plt.subplots(1, 2, figsize=(13, 4))

    im1 = _heatmap_panel(
        ax_noise, noise_pivot,
        [f"{v} dB" for v in noise_pivot.columns],
        "Additive Noise — Files by Type and SNR",
    )
    im2 = _heatmap_panel(
        ax_tx, tx_pivot,
        [f"Level {v}" for v in tx_pivot.columns],
        "Transforms — Files by Type and Severity",
    )

    fig.colorbar(im1, ax=ax_noise, label="Files")
    fig.colorbar(im2, ax=ax_tx, label="Files")
    fig.suptitle("Augmented Files By Condition", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_genre_noise_counts(counts: pd.DataFrame, output_path: Path) -> None:
    pivot = counts.pivot_table(
        index="genre",
        columns="degradation_type",
        values="count",
        aggfunc="sum",
        fill_value=0,
    ).sort_index()

    # Reorder columns: noise types first, transforms second.
    ordered_cols = [t for t in AUG_TYPE_ORDER if t in pivot.columns]
    pivot = pivot[ordered_cols]
    pivot.columns = [AUG_LABEL.get(c, c) for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(13, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Augmented Files Per Genre")
    ax.set_xlabel("")
    ax.set_ylabel("Files")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(title="Augmentation Type")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = args.data_root.expanduser().resolve()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = original_rows(root) + augmented_rows(root)
    data = pd.DataFrame(rows)
    split_counts = data.groupby("split", as_index=False).size().rename(columns={"size": "count"})
    original_genre_counts = (
        data[data["split"] == "original"].groupby("genre").size().rename("count").sort_index()
    )
    augmented_counts = (
        data[data["split"] == "augmented"]
        .groupby(["degradation_type", "degradation_value", "genre"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )

    if args.write_csv:
        data.to_csv(output_dir / "all_files.csv", index=False)
        split_counts.to_csv(output_dir / "split_counts.csv", index=False)
        original_genre_counts.to_csv(output_dir / "original_genre_counts.csv")
        augmented_counts.to_csv(output_dir / "augmented_counts_by_noise_db_genre.csv", index=False)

    plot_bar(
        original_genre_counts,
        output_dir / "original_genre_counts.png",
        "Clean Original Files Per Genre",
        "Files",
    )
    plot_noise_grid(augmented_counts, output_dir / "augmented_noise_db_counts.png")
    plot_genre_noise_counts(augmented_counts, output_dir / "augmented_genre_noise_counts.png")

    print("Genres:")
    print(", ".join(original_genre_counts.index.tolist()))
    print("\nSplit counts:")
    print(split_counts.to_string(index=False))
    print(f"\nWrote {output_dir.resolve()}")


if __name__ == "__main__":
    main()
