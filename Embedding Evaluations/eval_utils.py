"""Shared utilities for GTZAN embedding evaluation scripts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# Augmentation type sets used by both evaluation scripts.
NOISE_TYPES = {"crowd_noise", "street_noise", "white_noise"}
TRANSFORM_TYPES = {"pitch_shift_up", "pitch_shift_down", "lofi_filter", "lofi"}


@dataclass(frozen=True)
class EmbeddingItem:
    path: Path
    relative_path: str
    track_id: str
    genre: str
    degradation_type: str
    degradation_value: int


def safe_label(value: str) -> str:
    label = re.sub(r"\s+", "_", value.strip())
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label)
    return label.strip("_") or "model"


def checkpoint_label(root: Path) -> str | None:
    manifest_path = root / "embedding_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    checkpoint = manifest.get("checkpoint")
    return Path(checkpoint).stem if checkpoint else None


def model_labels(roots: list[Path], provided: list[str] | None) -> list[str]:
    if provided is not None and len(provided) != len(roots):
        raise ValueError("--model-label must be supplied once for each --embedding-root")
    labels = []
    for index, root in enumerate(roots):
        raw = provided[index] if provided is not None else checkpoint_label(root)
        labels.append((raw or root.name).strip() or root.name)
    if len(set(labels)) != len(labels):
        raise ValueError(f"Model labels must be unique, got {labels}")
    return labels


def numeric_level_value(name: str) -> int:
    """Parse a level folder name: '20dB' → 20, '1' → 1."""
    if name.lower().endswith("db"):
        try:
            return int(name[:-2])
        except ValueError:
            pass
    try:
        return int(name)
    except ValueError:
        raise ValueError(f"Expected a level folder (e.g. '20dB' or '1'), got {name!r}")


def original_item(path: Path, root: Path) -> EmbeddingItem:
    return EmbeddingItem(
        path=path,
        relative_path=path.relative_to(root).as_posix(),
        track_id=path.stem,
        genre=path.parent.name,
        degradation_type="clean",
        degradation_value=0,
    )


def augmented_item(path: Path, root: Path) -> EmbeddingItem:
    relative = path.relative_to(root)
    # Data/genres_augmented/{aug_type}/{level}/{genre}/{track}.npy
    parts = relative.parts
    if len(parts) < 6 or parts[1] != "genres_augmented":
        raise ValueError(f"Unexpected augmented embedding path: {relative.as_posix()}")
    return EmbeddingItem(
        path=path,
        relative_path=relative.as_posix(),
        track_id=path.stem,
        genre=path.parent.name,
        degradation_type=parts[2],
        degradation_value=numeric_level_value(parts[3]),
    )


def list_data(root: Path) -> tuple[list[EmbeddingItem], list[EmbeddingItem]]:
    original_root = root / "Data" / "genres_original"
    augmented_root = root / "Data" / "genres_augmented"

    if not original_root.exists():
        raise FileNotFoundError(f"Missing clean original embeddings: {original_root}")
    if not augmented_root.exists():
        raise FileNotFoundError(f"Missing augmented embeddings: {augmented_root}")

    originals = [original_item(p, root) for p in sorted(original_root.glob("*/*.npy"))]
    augmented = [augmented_item(p, root) for p in sorted(augmented_root.glob("*/*/*/*.npy"))]

    if not originals:
        raise FileNotFoundError(f"No clean original .npy files found under {original_root}")
    if not augmented:
        raise FileNotFoundError(f"No augmented .npy files found under {augmented_root}")

    return originals, augmented


def load_embedding(path: Path) -> np.ndarray:
    embedding = np.asarray(np.load(path), dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(embedding)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError(f"Embedding has invalid norm: {path}")
    return embedding / norm


def load_matrix(items: list[EmbeddingItem]) -> np.ndarray:
    return np.vstack([load_embedding(item.path) for item in items]).astype(np.float32)


def per_genre_bar_chart(
    data,
    metric: str,
    level_order: list[int],
    colors: dict[int, str],
    legend_labels: dict[int, str],
    title: str,
    xlabel: str,
    legend_title: str,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    pivot = (
        data.pivot(index="true_genre", columns="degradation_value", values=metric)
        .reindex(columns=level_order)
    )
    pivot["average"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("average", ascending=True).drop(columns="average")

    y = np.arange(len(pivot))
    bar_height = 0.24
    n = len(level_order)
    offsets = {lv: bar_height * (n // 2 - i) for i, lv in enumerate(level_order)}

    fig, ax = plt.subplots(figsize=(11, 7))
    for lv in level_order:
        values = pivot[lv].to_numpy()
        bars = ax.barh(y + offsets[lv], values * 100, height=bar_height,
                       color=colors[lv], label=legend_labels[lv])
        for bar, value in zip(bars, values):
            ax.text(min(value * 100 + 0.8, 101), bar.get_y() + bar.get_height() / 2,
                    f"{value * 100:.0f}%", va="center", ha="left", fontsize=9)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Ground-truth GTZAN genre")
    ax.set_yticks(y, labels=pivot.index)
    ax.set_xlim(0, 105)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(title=legend_title, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
