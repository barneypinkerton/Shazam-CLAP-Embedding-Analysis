# Shazam CLAP Embedding Analysis

This repository evaluates how well two CLAP embedding checkpoints preserve music identity and genre under noisy and transformed GTZAN audio augmentations, including additive noise (white, crowd, street at SNR 0/10/20 dB) and musical transforms (pitch shift ±1/2/3 semitones, lo-fi bandpass filter at three severity levels).

## What Is Evaluated

The scripts assume an embedding root with this layout:

```text
Data/
  genres_original/{genre}/{track_id}.npy
  genres_augmented/{noise_type}/{snr}dB/{genre}/{track_id}.npy        # additive noise
  genres_augmented/{transform_type}/{level}/{genre}/{track_id}.npy   # pitch shift / lo-fi
```

Ground truth is derived from the dataset paths:

- Genre classification ground truth is the GTZAN genre folder name.
- Exact-song retrieval ground truth is the clean original track with the same `track_id` filename stem as the noisy query.

## Results

The included plots are under `results/`:

- `results/genre_classification/`: train a classifier on clean original embeddings, then test augmented embeddings (noise and transforms) against the GTZAN folder genre.
- `results/exact_song_retrieval/`: query augmented embeddings (noise and transforms) against a clean original-track index and score whether the retrieved clean track has the same `track_id`.
- `results/data_overview/`: dataset count checks by genre, noise type, SNR, and transform severity.

Model labels used in the plots:

- `Base AudioSet embeddings`: embeddings from `630k-audioset-best.pt`
- `Fine-tuned Music embeddings`: embeddings from `music_audioset_epoch_15_esc_90.14.pt`

## Run

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Generate dataset overview plots:

```bash
python summarize_gtzan_data.py --data-root /path/to/embedding-root
```

Run genre classification evaluation:

```bash
python evaluate_gtzan_retrieval.py \
  --embedding-root /path/to/base-embeddings \
  --embedding-root /path/to/fine-tuned-embeddings \
  --model-label "Base AudioSet embeddings" \
  --model-label "Fine-tuned Music embeddings"
```

Run exact-song retrieval evaluation:

```bash
python evaluate_gtzan_exact_retrieval.py \
  --embedding-root /path/to/base-embeddings \
  --embedding-root /path/to/fine-tuned-embeddings \
  --model-label "Base AudioSet embeddings" \
  --model-label "Fine-tuned Music embeddings"
```

Add `--write-csv` to any script to write the underlying summary tables.
