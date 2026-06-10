"""Regenerate all Shazam evaluation plots from shazam_eval_combined.csv.

Run from the repo root:
    python Shazam/evaluation/generate_plots.py

Outputs to Shazam/evaluation/results/plots/.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent
CSV  = HERE / "results" / "shazam_eval_combined.csv"
OUT  = HERE / "results" / "plots"
OUT.mkdir(parents=True, exist_ok=True)

# ── Consistent style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",           # horizontal gridlines only
    "grid.color": "#e0e0e0",
    "grid.linewidth": 0.8,
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "legend.framealpha": 0.92,
    "legend.edgecolor": "#cccccc",
})

# ── Palette ───────────────────────────────────────────────────────────────────
C_CORRECT = "#27ae60"
C_FP      = "#e74c3c"
C_REJECT  = "#95a5a6"

# Per aug-type colours used in line / bar charts
AUG_COLORS = {
    "crowd_noise":      "#2980b9",
    "street_noise":     "#8e44ad",
    "white_noise":      "#16a085",
    "lofi_filter":      "#e67e22",
    "pitch_shift_up":   "#c0392b",
    "pitch_shift_down": "#7f1f1f",
}

AUG_LABELS = {
    "crowd_noise":      "Crowd Noise",
    "street_noise":     "Street Noise",
    "white_noise":      "White Noise",
    "lofi_filter":      "Lo-Fi Filter",
    "pitch_shift_up":   "Pitch Up",
    "pitch_shift_down": "Pitch Down",
}
AUG_ORDER = ["crowd_noise", "street_noise", "white_noise",
             "lofi_filter", "pitch_shift_up", "pitch_shift_down"]
NOISE_TYPES = {"crowd_noise", "street_noise", "white_noise"}

OUTCOME_COLORS = {"correct": C_CORRECT, "false_positive": C_FP, "reject": C_REJECT}
OUTCOME_LABELS = {"correct": "Correct", "false_positive": "False Positive", "reject": "Reject"}


def _save(fig, name: str) -> None:
    path = OUT / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.name}")


def _bar_label(ax, bars, fmt="{:.1f}%", pad=0.4, fontsize=9):
    for b in bars:
        v = b.get_height()
        if v >= 0.3:
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + pad,
                fmt.format(v),
                ha="center", va="bottom", fontsize=fontsize,
            )


# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data …")
df = pd.read_csv(CSV)
df["db_severity"] = df["db_severity"].astype(int)


def _outcome(row):
    if row["identified"] == "no":
        return "reject"
    return "correct" if row["correct"] == "yes" else "false_positive"


df["outcome"] = df.apply(_outcome, axis=1)
df["aug_category"] = df["aug_type"].apply(
    lambda x: "noise" if x in NOISE_TYPES else "transform"
)
df["aug_label"] = df["aug_type"].map(AUG_LABELS).fillna(df["aug_type"])
df["top_match_genre"] = (
    df["top_match_name"].fillna("").str.split(".").str[0].replace("", np.nan)
)

# Aggregated per (aug_type, db_severity)
agg = (
    df.groupby(["aug_type", "aug_label", "aug_category", "db_severity"])
    .apply(
        lambda g: pd.Series({
            "accuracy":       (g["outcome"] == "correct").mean() * 100,
            "false_positive": (g["outcome"] == "false_positive").mean() * 100,
            "reject":         (g["outcome"] == "reject").mean() * 100,
            "n": len(g),
        }),
        include_groups=False,
    )
    .reset_index()
)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 01 — Overall outcome breakdown
# ─────────────────────────────────────────────────────────────────────────────
print("Plot 01 …")
overall = (
    df["outcome"].value_counts(normalize=True)
    .reindex(["correct", "false_positive", "reject"]).fillna(0) * 100
)

fig, ax = plt.subplots(figsize=(7, 4.5))
bars = ax.bar(
    [OUTCOME_LABELS[k] for k in overall.index],
    overall.values,
    color=[OUTCOME_COLORS[k] for k in overall.index],
    width=0.55, zorder=3,
)
_bar_label(ax, bars, pad=0.5)
ax.set_ylabel("Share of all queries (%)")
ax.set_title("Overall Outcome Breakdown")
ax.set_ylim(0, max(overall.values) * 1.18)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))
fig.tight_layout()
_save(fig, "01_overall_outcome_breakdown.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 02 — Outcome by category (noise vs transform)
# ─────────────────────────────────────────────────────────────────────────────
print("Plot 02 …")
outcomes = ["correct", "false_positive", "reject"]
by_cat = (
    df.groupby("aug_category")["outcome"]
    .value_counts(normalize=True)
    .unstack(fill_value=0)
    .reindex(columns=outcomes, fill_value=0) * 100
)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
titles = {
    "noise":     "Additive Noise  (SNR 0 / 10 / 20 dB)",
    "transform": "Musical Transforms  (Pitch / Lo-Fi)",
}
for ax, cat in zip(axes, ["noise", "transform"]):
    vals = by_cat.loc[cat] if cat in by_cat.index else pd.Series(0, index=outcomes)
    bars = ax.bar(
        [OUTCOME_LABELS[o] for o in outcomes],
        vals.values,
        color=[OUTCOME_COLORS[o] for o in outcomes],
        width=0.55, zorder=3,
    )
    _bar_label(ax, bars, pad=0.5)
    ax.set_title(titles[cat])
    ax.set_ylabel("Share of queries (%)")
    ax.set_ylim(0, 110)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))

fig.suptitle("Outcome Breakdown by Augmentation Category", fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
_save(fig, "02_outcome_by_category.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 03 — Accuracy vs condition severity  (line chart, 2 panels)
# ─────────────────────────────────────────────────────────────────────────────
print("Plot 03 …")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: noise (x = SNR, displayed low→high so left=most noise)
noise_agg = agg[agg["aug_category"] == "noise"].copy()
noise_snr_vals = sorted(noise_agg["db_severity"].unique())  # [0, 10, 20]
for at in ["crowd_noise", "street_noise", "white_noise"]:
    sub = noise_agg[noise_agg["aug_type"] == at].sort_values("db_severity")
    axes[0].plot(
        sub["db_severity"], sub["accuracy"],
        marker="o", linewidth=2.5, markersize=8,
        color=AUG_COLORS[at], label=AUG_LABELS[at],
    )
axes[0].set_title("Additive Noise — Accuracy vs SNR")
axes[0].set_xlabel("SNR (dB)  —  right = cleaner signal")
axes[0].set_ylabel("Top-1 accuracy (%)")
axes[0].set_xticks(noise_snr_vals)
axes[0].set_xticklabels([f"{v} dB" for v in noise_snr_vals])
axes[0].set_ylim(0, 105)
axes[0].legend(loc="lower right")
axes[0].yaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))

# Right: transforms (x = severity level 1–3)
tx_agg = agg[agg["aug_category"] == "transform"].copy()
tx_levels = sorted(tx_agg["db_severity"].unique())  # [1, 2, 3]
for at in ["lofi_filter", "pitch_shift_up", "pitch_shift_down"]:
    sub = tx_agg[tx_agg["aug_type"] == at].sort_values("db_severity")
    axes[1].plot(
        sub["db_severity"], sub["accuracy"],
        marker="o", linewidth=2.5, markersize=8,
        color=AUG_COLORS[at], label=AUG_LABELS[at],
    )
axes[1].set_title("Musical Transforms — Accuracy vs Severity")
axes[1].set_xlabel("Severity level  —  right = more distortion")
axes[1].set_ylabel("Top-1 accuracy (%)")
axes[1].set_xticks(tx_levels)
axes[1].set_xticklabels([f"L{v}" for v in tx_levels])
axes[1].set_ylim(-2, 105)
axes[1].legend(loc="upper right")
axes[1].yaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))

fig.suptitle("Top-1 Accuracy vs Degradation Severity", fontsize=14, fontweight="bold")
fig.tight_layout()
_save(fig, "03_accuracy_vs_severity.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 04 — Stacked outcome composition per aug type
#           Redesigned as grouped bars (correct/FP/reject) per severity level
# ─────────────────────────────────────────────────────────────────────────────
print("Plot 04 …")
fig, axes = plt.subplots(1, 6, figsize=(18, 5), sharey=True)

for ax, at in zip(axes, AUG_ORDER):
    cat = "noise" if at in NOISE_TYPES else "transform"
    sub = agg[agg["aug_type"] == at].sort_values("db_severity")
    levels = sub["db_severity"].tolist()
    x = np.arange(len(levels))

    # Stacked bars: correct (bottom), false_positive (middle), reject (top)
    bottoms = np.zeros(len(levels))
    for outcome_key, color in [
        ("accuracy", C_CORRECT),
        ("false_positive", C_FP),
        ("reject", C_REJECT),
    ]:
        vals = sub[outcome_key].values
        bars = ax.bar(x, vals, bottom=bottoms, color=color, width=0.6, zorder=3)
        # Label each segment if large enough to read
        for bar, bot, val in zip(bars, bottoms, vals):
            if val >= 5:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bot + val / 2,
                    f"{val:.0f}%",
                    ha="center", va="center", fontsize=8,
                    color="white", fontweight="bold",
                )
        bottoms = bottoms + vals

    x_labels = [f"{lv} dB" for lv in levels] if cat == "noise" else [f"L{lv}" for lv in levels]
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_title(AUG_LABELS[at], fontsize=11, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.tick_params(axis="x", length=0)

axes[0].set_ylabel("Share of queries (%)")
axes[0].yaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))

# Shared legend
handles = [
    plt.Rectangle((0, 0), 1, 1, fc=C_CORRECT, label="Correct"),
    plt.Rectangle((0, 0), 1, 1, fc=C_FP,      label="False Positive"),
    plt.Rectangle((0, 0), 1, 1, fc=C_REJECT,   label="Reject"),
]
fig.legend(handles=handles, loc="lower center", ncol=3,
           bbox_to_anchor=(0.5, -0.04), framealpha=0.92, edgecolor="#cccccc")
fig.suptitle("Outcome Composition per Augmentation Type", fontsize=14, fontweight="bold")
fig.tight_layout()
_save(fig, "04_outcome_composition_per_aug_type.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 05 — Per-genre accuracy (averaged across all conditions)
# ─────────────────────────────────────────────────────────────────────────────
print("Plot 05 …")
genre_acc = (
    df.groupby("genre")["outcome"]
    .apply(lambda s: (s == "correct").mean() * 100)
    .sort_values()
)

fig, ax = plt.subplots(figsize=(9, 5))
colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, len(genre_acc)))
bars = ax.barh(genre_acc.index, genre_acc.values, color=colors, height=0.6, zorder=3)
for b, v in zip(bars, genre_acc.values):
    ax.text(v + 0.4, b.get_y() + b.get_height() / 2,
            f"{v:.1f}%", va="center", fontsize=9)
ax.set_xlabel("Top-1 accuracy (%)  —  averaged across all conditions")
ax.set_title("Per-Genre Accuracy")
ax.set_xlim(0, max(genre_acc.values) * 1.12)
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))
ax.grid(axis="x")
ax.grid(axis="y", visible=False)
fig.tight_layout()
_save(fig, "05_per_genre_accuracy.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 06 — Per-genre accuracy by severity (2 panels: noise, transforms)
# ─────────────────────────────────────────────────────────────────────────────
print("Plot 06 …")
genre_order = (
    df.groupby("genre")["outcome"]
    .apply(lambda s: (s == "correct").mean())
    .sort_values().index
)

fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
snr_palette   = {0: C_FP,      10: "#f39c12", 20: C_CORRECT}
level_palette = {1: C_CORRECT, 2:  "#f39c12",  3: C_FP}

for ax, cat, title, palette in [
    (axes[0], "noise",     "Additive Noise  (by SNR)",        snr_palette),
    (axes[1], "transform", "Musical Transforms  (by severity)", level_palette),
]:
    sub = df[df["aug_category"] == cat]
    lvls = sorted(sub["db_severity"].unique())
    pivot = (
        sub.groupby(["genre", "db_severity"])["outcome"]
        .apply(lambda s: (s == "correct").mean() * 100)
        .unstack("db_severity")
        .reindex(genre_order)
    )
    y = np.arange(len(genre_order))
    bh = 0.25
    for i, lv in enumerate(lvls):
        offset = (i - 1) * bh
        label = f"{lv} dB" if cat == "noise" else f"Level {lv}"
        bars = ax.barh(y + offset, pivot[lv].values, bh,
                       color=palette[lv], label=label, zorder=3)
        for b, v in zip(bars, pivot[lv].values):
            if not np.isnan(v) and v >= 3:
                ax.text(v + 0.5, b.get_y() + b.get_height() / 2,
                        f"{v:.0f}%", va="center", fontsize=7)
    ax.set_yticks(y)
    ax.set_yticklabels(genre_order)
    ax.set_xlabel("Top-1 accuracy (%)")
    ax.set_title(title)
    ax.set_xlim(0, 115)
    ax.legend(loc="lower right", title="Severity")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)

fig.suptitle("Per-Genre Accuracy by Degradation Severity", fontsize=14, fontweight="bold")
fig.tight_layout()
_save(fig, "06_per_genre_accuracy_by_severity.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 07 — Genre × condition accuracy heatmaps
# ─────────────────────────────────────────────────────────────────────────────
print("Plot 07 …")
fig, axes = plt.subplots(1, 6, figsize=(22, 6), sharey=True,
                         gridspec_kw={"wspace": 0.08})

for ax, at in zip(axes, AUG_ORDER):
    cat = "noise" if at in NOISE_TYPES else "transform"
    sub = df[df["aug_type"] == at]
    col_order = (
        sorted(sub["db_severity"].unique(), reverse=True)
        if cat == "noise" else
        sorted(sub["db_severity"].unique())
    )
    pivot = sub.pivot_table(
        index="genre", columns="db_severity",
        values="outcome",
        aggfunc=lambda s: (s == "correct").mean() * 100,
    ).reindex(columns=col_order)

    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)
    col_labels = (
        [f"{c} dB" for c in col_order] if cat == "noise"
        else [f"L{c}" for c in col_order]
    )
    ax.set_xticks(np.arange(len(col_order)))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_title(AUG_LABELS[at], fontsize=11, fontweight="bold")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not np.isnan(v):
                color = "white" if (v < 25 or v > 75) else "black"
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        color=color, fontsize=9, fontweight="bold")
    ax.grid(visible=False)

fig.suptitle("Top-1 Accuracy by Genre and Condition", fontsize=14, fontweight="bold", y=1.01)
fig.subplots_adjust(left=0.07, right=0.90, top=0.90, bottom=0.06)
cbar_ax = fig.add_axes([0.92, 0.10, 0.012, 0.75])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label("Accuracy (%)", fontsize=10)
_save(fig, "07_accuracy_heatmap_by_genre.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 08 — Identification latency
# ─────────────────────────────────────────────────────────────────────────────
print("Plot 08 …")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: overall histogram
axes[0].hist(df["elapsed_s"], bins=50, color="#2980b9", edgecolor="white", zorder=3)
med = df["elapsed_s"].median()
axes[0].axvline(med, color=C_FP, linestyle="--", linewidth=1.8, label=f"Median  {med:.2f} s")
axes[0].set_xlabel("Elapsed time (s)")
axes[0].set_ylabel("Query count")
axes[0].set_title("Identification Latency — Overall")
axes[0].legend()

# Right: horizontal boxplot per condition (easier to read than rotated vertical)
conditions = [
    (at, lv)
    for at in AUG_ORDER
    for lv in sorted(df[df["aug_type"] == at]["db_severity"].unique())
]
cat_of = {at: ("noise" if at in NOISE_TYPES else "transform") for at in AUG_ORDER}
labels = [
    f"{AUG_LABELS[at]}  {'%d dB' % lv if cat_of[at]=='noise' else 'L%d' % lv}"
    for at, lv in conditions
]
data = [
    df[(df["aug_type"] == at) & (df["db_severity"] == lv)]["elapsed_s"].values
    for at, lv in conditions
]
colors_bp = [
    AUG_COLORS[at] for at, _ in conditions
]

bplot = axes[1].boxplot(data, vert=False, showfliers=False, patch_artist=True, tick_labels=labels)
for patch, col in zip(bplot["boxes"], colors_bp):
    patch.set_facecolor(col)
    patch.set_alpha(0.7)
for median in bplot["medians"]:
    median.set_color("black")
    median.set_linewidth(1.5)

axes[1].set_xlabel("Elapsed time (s)")
axes[1].set_title("Latency per Condition  (outliers hidden)")
axes[1].tick_params(axis="y", labelsize=9)
axes[1].grid(axis="x")
axes[1].grid(axis="y", visible=False)

fig.tight_layout()
_save(fig, "08_latency.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 09 — Latency by outcome
# ─────────────────────────────────────────────────────────────────────────────
print("Plot 09 …")
fig, ax = plt.subplots(figsize=(7, 4.5))
groups  = ["correct", "false_positive", "reject"]
data    = [df[df["outcome"] == g]["elapsed_s"].values for g in groups]
colors9 = [OUTCOME_COLORS[g] for g in groups]

bplot = ax.boxplot(data, tick_labels=[OUTCOME_LABELS[g] for g in groups],
                   showfliers=False, patch_artist=True)
for patch, col in zip(bplot["boxes"], colors9):
    patch.set_facecolor(col)
    patch.set_alpha(0.75)
for median in bplot["medians"]:
    median.set_color("black")
    median.set_linewidth(1.8)

ax.set_ylabel("Elapsed time (s)")
ax.set_title("Identification Latency by Outcome")
fig.tight_layout()
_save(fig, "09_latency_by_outcome.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 10 — Score / confidence distributions
# ─────────────────────────────────────────────────────────────────────────────
print("Plot 10 …")
ided = df[df["identified"] == "yes"].copy()
ided["score"]      = pd.to_numeric(ided["score"],      errors="coerce")
ided["confidence"] = pd.to_numeric(ided["confidence"], errors="coerce")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, col, title in [
    (axes[0], "score",      "Match Score Distribution"),
    (axes[1], "confidence", "Confidence Distribution"),
]:
    for outcome_key, color, label in [
        ("correct",        C_CORRECT, "Correct"),
        ("false_positive", C_FP,      "False Positive"),
    ]:
        vals = ided[ided["outcome"] == outcome_key][col].dropna()
        if len(vals):
            ax.hist(vals, bins=40, alpha=0.65, color=color, label=label, zorder=3)
    ax.set_xlabel(col.replace("_", " ").title())
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()

fig.suptitle("Score and Confidence — Correct vs False Positive  (identified queries only)",
             fontsize=12, fontweight="bold")
fig.tight_layout()
_save(fig, "10_score_confidence_histograms.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 11 — False-positive genre confusion
# ─────────────────────────────────────────────────────────────────────────────
print("Plot 11 …")
wrong = df[df["outcome"] == "false_positive"].copy()
all_genres = sorted(df["genre"].unique())

if len(wrong) == 0:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.text(0.5, 0.5, "No false positives", ha="center", va="center", transform=ax.transAxes)
else:
    confusion = (
        pd.crosstab(wrong["genre"], wrong["top_match_genre"], normalize="index") * 100
    ).reindex(index=all_genres, columns=all_genres, fill_value=0)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(confusion.values, cmap="Reds", vmin=0, vmax=100)
    ax.set_xticks(np.arange(len(all_genres)))
    ax.set_xticklabels(all_genres, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(all_genres)))
    ax.set_yticklabels(all_genres, fontsize=9)
    ax.set_xlabel("Wrongly-picked genre")
    ax.set_ylabel("Ground-truth genre")
    ax.set_title("False-Positive Genre Confusion\n(% of wrong predictions per row)")
    for i in range(len(all_genres)):
        for j in range(len(all_genres)):
            v = confusion.values[i, j]
            if v >= 1:
                color = "white" if v > 60 else "black"
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        fontsize=9, color=color, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="%")
    ax.grid(visible=False)

fig.tight_layout()
_save(fig, "11_false_positive_confusion.png")

print("\nAll plots saved to", OUT)
