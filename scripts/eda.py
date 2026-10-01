"""
Phase 2 -- Exploratory Data Analysis (EDA)

Looks at the data BEFORE any model is trained, to answer:
  1. How many images / samples per class?        -> class_counts.png
  2. What do Positive and Negative images look like,
     and what do the HARD cases look like?         -> sample_grid.png
  3. How different is brightness between classes? -> brightness_hist.png
  4. Is there an easy "shortcut" in the files themselves
     (image size, file format) that differs by class? -> printed tables

Uses only manifest.csv + the image pixels (no spreadsheet metadata).
Outputs go to outputs/eda/.  Run:  python scripts/eda.py
"""
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use("Agg")                     # draw to files, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score

import config

OUT = config.OUTPUTS_DIR / "eda"
CLASS_NAME = {0: "Negative", 1: "Positive"}
COLOR = {0: "#2a78d6", 1: "#eb6834"}      # validated categorical slots 1 & 2
INK, INK_2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def image_stats(raw_file):
    """Open one raw image and measure its brightness.

    HEp-2 fluorescence is GREEN (FITC dye), so the green channel carries the
    signal. We measure on a 512-px copy: same statistics, much faster.
    """
    path = config.AIDA_RAW_DIR / raw_file
    with Image.open(path) as im:
        width, height = im.size
        im.draft("RGB", (1024, 1024))      # JPEGs decode at reduced size (speed)
        im = im.convert("RGB")
        im.thumbnail((683, 512))
        green = np.asarray(im)[:, :, 1].astype(np.float32)
    return {
        "width": width, "height": height,
        "format": path.suffix.lower().lstrip("."),
        "green_mean": green.mean(),                 # overall brightness
        "green_p99": np.percentile(green, 99),      # brightness of the brightest spots
    }


def style(ax, title):
    ax.set_title(title, loc="left", color=INK, fontsize=11, pad=10)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c8c3")
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="y", color="#e8e7e3", linewidth=0.8)
    ax.set_axisbelow(True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(config.MANIFEST_PATH)
    file_map = pd.read_csv(config.DATA_DIR / "aida_file_map.csv")
    df = manifest.merge(file_map, on="image_path")

    # ---------------------------------------------------------- measure every image
    print(f"Measuring {len(df)} images (takes a few minutes)...")
    with ProcessPoolExecutor() as pool:
        stats = list(pool.map(image_stats, df["raw_file"], chunksize=16))
    df = pd.concat([df, pd.DataFrame(stats)], axis=1)
    df.to_csv(OUT / "image_stats.csv", index=False)

    # ---------------------------------------------------------- 1. class counts
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), facecolor=SURFACE)
    samples = df.drop_duplicates("sample_id")
    for ax, (data, title) in zip(axes, [(df, "Images per class"), (samples, "Samples (wells) per class")]):
        counts = data["label"].value_counts().sort_index()
        bars = ax.bar([CLASS_NAME[k] for k in counts.index], counts.values,
                      color=[COLOR[k] for k in counts.index], width=0.55)
        ax.bar_label(bars, color=INK, fontsize=9, padding=3)
        style(ax, title)
        ax.set_ylim(0, counts.max() * 1.15)
    fig.tight_layout()
    fig.savefig(OUT / "class_counts.png", dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------- 2. sample grid
    rng = np.random.default_rng(config.SEED)
    neg, pos = df[df.label == 0], df[df.label == 1]
    rows = [
        ("Typical Negative", neg.iloc[rng.choice(len(neg), 6, replace=False)]),
        ("Typical Positive", pos.iloc[rng.choice(len(pos), 6, replace=False)]),
        ("Brightest Negatives (hard)", neg.nlargest(6, "green_mean")),
        ("Dimmest Positives (hard)", pos.nsmallest(6, "green_mean")),
    ]
    fig, axes = plt.subplots(4, 6, figsize=(13, 9.2), facecolor=SURFACE)
    for r, (title, subset) in enumerate(rows):
        for c, (_, row) in enumerate(subset.iterrows()):
            ax = axes[r, c]
            with Image.open(config.AIDA_RAW_DIR / row["raw_file"]) as im:
                im.draft("RGB", (600, 600))
                im = im.convert("RGB")
                im.thumbnail((300, 225))
                ax.imshow(im)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor(COLOR[row["label"]]); s.set_linewidth(2)
            ax.set_xlabel(f"{row['raw_file']}  |  mean {row['green_mean']:.0f}", fontsize=8, color=INK_2)
        axes[r, 0].set_ylabel(title, fontsize=10, color=INK, labelpad=8)
    fig.suptitle("AIDA images: frame colour = expert label (blue Negative, orange Positive); "
                 "'mean' = average green brightness 0-255", fontsize=10, color=INK_2, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "sample_grid.png", dpi=130)
    plt.close(fig)

    # ---------------------------------------------------------- 3. brightness histograms
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), facecolor=SURFACE)
    for ax, col, title in [(axes[0], "green_mean", "Average green brightness per image"),
                           (axes[1], "green_p99", "Brightest-spot level (99th percentile)")]:
        bins = np.linspace(0, 255, 52)
        for k in (0, 1):
            ax.hist(df.loc[df.label == k, col], bins=bins, density=True, histtype="step",
                    linewidth=2, color=COLOR[k], label=CLASS_NAME[k])
        auc = roc_auc_score(df["label"], df[col])
        style(ax, f"{title}\nAUC if used alone as a classifier: {auc:.3f}")
        ax.set_xlabel("green value (0 = black, 255 = full brightness)", color=INK_2, fontsize=9)
        ax.set_ylabel("share of images (density)", color=INK_2, fontsize=9)
        ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUT / "brightness_hist.png", dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------- 4. shortcut check (file-level)
    df["size"] = df["width"].astype(str) + "x" + df["height"].astype(str)
    size_tab = pd.crosstab(df["size"], df["label"].map(CLASS_NAME))
    size_tab["% Positive"] = (100 * size_tab["Positive"] / size_tab.sum(axis=1)).round(0)
    fmt_tab = pd.crosstab(df["format"], df["label"].map(CLASS_NAME))
    fmt_tab["% Positive"] = (100 * fmt_tab["Positive"] / fmt_tab.sum(axis=1)).round(0)

    # ---------------------------------------------------------- summary numbers
    thr_acc = max(((df["green_mean"] >= t) == df["label"]).mean() for t in range(0, 256))
    bal = []
    for t in range(0, 256):
        pred = df["green_mean"] >= t
        sens = pred[df.label == 1].mean(); spec = (~pred[df.label == 0]).mean()
        bal.append((sens + spec) / 2)
    report = [
        "Brightness summary (green_mean):",
        df.groupby(df["label"].map(CLASS_NAME))[["green_mean", "green_p99"]].describe().round(1).T.to_string(),
        "",
        f"Best single-threshold balanced accuracy using green_mean only: {max(bal):.3f} "
        f"(at threshold {int(np.argmax(bal))})",
        "",
        "Image size vs class:", size_tab.to_string(), "",
        "File format vs class:", fmt_tab.to_string(),
    ]
    (OUT / "eda_summary.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    print(f"\nFigures and tables saved in {OUT}")


if __name__ == "__main__":
    main()
