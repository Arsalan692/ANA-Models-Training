# Phase 2 — EDA Findings (AIDA_HEp-2, cleaned manifest)

Script: `scripts/eda.py` · Figures: `class_counts.png`, `sample_grid.png`, `brightness_hist.png` · Numbers: `eda_summary.txt`, `image_stats.csv`

## 1. Class balance (`class_counts.png`)
- Images: **1,492 Positive vs 581 Negative** (2.6 : 1) → imbalanced at image level.
- Samples: **468 Positive vs 530 Negative** → roughly balanced at patient level.
- Cause: positive wells have ~3 images each, negative wells mostly 1.
- **Consequence:** use a class-weighted loss and balanced accuracy (Phase 5, 8).

## 2. Brightness separates the classes — but not completely (`brightness_hist.png`)
| Measure (green channel, 0–255) | Negative median | Positive median | AUC alone |
|---|---|---|---|
| Average brightness | 20.6 | 42.4 | 0.914 |
| Brightest spots (99th percentile) | 34 | 104 | 0.957 |

- The best single brightness cut-off reaches only **84.3% balanced accuracy**.
- Brightest-spot level separates better than average brightness → **where** the light is (small bright structures in nuclei) matters more than **how much** light there is overall.
- **Consequence:** a CNN is justified. It can learn spatial structure (nuclear staining, dots, speckles) that a single brightness number cannot capture.

## 3. The hard cases (`sample_grid.png`, rows 3–4)
- **Dimmest positives** (e.g. `103_*`, `455_*`, mean brightness 7–10): almost black overall, but cells show small bright nuclear dots. Positive because of *structure*, not overall brightness.
- **Brightest negatives** (mean ~40): diffuse glow over the whole cell without distinct nuclear structure; some have a grey colour cast (`984_1`, `987_1`) or debris artefacts (`684_1`).
- **Consequences:**
  - No per-image contrast stretching (Phase 3) — it would make dim positives and bright negatives indistinguishable in a different way.
  - Brightness augmentation must stay mild (Phase 5); a dim positive darkened further may lose its dots.
  - Downscaling must keep small dots visible → keep 512 px (not smaller) in Phase 3.

## 4. File-level shortcut check
| Property | Finding | Handled by |
|---|---|---|
| Image size | 8 sizes; **1388×1036 is 100% Positive (111 images)**, 2048×1536 is 89% Positive | Phase 3 resize to 512 px short side + Phase 5 square crops (removes size and aspect ratio) |
| File format | tif 81% Positive, png 65% — all formats contain both classes | Phase 3 conversion to PNG |

Both are removed by preprocessing, using pixels only (no spreadsheet metadata).

## Decisions confirmed for next phases
1. Resize shorter side to **512 px**, convert to RGB PNG, **no contrast normalisation** (Phase 3).
2. **Class-weighted** loss; report **balanced accuracy + AUC** (Phase 5, 8).
3. **Mild** brightness augmentation only (Phase 5).
