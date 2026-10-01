# ANA Positive / Negative Classifier (AIDA_HEp-2)

Stage 1 of *Automated Classification of ANA Immunofluorescence Patterns Using Deep Learning*:
an image-level classifier that labels a HEp-2 immunofluorescence image as **ANA-Positive** or
**ANA-Negative** and reports a confidence.

Full plan, decisions and results: [`Docs/Phase_Plan_ANA_PosNeg.md`](Docs/Phase_Plan_ANA_PosNeg.md)

## Results so far

5-fold cross-validation split by patient sample (2,073 images, 998 samples), test sets, cut-off 0.5:

| Model | Parameters | Balanced accuracy | Sensitivity | Specificity | AUC |
|---|---|---|---|---|---|
| EfficientNet-B0 (chosen) | 4.0 M | 0.949 ± 0.019 | 0.928 ± 0.031 | 0.971 ± 0.012 | 0.989 ± 0.007 |
| ResNet-50 | 23.5 M | 0.950 ± 0.015 | 0.932 ± 0.032 | 0.967 ± 0.009 | 0.986 ± 0.009 |
| ConvNeXt-Tiny | 27.8 M | 0.954 ± 0.015 | 0.948 ± 0.034 | 0.960 ± 0.025 | 0.987 ± 0.011 |

The three models are statistically tied; EfficientNet-B0 was chosen as the smallest and fastest.
Temperature scaling lowers the pooled calibration error (ECE) from 0.042 to 0.025.
These results are on AIDA_HEp-2 only; performance on images from other laboratories is untested.

## Folders

| Folder | Contents |
|---|---|
| `Docs/` | Phase plan, website integration notes |
| `scripts/` | Local data pipeline: cleaning, EDA, preprocessing, folds (`config.py` holds all paths/settings) |
| `notebooks/` | Colab notebooks: setup, training, model comparison, calibration |
| `outputs/` | Logs, metric tables and plots |

## Data is not included

The AIDA_HEp-2 dataset may not be redistributed, so the images, labels, dataset documents and trained
weights are **not** in this repository. To reproduce:

1. Obtain AIDA_HEp-2 from its authors and place the image folder at `../Model Training/aida_project_database`
   and the ground-truth spreadsheet at `Docs/AIDA_HEp-2_GroundTruth.xlsx`.
2. `python -m pip install -r requirements-local.txt`
3. Run, in order: `scripts/prepare_manifest.py`, `scripts/eda.py`, `scripts/preprocess.py`, `scripts/make_folds.py`.
   This creates `data/manifest.csv`, `data/processed/` and `data/AIDA_processed.zip`.
4. Upload the zip to a **private** Google Drive folder `ANA_PosNeg/data/`, then run the notebooks in order on a Colab T4 GPU.

## Citation (required by the AIDA licence)

A. Benammar Elgaaied et al., "Computer-Assisted Classification Patterns in Autoimmune Diagnostics: The AIDA Project," *BioMed Research International*, vol. 2016, Art. ID 2073076, 2016, doi: 10.1155/2016/2073076.

For research and educational use only; not a diagnostic tool.
