# Plan — ANA Positive / Negative Classifier (with Confidence)

**Project:** Automated Classification of ANA Immunofluorescence Patterns Using Deep Learning
**This stage:** Objective O2 — image-level ANA-Positive vs ANA-Negative, with a confidence score
**Supervisor:** Ubaid Chawla  **Prepared:** 1 October 2026  **Last updated:** 1 October 2026
**Scope of this stage:** AIDA_HEp-2 dataset only. Other datasets (I3A, Indus) are out of scope for now.

## Current status

| Phase | Status | Headline result |
|---|---|---|
| 0 Setup | ✅ Done | Folders, libraries, Colab T4 + Drive verified |
| 1 Cleaning | ✅ Done | 2,073 images / 998 samples in `manifest.csv` |
| 2 EDA | ✅ Done | Brightness alone reaches only 84.3% balanced accuracy |
| 3 Preprocessing | ✅ Done | 2,073 PNGs at 512 px, brightness unchanged |
| 4 Folds | ✅ Done | 5 folds by patient, leakage checks pass |
| 5 Baseline | ✅ Done | EfficientNet-B0: **0.949 ± 0.019 balanced accuracy, AUC 0.989** |
| 6 Model comparison | ✅ Done | 3 models statistically tied on validation and test → **EfficientNet-B0** chosen (smallest, fastest) |
| 7 Calibration | ✅ Done | Mean T = 2.09; pooled test ECE 0.042 → 0.025; errors among "≥ 99% confident" predictions 2.3% → 0.6%. Shift/cut-off question still open |
| 8 Final evaluation | ☐ To do | |
| 9 `predict.py` | ☐ To do | |
| 10 Write-up | ☐ To do | |

---

## 0. The goal in one paragraph

We give the model one HEp-2 microscope image. It answers **"ANA-Positive"** or **"ANA-Negative"** and says **how sure it is**, e.g. `ANA-Positive — confidence 94.3%`. We train it on the AIDA_HEp-2 dataset using **PyTorch** on a **Google Colab T4 GPU**. This is Stage 1 of the bigger system: later, only images called "Positive" will go on to pattern classification (AC-1, speckled, nucleolar …).

> **Why start with positive/negative?** It is the first thing a lab reader decides, it is the easiest decision to learn, and AIDA is the only public dataset that has negative images — so this is the stage we can build right now while IRB approval for Indus data is pending.

---

## 1. What we know about the data (from inspection)

| Fact | Value | Why it matters |
|---|---|---|
| Labelled images (as published) | 2,080 (1,498 Positive, 582 Negative) | Positives are ~2.6× more common → **class imbalance** |
| **Images we use (after cleaning)** | **2,073 (1,492 Positive, 581 Negative)** | See Phase 1 result |
| Wells (patient samples) | 998 usable (468 Positive, 530 Negative) | Images of the same patient look alike → must split by well |
| Images per well | Negative: mostly 1 · Positive: mostly 3 (up to 5) | Explains why positives dominate at image level |
| Label consistency | Every image in a well has the same label | Safe to give each image its well's label |
| Resolutions | 8 different sizes, 640×480 → 3136×2352 | Must resize to one size |
| File formats | bmp, jpg, png, tif (+ upper/lower case) | Must convert to one format |
| Extra metadata columns | Age, sex, microscope, camera, pixel size | **Not used** — other datasets won't have them (see design rule below) |
| Sheet ↔ disk mismatches | 28 wrong extensions/typos, 5 labelled files missing, repeated names hiding 11 unlabelled files, 6 names existing in two formats | Cleaned in Phase 1 |
| Licence | AIDA must **not** be redistributed | Keep data in our private Google Drive only |

> **Design rule — the pipeline uses only what every dataset has.** Every ANA dataset we will ever use (AIDA, I3A, Indus clinical data) gives us, at minimum: an **image**, its **label**, and **which sample/patient it came from**. Metadata such as camera or microscope exists only in the AIDA spreadsheet, so no step of training, splitting or evaluation depends on it. Each new dataset only needs a small *adapter* that converts its own label file into our standard manifest format (§ Phase 1); everything after that is identical for all datasets.

---

## 2. Phases at a glance

| Phase | Name | Where it runs | Main output |
|---|---|---|---|
| 0 | Environment & folder setup | Laptop + Colab | Folder structure, `scripts/config.py`, `notebooks/00_colab_setup.ipynb` |
| 1 | Data cleaning → manifest | Laptop | `scripts/prepare_manifest.py` → `data/manifest.csv` |
| 2 | Exploratory Data Analysis (EDA) | Laptop | `scripts/eda.py` → `outputs/eda/` |
| 3 | Preprocessing (resize/convert) | Laptop | `scripts/preprocess.py` → `data/processed/`, `data/AIDA_processed.zip` |
| 4 | 5-fold cross-validation split | Laptop | `scripts/make_folds.py`, `scripts/splits.py` → `fold` column |
| 5 | Baseline model training | Colab T4 | `notebooks/01_train_baseline.ipynb` → Drive `outputs/efficientnet_b0/` |
| 6 | Model comparison (3 backbones) | Colab T4 | Comparison table |
| 7 | Confidence calibration | Colab T4 | Temperature value `T` |
| 8 | Final evaluation (5-fold results) | Colab T4 | Metrics (mean ± std), confusion matrix |
| 9 | Prediction script | Laptop / Colab | `predict.py image.png → label + confidence` |
| 10 | Write-up for FYP-I report | — | Results section + figures |

We finish and discuss each phase before starting the next.

---

## 3. Phases in detail

Each phase lists **What**, **Why**, **Key terms**, and **Done when**.

### Phase 0 — Environment & folder setup

**What**
```
Second Model Training/
├── Docs/                        ← proposal, handoff, AIDA docs, this plan
├── data/
│   ├── manifest.csv             ← clean label table + fold column (Phases 1, 4)
│   ├── aida_file_map.csv        ← which raw file each row came from (Phase 1)
│   ├── processed/AIDA/*.png     ← resized PNGs (Phase 3) — private, never shared
│   └── AIDA_processed.zip       ← upload to Drive (Phase 3/4)
├── scripts/
│   ├── config.py                ← all paths + settings (seed 42, 5 folds, 512 px, threshold 0.5)
│   ├── prepare_manifest.py      ← Phase 1
│   ├── eda.py                   ← Phase 2
│   ├── preprocess.py            ← Phase 3
│   ├── make_folds.py, splits.py ← Phase 4
├── notebooks/
│   ├── 00_colab_setup.ipynb     ← Phase 0 Colab check
│   └── 01_train_baseline.ipynb  ← Phases 5–6 training
├── outputs/                     ← logs, EDA figures (local)
├── requirements-local.txt
└── README.md

Google Drive:  My Drive/ANA_PosNeg/data/AIDA_processed.zip
               My Drive/ANA_PosNeg/outputs/<model_name>/round_<k>/   ← trained models + predictions
Raw images (read-only):  ../Model Training/aida_project_database
```
- Local Python libraries: `pandas`, `openpyxl`, `pillow`, `scikit-learn`, `matplotlib`, `tqdm` (+ CPU `torch`, `torchvision`, `timm` for smoke tests and `predict.py`).
- Colab: `torch`, `torchvision` (pre-installed), `timm` (library of pretrained models).

**Why** — Keeping raw data, cleaned data, code and results apart means we never accidentally overwrite the original dataset, and anyone (supervisor, teammates) can follow the project. `config.py` holds every path and setting in one place, so a change is made once.

**Done when** — folders exist, libraries install, Colab can see our Google Drive.

**Result (done 1 Oct 2026)** — folders and files above created; Colab check passed (T4 GPU, PyTorch sees GPU, `timm` loads EfficientNet-B0, Drive folder `ANA_PosNeg` created).

---

### Phase 1 — Data cleaning → `manifest.csv`

**What**
1. Read `AIDA_HEp-2_GroundTruth.xlsx`; use only two columns: filename and Positive/Negative.
2. Fix filename typos and wrong extensions (`34_3.bpm`, `466_1_bmp`, `277_3,jpg`, `104_1.tif` that is really `.bmp`) by matching on the name without extension.
3. Drop the 5 rows whose image is missing from disk.
3b. Where the sheet repeats a name by mistake (e.g. `198_1` typed twice instead of `198_4`), keep one row and **recover the forgotten file with its well's label** — safe because every image of a well has the same label.
4. Files sharing one name but with different extensions are **different pictures** (checked pixel by pixel). Rule (decided 1 Oct 2026):
   - `281_1/2/3`: the sheet labels both `.png` and `.tif` → **keep the `.png`**, drop the `.tif`.
   - `320_2`, `320_3`, `433_3`: the sheet labels only the `.jpg` → keep the `.jpg`; the extra `.tif`/`.png` files have no label and are ignored.
5. Build the **standard manifest** — the same 4 columns for every dataset we ever add:

| Column | Meaning | AIDA example |
|---|---|---|
| `image_path` | where the processed PNG is, relative to `data/processed/` | `AIDA/281_1.png` |
| `sample_id` | which patient sample it belongs to (dataset name + ID, so IDs never clash across datasets) | `AIDA_281` |
| `label` | `1` = Positive, `0` = Negative | `1` |
| `source` | which dataset it came from | `AIDA` |

   (Phase 4 later adds a fifth column, `fold`.)

6. Save as `manifest.csv`. The script stops with an error (assertions) if a label isn't 0/1, two rows point to one image, a file is missing, or a patient has mixed labels. The AIDA-specific code (reading the Excel file, fixing its typos) lives in one small **adapter** function; nothing after this step knows or cares that the data came from AIDA.

**Why**
- A model is only as good as its labels. One wrong filename-to-label match teaches the model something false.
- A single clean table (a **manifest**) means every later step reads the same truth, and we can show exactly which images were used and which were dropped.
- **Only 4 generic columns** — camera, microscope, age, sex and pixel size are dropped because other datasets won't have them. If the pipeline needed them, it would break the moment we add Indus or I3A data.
- `sample_id` is kept because *every* dataset has some notion of "which patient/slide", and we need it to split correctly (Phase 4).
- We **don't** use the `StainingPatterns` column now — it's for the next stage.

**Key terms**
- **Ground truth** — the correct answer for each image, here agreed by three expert immunologists.
- **Manifest** — a table listing every data item and its label.
- **Adapter** — a small dataset-specific piece of code that converts that dataset's own format into our standard format.

**Done when** — manifest has ~2,070 rows, each pointing to a real file, with a short log of what was dropped and why.

**Result (done 1 Oct 2026)** — `scripts/prepare_manifest.py` → `data/manifest.csv`, log in `outputs/phase1_cleaning_log.txt`
- **2,073 images** (1,492 Positive, 581 Negative) from **998 samples** (468 Positive, 530 Negative).
- 28 filename/extension typos fixed; 5 labelled images missing from disk dropped (well 825 lost entirely; well 73 never existed in the dataset).
- **Typing mistakes found:** the sheet repeats a name where the 4th/5th image was meant (e.g. `198_1` typed twice while `198_4`, `198_5` had no label). Since all images of a well share one label, 11 such unlabelled files were **recovered with their well's label**.
- Ignored: `281_1/2/3.tif` (png kept), `320_2/3.tif`, `433_3.png` (unlabelled extra copies).

---

### Phase 2 — Exploratory Data Analysis (EDA)

**What** — Look at the data before modelling:
- Class counts (images and samples).
- Sample grid of Positive vs Negative images.
- Brightness histograms per class (how much do positives and negatives overlap?).
- Image sizes per class.

**Why**
- To *see* what separates the classes (positives glow; negatives are dim) and spot problems early.
- To find the hard cases: weak positives whose brightness overlaps with negatives.
- To check for **shortcuts** that are visible from the images themselves (e.g. image size differing between classes) — this needs no metadata, only the pixels.

**Key terms**
- **EDA** — exploring data with statistics and plots before building models.
- **Confounder / shortcut** — a feature that happens to correlate with the label in our data but isn't the real cause (e.g. if all positive images happened to be a different size). Models love shortcuts because they are easy; uniform preprocessing (Phase 3) removes the obvious ones.

**Done when** — a one-page note of findings with 3–4 figures.

**Result (done 1 Oct 2026)** — `scripts/eda.py` → `outputs/eda/` (findings in `EDA_findings.md`)
- Brightness alone: AUC 0.914 (average) / 0.957 (brightest spots); best single cut-off only **84.3% balanced accuracy** → a CNN that learns structure is needed.
- Dimmest positives are nearly black but show small nuclear dots; brightest negatives have diffuse glow → confirms no contrast stretching, mild brightness augmentation, keep 512 px.
- 1388×1036 images are 100% Positive → removed by resize + square crops.

---

### Phase 3 — Preprocessing (resize + convert)

**What**
- Convert every image to **RGB PNG** (removes the bmp/tif/jpg mix; some files are palette `P` or `RGBA` mode).
- Resize so the **shorter side is 512 pixels**, keeping the aspect ratio (so cells don't get stretched).
- Resampling filter: LANCZOS (keeps small bright dots sharp when shrinking).
- Zip the result with `manifest.csv` (760 MB instead of ~6 GB) and upload to our **private** Google Drive: `My Drive/ANA_PosNeg/data/`.
- Images are stored **uncropped**; cropping happens when each image is loaded for training/testing (see Phase 5), so no content is permanently discarded.

**What we deliberately do NOT do:** per-image contrast stretching or histogram equalisation.

**Why**
- **One size & format** removes resolution and file type as shortcuts, and neural networks need fixed-size input anyway.
- **512 px** keeps enough detail to see nuclear staining, while being small enough for fast Colab loading and T4 memory. We'll train on 512×512 square crops (see Phase 5).
- **No contrast stretching** — this is the most important preprocessing decision. The difference between Positive and Negative *is* brightness (fluorescence intensity). Auto-contrast would brighten a dark negative until it looks like a positive, destroying the signal we want to learn.
- **Lossless PNG** so we don't add compression artefacts.

**Key terms**
- **Preprocessing** — transforming raw data into a consistent form the model can use.
- **Aspect ratio** — width ÷ height; keeping it avoids distorting cell shapes.

**Done when** — every manifest row has a processed PNG; spot-check that images still look right.

**Result (done 1 Oct 2026)** — `scripts/preprocess.py` → `data/processed/AIDA/*.png` + `data/AIDA_processed.zip` (760 MB)
- 2,073 PNGs; output sizes 683×512, 659×512, 612×512, 686×512; 24 small (640×480) images upscaled.
- Brightness unchanged: mean difference 0.03, max 0.26 (on 0–255) — no hidden normalisation.
- Faint nuclear dots in the dimmest positives remain visible after resizing.
- ⚠ **686×512 = exactly the 111 all-positive images** → width still leaks the label. The model must therefore **always receive a square crop** (training, validation, test and `predict.py`), never the full-width image.

---

### Phase 4 — 5-fold cross-validation split  *(decided 1 Oct 2026)*

**What**
- Divide the 998 wells (`sample_id`) into **5 folds** (~200 wells each), split **by sample**, not by image.
- Keep the Positive/Negative ratio similar in every fold (uses only `label` and `sample_id`).
- Run the experiment 5 times. In round *k*:
  - **Test** = fold *k* (~20% of patients; never touched during training)
  - **Validation** = 1/6 of the remaining patients (~13%), picked per class
  - **Train** = everything else (~67%)
- Report every metric as **mean ± standard deviation over the 5 rounds**.
- Tools: folds by scikit-learn `StratifiedGroupKFold` (seed 42), stored in the manifest's `fold` column; validation patients by numpy `RandomState(42 + k)` in `splits.get_round`.
- `sample_id` is used **only here, to decide which fold each image goes to**. It is **never an input to the model** — the model only ever receives pixels.

**Why**
- **Train set** — the model learns from it.
- **Validation set** — we use it to choose settings (when to stop, calibration temperature). The model never trains on it.
- **Test set** — used only to report honest numbers. If we tuned on it, our reported accuracy would be optimistic.
- **Why 5-fold instead of one split?** With one split, the test set is only ~150 wells; results depend a lot on *which* wells landed there (luck). With 5-fold, **every well is tested exactly once**, and the ± spread tells us how stable the model is. Cost: 5× the training time, acceptable for ~2,000 images on a T4.
- **Why by well?** A positive well has 3–6 photos of the *same* patient's slide. If photo 1 is in training and photo 2 in test, the model has effectively already "seen" the test sample → inflated accuracy. This is **data leakage**. It's also a locked decision in the project handoff.
- **Why stratify?** A purely random split could by chance put too few negatives into one fold, making that fold's numbers unreliable.

**Key terms**
- **Data leakage** — information from the test set sneaking into training, making results look better than reality.
- **Stratified split** — each split keeps the same class proportions as the whole dataset.
- **Group split** — all items of one group (one well) go to the same split.
- **Random seed** — a fixed number that makes "random" operations repeatable.

**Key terms (extra)**
- **k-fold cross-validation** — split data into k parts; train k times, each time testing on a different part; average the results.
- **Mean ± std** — average performance and how much it varies between folds.

**Done when** — manifest has a `fold` column (0–4); a table shows class counts per fold; zero samples appear in two folds (checked by code).

**Result (done 1 Oct 2026)** — `scripts/make_folds.py` (+ `scripts/splits.py`), log in `outputs/phase4_folds_log.txt`
- 5 folds of ~415 images / ~200 samples each; every fold ≈ 72% Positive images, ≈ 47% Positive samples.
- Each round: **train ~1,380–1,392 images (~666 samples) / validation ~267–280 (~133) / test ~415 (~200)** ≈ 67% / 13% / 20%.
- Validation is carved from the non-test folds (`splits.get_round`, seed 42 + k); the same function is used in the Colab notebook.
- **Fix (1 Oct 2026):** the first version used scikit-learn's shuffling, which gave a different validation set on Colab (different library version). Now uses numpy `RandomState` (identical across versions); verified notebook split == local split. Round 0 is now train 1,392 / val 267 / test 414.
- Code-checked: no patient in two folds; no patient shared between train/val/test in any round.
- `AIDA_processed.zip` rebuilt with the updated manifest.

---

### Phase 5 — Baseline model training (Colab T4)

**What**
- **Model:** EfficientNet-B0, **pretrained on ImageNet**, last layer replaced by **one output** (a logit).
- **Input:** **512×512 square crops** of the 512-px-tall images (decided 1 Oct 2026) — keeps ~75% of each image at full resolution (no further shrinking, so small nuclear dots stay sharp).
  - Training: **random** 512×512 crop (slides left/right each epoch, so over many epochs the model sees the whole image).
  - Validation / test / `predict.py`: **centre** 512×512 crop (the outer ~12% on each side is not seen).
  - Always square, so image width (which reveals the 111 all-positive images) never reaches the model.
  - Possible later upgrade (option D, deferred): average predictions over left + right crops to cover 100% of the image at test time.
- **Augmentation (train only):** random crop, horizontal/vertical flip, 90° rotations, brightness ×0.9–1.1 only (small, so intensity signal survives). Applied on-the-fly when an image is loaded, so every epoch sees a different version.
- **Loss:** Binary Cross-Entropy with a **positive-class weight** = (negatives ÷ positives) in that round's *training* set ≈ 0.39, to handle imbalance.
- **Optimizer:** AdamW with 1 warm-up epoch then cosine decay.
- **Early stopping** on validation balanced accuracy (ties → lower validation loss).
- **Mixed precision (AMP)** for speed on the T4.
- All images held in RAM (~2.2 GB) so PNGs are decoded once, not every epoch.
- Per round, saved to Drive `outputs/<model>/round_<k>/`: `best.pt` (best model), `history.csv` (per-epoch numbers), `predictions.csv` (raw logits for validation + test images — reused by Phases 7–8 without retraining), `summary.json`. A finished round is skipped on re-run, so a Colab disconnect loses at most one round.

**Settings actually used** (`CFG` in the notebook)

| Setting | Value | Why |
|---|---|---|
| `img_size` | 512 | largest square crop that fits the image height |
| `batch_size` | 32 | fits a T4 with mixed precision (use 16 if "CUDA out of memory") |
| `epochs` | 30 (maximum) | fine-tuning converges fast (best epochs were mostly 7–9); also sets the length of the learning-rate schedule; worst case ~10 min/round |
| `patience` | 7 | validation scores jump ±0.03 between epochs, so 2–3 would stop on a random dip; 7 rides out the noise without wasting time |
| `lr` | 3e-4 | common fine-tuning value for AdamW |
| `weight_decay` | 1e-2 | mild regularisation against overfitting |
| `warmup_epochs` | 1 | small first steps so pretrained weights aren't disturbed |
| `brightness` | 0.1 | ±10% only — brightness is the signal |
| `threshold` | 0.5 | decided 1 Oct 2026 |
| `seed` | 42 (+ round number) | reproducible runs |

These are sensible defaults, **not tuned**. If ever tuned, it must be on validation scores only.

**Test-set isolation (verified in the code)** — within a round the test fold is used exactly once, after training has finished and the best model is frozen. Weight updates and the class weight use the training set only; choosing the best epoch and stopping use the validation set only.

**Why these choices**
- **Transfer learning / pretrained model** — we only have ~1,385 training images per round. Training a CNN from scratch needs far more. A model pretrained on ImageNet (1.2 M everyday photos) already knows how to detect edges, textures and blobs; we only fine-tune it to our task. Every paper in our literature review does this.
- **EfficientNet-B0** — small (5.3 M parameters, ~0.39 GFLOPs vs ~4 for ResNet-50 / ConvNeXt-Tiny), accurate, and fast on a free T4. Big models overfit small datasets and train slowly. Rahman et al. (our review paper) also found "deeper is not better" for HEp-2. It is the **baseline**, not the final choice — Phase 6 compares it fairly against two others.
- **One output + sigmoid** — binary problem, so one number is enough: `p = sigmoid(logit)` = probability of Positive. Negative probability is `1 − p`.
- **Class weight** — without it the model can score ~72% by mostly guessing "Positive". The weight makes each negative mistake cost more. (Side effect seen in the results: the model leans slightly towards "Negative", so sensitivity < specificity.)
- **Random crops at training time, not saved crops** — a saved crop would be the same every epoch and would throw 25% of each image away for good; random crops let the model see the whole image over the epochs and act as extra augmentation.
- **Rotations & flips** — cells have no natural "up" direction, so a rotated image is still a valid example. Rahman et al. found rotation augmentation essential for HEp-2.
- **Only mild brightness jitter** — same reason as Phase 3: brightness is the signal.
- **512×512 crops** — the largest square that fits the 512-px image height: keeps ~75% of the field and full detail. Costs ~1.8× the compute of 384 crops; fine on a T4 for ~2,000 images (batch size may need to be 16–32 for the larger models).

**Key terms**
- **Epoch** — one full pass over the training set.
- **Batch** — the small group of images processed together before each weight update.
- **Learning rate** — how big a step the optimizer takes when updating weights.
- **Loss function** — a number measuring how wrong the model is; training minimises it.
- **Logit** — the raw model output before sigmoid; can be any real number.
- **Sigmoid** — squashes a logit into 0–1 so it reads as a probability.
- **Fine-tuning** — continuing training of a pretrained model on our own data.
- **Data augmentation** — making random, label-preserving changes to training images so the model sees more variety.
- **Overfitting** — the model memorises training images and performs worse on new ones. Early stopping and augmentation fight this.
- **Early stopping** — stop training when validation performance stops improving.

**Done when** — training curves (loss & balanced accuracy vs epoch) look sensible and validation balanced accuracy is recorded.

**History** — a first run of round 0 (old validation split, since replaced) gave test balanced accuracy 0.925; it exposed the Colab/laptop split mismatch fixed in Phase 4. The notebook was smoke-tested on CPU before any GPU run. Speed on T4: ~20 s/epoch, ~35 min for all 5 rounds.

**Final baseline — EfficientNet-B0, all 5 rounds (1 Oct 2026, fixed split)** — test sets, uncalibrated, threshold 0.5:

| Round | Best epoch | Bal. acc | Sensitivity | Specificity | AUC |
|---|---|---|---|---|---|
| 0 | 28 | 0.930 | 0.903 | 0.957 | 0.979 |
| 1 | 8 | 0.941 | 0.900 | 0.983 | 0.993 |
| 2 | 9 | 0.948 | 0.930 | 0.966 | 0.986 |
| 3 | 7 | 0.948 | 0.930 | 0.966 | 0.989 |
| 4 | 7 | 0.980 | 0.976 | 0.983 | 0.998 |
| **Mean ± std** | | **0.949 ± 0.019** | **0.928 ± 0.031** | **0.971 ± 0.012** | **0.989 ± 0.007** |

- Every round: sensitivity < specificity → systematic tendency to under-call positives at 0.5.
- Validation metrics jump ±0.03 between epochs; best epoch ranges 7–28 → model selection on balanced accuracy at a fixed 0.5 threshold is noisy.
- Train loss → ~0.001 while validation loss rises after ~epoch 7 in rounds 3–4 → overfitting, contained by early stopping.
- For reference: the best brightness-only rule (Phase 2) reached 0.843 balanced accuracy → the CNN adds ~10 points.
- AUC 0.989 means the model ranks images almost perfectly; the lower sensitivity is about where the 0.5 cut-off sits, not about what the model learned.

---

### Phase 6 — Model comparison

**What** — Train three backbones with the **same** split, augmentation and settings:

| Model | Size | Why include it |
|---|---|---|
| ResNet-50 | 23.5 M | Classic, widely used baseline in HEp-2 papers |
| EfficientNet-B0 | 4.0 M | Efficient, strong on small data (our default) |
| ConvNeXt-Tiny | 27.8 M | Modern CNN, often best accuracy in 2020s |

Pick the winner by **validation** balanced accuracy and AUC (not test!). Test scores are shown only as a separate, final column.

All three are compared using the same 5 folds (3 models × 5 folds = 15 training runs).

**How to run (status: all three models trained)** — in `notebooks/01_train_baseline.ipynb` change `model_name` to `"resnet50"`, keep `ROUNDS = [0, 1, 2, 3, 4]`, Run all (~1–1.5 h); repeat with `"convnext_tiny"`. Each model writes to its own Drive folder. If "CUDA out of memory": set `batch_size=16`. If Colab disconnects: Run all again — finished rounds are skipped.

| Model | Status |
|---|---|
| `efficientnet_b0` | ✅ done (Phase 5 table) |
| `resnet50` | ✅ done (table below) |
| `convnext_tiny` | ✅ done (table below) |

**ResNet-50, all 5 rounds (1 Oct 2026)** — same split and settings as the baseline; test sets, uncalibrated, threshold 0.5:

| Round | Best epoch | Best val bal. acc | Test bal. acc | Sensitivity | Specificity | AUC |
|---|---|---|---|---|---|---|
| 0 | 14 | 0.979 | 0.939 | 0.913 | 0.966 | 0.975 |
| 1 | 21 | 0.985 | 0.951 | 0.920 | 0.983 | 0.981 |
| 2 | 6 | 0.980 | 0.962 | 0.967 | 0.957 | 0.994 |
| 3 | 2 | 0.970 | 0.931 | 0.896 | 0.966 | 0.981 |
| 4 | 12 | 0.964 | 0.966 | 0.966 | 0.966 | 0.997 |
| **Mean ± std** | | **0.975 ± 0.008** | **0.950 ± 0.015** | **0.932 ± 0.032** | **0.967 ± 0.009** | **0.986 ± 0.009** |

- Test balanced accuracy 0.950 vs EfficientNet-B0's 0.949: a tie, far inside the ± spread. AUC is slightly lower (0.986 vs 0.989), also inside the spread.
- Cost: ~28 s/epoch vs ~20 s, and 25 M parameters vs 5 M → five times bigger for no measurable gain so far.
- Same pattern as the baseline: sensitivity below specificity in 4 of 5 rounds; validation scores jump ±0.03 between epochs; best epoch ranges 2–21.
- Validation scores (0.976) sit above test scores (0.950) because the best epoch is *picked* on validation, which flatters it (selection bias). Compare models validation-to-validation only.
- Still needed for the fair comparison: EfficientNet-B0's best validation scores (saved in Drive `outputs/efficientnet_b0/round_*/summary.json` and `history.csv`).

**ConvNeXt-Tiny, all 5 rounds (1 Oct 2026)** — same split and settings; test sets, uncalibrated, threshold 0.5:

| Round | Best epoch | Best val bal. acc | Test bal. acc | Sensitivity | Specificity | AUC |
|---|---|---|---|---|---|---|
| 0 | 25 | 0.943 | 0.936 | 0.923 | 0.948 | 0.969 |
| 1 | 21 | 0.982 | 0.943 | 0.903 | 0.983 | 0.985 |
| 2 | 27 | 0.977 | 0.974 | 0.957 | 0.991 | 0.993 |
| 3 | 20 | 0.981 | 0.961 | 0.983 | 0.940 | 0.990 |
| 4 | 18 | 0.956 | 0.957 | 0.973 | 0.940 | 0.997 |
| **Mean ± std** | | **0.968 ± 0.017** | **0.954 ± 0.015** | **0.948 ± 0.034** | **0.960 ± 0.025** | **0.987 ± 0.011** |

- Slowest of the three: ~36 s/epoch and 140 epochs in total (~85 min), because it learns later (best epoch 18–27).
- Rounds 0 and 2 used all 30 epochs without early stopping → it may not have finished learning. The shared learning rate (3e-4) was not tuned for ConvNeXt, so this model may be slightly under-served by our settings.
- Sensitivity and specificity are the most balanced of the three (0.948 / 0.960).

**Three-model summary** (mean ± std over 5 rounds):

| Model | Parameters | Time / epoch | Best val bal. acc | Test bal. acc | Test sensitivity | Test specificity | Test AUC |
|---|---|---|---|---|---|---|---|
| EfficientNet-B0 | 4.0 M | ~20 s | 0.971 ± 0.012 | 0.949 ± 0.019 | 0.928 ± 0.031 | 0.971 ± 0.012 | 0.989 ± 0.007 |
| ResNet-50 | 23.5 M | ~28 s | 0.975 ± 0.008 | 0.950 ± 0.015 | 0.932 ± 0.032 | 0.967 ± 0.009 | 0.986 ± 0.009 |
| ConvNeXt-Tiny | 27.8 M | ~36 s | 0.968 ± 0.017 | 0.954 ± 0.015 | 0.948 ± 0.034 | 0.960 ± 0.025 | 0.987 ± 0.011 |

Numbers above are from `02_compare_models.ipynb` run on the real outputs (files copied to `outputs/comparison/`). Extra columns from that run:

| Model | Val AUC | Epochs run (5 rounds) | Train time | Val rounds won | Test rounds won |
|---|---|---|---|---|---|
| EfficientNet-B0 | 0.995 ± 0.004 | 89 | ~30 min | 1 | 1 |
| ResNet-50 | 0.991 ± 0.008 | 90 | ~42 min | 3 | 2 |
| ConvNeXt-Tiny | 0.991 ± 0.007 | 140 | ~84 min | 1 | 2 |

- Validation balanced accuracy spans 0.968–0.975 (a 0.007 range) against round-to-round spreads of 0.008–0.017 → a tie on validation too.
- ResNet-50 is nominally first on validation balanced accuracy; EfficientNet-B0 is first on validation AUC and on test AUC; ConvNeXt-Tiny is first on test balanced accuracy. No model leads on every measure.

- All three test balanced accuracies lie within 0.005 of each other, while the spread between rounds is 0.015–0.019 → no model is measurably better.
- Validation and test rank the two big models in opposite orders (ResNet-50 leads on validation, ConvNeXt-Tiny on test) → further evidence that the differences are noise.
- Resolved: validation is the selection criterion, test is shown alongside; final choice is in "Outcome" below.

**Comparison notebook** — `notebooks/02_compare_models.ipynb` (no GPU, no training, under a minute). It reads `predictions.csv`, `history.csv` and `summary.json` of every round from Drive and produces: a table with validation **and** test scores side by side (sorted by validation), a round-by-round table with "rounds won", and a chart. Outputs go to Drive `outputs/comparison/` (`model_comparison.csv`, `per_round.csv`, `model_comparison.png`). Run on the real outputs on 1 Oct 2026; its test numbers match the training notebook's summary tables exactly.

Exact parameter counts with our 1-output head (counted by the notebook): EfficientNet-B0 4.0 M, ResNet-50 23.5 M, ConvNeXt-Tiny 27.8 M. The commonly quoted 5.3 M / 25.6 M / 28.6 M include the original 1000-class ImageNet head, which we remove.

**Why** — The FYP-I deliverables include an "initial architecture comparison for the positive/negative stage". Comparing under identical conditions is the only fair way to choose; the decision is then based on evidence, not preference.

**Done when** — a comparison table with validation metrics and training time per model.

**Outcome (decided 1 Oct 2026): EfficientNet-B0.** The three architectures were statistically indistinguishable on validation (0.968–0.975 balanced accuracy, spreads 0.008–0.017) and on test (0.949–0.954). With accuracy unable to separate them, the smallest and fastest model was taken: 4.0 M parameters (about 6–7× smaller) and ~30 min for 5 rounds; it also has the best AUC on validation and test. Noted: ConvNeXt-Tiny had the highest test sensitivity (0.948) but not on validation (0.959 vs 0.956 / 0.963), and two of its rounds hit the 30-epoch limit, so it may be under-trained with our untuned settings.

---

### Phase 7 — Confidence calibration

**What** — **Temperature scaling:** learn one number `T` on the validation set so that `p = sigmoid(logit / T)` becomes an honest probability. Measure before/after with a **reliability diagram** and **ECE**.

The final output for each image:
```
p = sigmoid(logit / T)
label      = "ANA-Positive" if p ≥ threshold else "ANA-Negative"
confidence = p if Positive else (1 − p)
```

**Why** — Neural networks are usually **overconfident**: they say "99%" even when they're right only 85% of the time. A clinician will read "confidence 95%" literally, so it must mean "right about 95 times out of 100". Temperature scaling fixes this cheaply without changing which label is predicted.

**Threshold (decided):** start with **0.5**. Later we may test a sensitivity-oriented threshold, since for a *screening* test missing a positive is usually worse than a false alarm — and the baseline does miss more positives (sensitivity 0.928) than it wrongly flags negatives (specificity 0.971).
- Temperature scaling does **not** change any label at the 0.5 cut-off (dividing a logit by `T` keeps its sign); it only changes the confidence number.
- If the cut-off is ever changed, the new value must be chosen on **validation** data and only then reported on test.

**Input** — the `predictions.csv` files saved in Phase 5/6 (validation logits to learn `T`, test logits to check it). No retraining needed.

**How (status: done)** — `notebooks/03_calibrate.ipynb`, no GPU, under a minute:
1. For each round, find the `T` that minimises **NLL** on that round's validation logits (each round has its own model, so its own `T`).
2. Check on that round's test logits: ECE and NLL before vs after. An assertion confirms no label changes.
3. Pool all 2,073 test predictions (every image is a test image exactly once) → overall accuracy vs average confidence, per-bin table, reliability diagram.
4. Save `temperature.json` next to each round's `best.pt`; tables and the diagram go to Drive `outputs/calibration/`.
5. *Information only:* also fit a temperature **plus a shift** `b` (`p = sigmoid(logit/T + b)`). Training used `pos_weight ≈ 0.39`, which can push all logits towards "Negative" — the likely cause of sensitivity < specificity. Temperature cannot fix a shift. A shift changes labels (it is equivalent to a different cut-off), so it is **not adopted**, only measured.

Caveat: `T` is learned on validation images that were also used to pick the best epoch, so the model looks slightly better there than on unseen data; `T` may therefore under-correct. The test check shows by how much.

**Results (run 1 Oct 2026, EfficientNet-B0; files in `outputs/calibration/`)**

| Round | T | Val ECE before → after | Test ECE before → after | Test NLL before → after |
|---|---|---|---|---|
| 0 | 1.78 | 0.031 → 0.040 | 0.072 → 0.063 | 0.437 → 0.270 |
| 1 | 1.75 | 0.022 → 0.021 | 0.065 → 0.048 | 0.260 → 0.173 |
| 2 | 1.69 | 0.022 → 0.013 | 0.045 → 0.042 | 0.236 → 0.166 |
| 3 | 2.38 | 0.040 → 0.036 | 0.047 → 0.021 | 0.215 → 0.141 |
| 4 | 2.84 | 0.052 → 0.036 | 0.017 → 0.037 | 0.058 → 0.077 |
| **Mean** | **2.09** | **0.034 → 0.029** | **0.049 → 0.042** | **0.241 → 0.165** |

All 2,073 test predictions pooled (plain accuracy 0.940 in both rows — no label changes):

| | Average confidence | Gap to accuracy | ECE | NLL | Images at ≥ 99% confidence | of which wrong |
|---|---|---|---|---|---|---|
| Before | 0.982 | +0.042 | 0.042 | 0.241 | 1,819 | 42 (2.3%) |
| After | 0.960 | +0.020 | 0.025 | 0.165 | 1,456 | 9 (0.6%) |

- `T > 1` in every round → the raw model was overconfident, as expected.
- Scaling halves the overconfidence (gap 4.2 → 2.0 points) but does not remove it — the under-correction predicted in the caveat above.
- The clearest gain is at the top: a "≥ 99%" prediction used to be wrong 2.3% of the time, now 0.6%.
- Round 4 gets worse on test (its test set happened to be very easy, so the raw model was already honest there). Per-round changes are noisy; the pooled numbers are the reliable ones.
- The middle bins of the reliability diagram hold only 10–60 images each, so the zig-zag there is noise.

**Shift check (information only, not adopted)** — temperature + shift `b`, both fitted on validation:

| | Sensitivity | Specificity | Balanced acc. | Test ECE |
|---|---|---|---|---|
| Temperature only (cut-off 0.5) | 0.928 | 0.971 | 0.949 | 0.042 |
| Temperature + shift | 0.955 | 0.941 | 0.948 | 0.035 |

- `b` is positive in all 5 rounds (0.98 – 2.07, mean 1.36) → the model does lean towards "Negative", consistent with the `pos_weight` explanation.
- The shift changes 58 of 2,073 labels: about 41 more positives caught, about 17 more false alarms. Balanced accuracy is unchanged; it is a trade, not an improvement.
- The shift fitted here matches AIDA's own class mix (72% positive images), which another lab would not share.

**Key terms**
- **Calibration** — how well predicted confidence matches actual correctness.
- **Reliability diagram** — plot of "predicted confidence" vs "actual accuracy"; perfect calibration is the diagonal.
- **ECE (Expected Calibration Error)** — average gap between confidence and accuracy; lower is better.
- **Temperature scaling** — dividing logits by a learned `T > 1` to soften overconfident outputs.
- **NLL (negative log-likelihood, log loss)** — a score that punishes confident mistakes heavily; minimised to find `T`. Lower is better.
- **Threshold** — the probability cut-off for calling an image Positive.

**Done when** — ECE on validation is lower after scaling, and `T` is saved with the model.

---

### Phase 8 — Final evaluation (5-fold results)

**What** — For the chosen, calibrated model, collect each fold's test predictions and report (as mean ± std over 5 folds):
- Confusion matrix (all 5 test folds combined)
- Sensitivity, specificity, precision, F1, **balanced accuracy**, **ROC-AUC**, Cohen's kappa
- ECE + reliability diagram
- A few examples of wrong predictions, inspected by eye (Grad-CAM heat-maps deferred — decided 1 Oct 2026)
- Sensitivity/specificity at several cut-offs (information only; 0.5 stays the reported one)

**Why these metrics**
- **Plain accuracy is misleading** with imbalance (72% by always saying Positive).
- **Sensitivity** — of all truly positive images, how many we caught. Missing positives is the costly error in screening.
- **Specificity** — of all truly negative images, how many we correctly cleared.
- **Balanced accuracy** — average of sensitivity and specificity; fair under imbalance.
- **ROC-AUC** — how well the model ranks positives above negatives across *all* thresholds (1.0 perfect, 0.5 random).
- **Cohen's kappa** — agreement beyond chance; the proposal promises kappa, and the literature (Wu, Mayr) reports it.

**Key terms**
- **Confusion matrix** — 2×2 table: True Positive, False Negative, False Positive, True Negative.
- **Grad-CAM** — a heat-map showing which parts of the image the model looked at; lets us check it looks at cells, not background or artefacts.
- **Domain shift** — performance drop when data comes from a different source (another lab, microscope or dataset).

**Done when** — final metrics table and figures saved in `outputs/`.

---

### Phase 9 — Prediction script

**What** — `predict.py` that takes a path to any image (any format/size), applies the **same** preprocessing as training, and prints:
```
ANA-Positive — confidence 94.3%
```
Optionally a folder mode that writes a CSV of results.

**Why** — This is the piece the future web application (FastAPI backend, FYP-II) will call. Using identical preprocessing in training and prediction is critical; a mismatch silently ruins accuracy.

**Done when** — works on a few test images and on an image in a format the model hasn't seen.

---

### Phase 10 — Write-up

Short results section for the FYP-I report: dataset cleaning summary, split table, model comparison, final metrics, calibration plot, limitations. Following the proposal's style (concise, bullets, IEEE references; cite Benammar Elgaaied et al. 2016 for AIDA as required by the licence).

---

## 4. Decisions summary (and alternatives we rejected)

| Decision | Chosen | Alternative | Why chosen |
|---|---|---|---|
| Framework | PyTorch | TensorFlow/Keras | Team choice; standard in research, `timm` model library |
| Hardware | Colab T4 (16 GB) | Local GPU | Available now; fine for public data (clinical data must stay local later) |
| Data format | Generic manifest (`image_path, sample_id, label, source`) | Keep AIDA metadata (camera etc.) | Works for any future dataset |
| Duplicate names | Keep `.png` for well 281; keep the labelled `.jpg` for 320/433 | Keep both | Decided 1 Oct 2026; extra copies are different pictures or unlabelled |
| Mistyped rows | Recover the forgotten file with its well's label | Drop them | All images of a well share one label; saves 11 images |
| Stored image | Full 512-px-tall PNG, uncropped | Save pre-cropped images | Keeps all content; crop choice stays flexible |
| Model input | 512×512 square crop (random in training, centre at test) | 384 crop (sees only 42%) / squashed full image | 75% of the image at full detail; square hides image width |
| Split unit | Well (sample) | Image | Prevents leakage |
| Validation choice | numpy `RandomState` | scikit-learn shuffle | Same split on every machine/library version |
| Evaluation | 5-fold CV (mean ± std) | Single 70/15/15 split | Every well tested once; less dependent on luck |
| Output | 1 logit + sigmoid | 2-class softmax | Equivalent for binary, simpler to calibrate/threshold |
| Imbalance | Class-weighted loss | Oversampling / undersampling | Uses all data, no duplicated images, simple |
| Intensity handling | Keep raw brightness | Per-image contrast normalisation | Brightness *is* the label signal |
| Backbone | Compare 3, default EfficientNet-B0 | Only one model | FYP-I promises an architecture comparison |
| Confidence | Temperature-scaled probability | Raw softmax | Raw scores are overconfident |
| Main metric | Balanced accuracy + AUC | Accuracy | Accuracy misleading under imbalance |

---

## 5. Risks and how we handle them

| Risk | Handling |
|---|---|
| Model learns image-level shortcuts (size, format) instead of fluorescence | Uniform resize + PNG conversion; **always a square crop** (width 686 would reveal the 111 all-positive images) |
| Weak positives look like negatives | Keep raw brightness; review misclassified images; threshold may be tuned for sensitivity later (on validation) |
| Small dataset → noisy numbers | 5-fold cross-validation, report mean ± std |
| Overfitting (train loss → 0.001) | Early stopping keeps the best validation epoch; augmentation; weight decay |
| Test set influencing decisions | Models, cut-offs and temperature are chosen on validation only; test is reported, never used to decide |
| Edge content unseen at test time (centre crop hides ~12% each side) | Accepted for now; option D (left + right crops averaged) is the planned upgrade if errors point to it |
| Colab session timeouts | Best model + predictions saved to Drive per round; finished rounds are skipped on re-run |
| Different library versions giving different results | Folds stored in the manifest; validation picked with numpy `RandomState` |
| Licence breach | Data only in private Drive; never pushed to GitHub or shared publicly |
| Results may not transfer to other labs' images | Stated as a limitation of this stage (AIDA only) |

---

## 6. Decisions log and open points

**Decided (1 Oct 2026)**
1. Threshold: **0.5** to start.
2. Negative wells with 3 images: **keep all images**.
3. Evaluation: **5-fold cross-validation**, split by well.
4. Grad-CAM: **not for now**.
5. **All AIDA images are kept** (including every camera's images).
6. **No metadata columns** (camera, microscope, age, sex, pixel size) are used anywhere — only `image_path, sample_id, label, source`, so the pipeline works for any dataset.
7. Scope: **AIDA only** for now; testing on other datasets (Indus) is left for much later.
8. Well 281: keep the `.png` images, drop the `.tif` ones.
9. Model input: **one 512×512 crop** (~75% of the image). Option D (two crops covering 100%) is deferred.
10. Baseline model: EfficientNet-B0; training limits 30 epochs, patience 7.
11. **Backbone: EfficientNet-B0.** All three models tied on validation and test; chosen as the smallest and fastest. Validation is the selection criterion; test scores are shown alongside, never used to choose.

**Still open**
12. **Final model for `predict.py`:** 5-fold CV gives 5 trained models. Either average all 5 (an *ensemble*, usually more accurate and better calibrated) or retrain one model on all data. Decide after Phase 8. **For the website demo (2 Oct 2026):** a single model, `efficientnet_b0/round_1/best.pt` with `T = 1.754`, picked on validation (joint-highest balanced accuracy 0.982, highest AUC 0.999). Integration instructions for the FastAPI backend are in `Docs/Website_Integration_Prompt.md`. Phases 8–10 are paused until after the demo.
13. **Shift / cut-off:** whether to correct the model's lean towards "Negative" (equivalent to a lower cut-off, chosen on validation). Measured in Phase 7: sensitivity 0.928 → 0.955, specificity 0.971 → 0.941, balanced accuracy unchanged. Awaiting a decision.

**Possible improvements noted along the way (not adopted yet)**
- Pick the best epoch by validation **loss** instead of balanced accuracy at 0.5 (smoother, less noisy).
- A lower cut-off to raise sensitivity (must be chosen on validation).
- Option D: average left + right crops at test time to see 100% of the image.

**Honest caveats for the report**
- Phase 2 (EDA) looked at all images, including ones later used as test images. It informed only general design choices (no contrast stretching, 512 px, square crops), not anything tuned to raise test scores.
- Training settings (learning rate, epochs, patience) are untuned defaults.
- All results are on AIDA only; performance on another lab's images is unknown.

---

## 7. Glossary (quick reference)

| Term | Meaning |
|---|---|
| ANA / IIF / HEp-2 | Antinuclear antibody test, done by indirect immunofluorescence on HEp-2 cells |
| Well / sample | One patient serum sample on the slide; has 1–6 images here (`sample_id`) |
| Manifest / adapter | Standard label table / dataset-specific code that produces it |
| k-fold cross-validation | Train k times, each time testing on a different 1/k of the data |
| CNN | Convolutional Neural Network — a network that learns image features with sliding filters |
| Backbone | The main feature-extracting part of a CNN (e.g. EfficientNet-B0) |
| Pretrained / ImageNet | Weights already learned on 1.2 M everyday images |
| Transfer learning / fine-tuning | Reusing a pretrained network and training it further on our data |
| Epoch / batch / learning rate | Full data pass / group of images per update / update step size |
| Loss (BCE) | How wrong the model is; Binary Cross-Entropy for yes/no tasks |
| Logit / sigmoid | Raw output / function mapping it to a 0–1 probability |
| Augmentation | Random label-preserving changes to training images |
| Overfitting | Memorising training data, failing on new data |
| Early stopping / patience | Stop when validation stops improving / how many epochs we wait first |
| Parameters / GFLOPs | Numbers the model learns / billions of calculations per image (cost) |
| On-the-fly augmentation | Random changes applied when an image is loaded, not saved to disk |
| Test-time augmentation (TTA) | Predicting on several views of one image and averaging |
| Selection bias | A score looks too good because we picked the best of many tries on that same data |
| Deterministic | Same input always gives the same output (centre crop) |
| Reproducible | Someone else re-running the code gets the same result (fixed seed) |
| Lossless / LANCZOS | Stores pixels exactly (PNG) / high-quality resize filter |
| Percentile | Value below which a given % of the data falls (99th = brightest 1%) |
| Assertion | A check in code that stops the program if something is wrong |
| Train / val / test | Learn / tune / final honest check |
| Data leakage | Test information leaking into training |
| Stratified / group split | Keep class ratios / keep a group together |
| Class imbalance / class weight | One class much more common / extra penalty for rare-class mistakes |
| Confounder / shortcut | Spurious feature correlated with the label |
| Domain shift | Performance drop on data from a new source |
| Sensitivity (recall) | TP ÷ (TP + FN) — positives caught |
| Specificity | TN ÷ (TN + FP) — negatives correctly cleared |
| Precision | TP ÷ (TP + FP) — how many "Positive" calls were right |
| F1 | Harmonic mean of precision and recall |
| Balanced accuracy | (Sensitivity + Specificity) ÷ 2 |
| ROC-AUC | Ranking quality across all thresholds |
| Cohen's kappa | Agreement corrected for chance |
| Calibration / ECE | Confidence matching reality / its error measure |
| Temperature scaling | Dividing logits by T to fix overconfidence |
| Grad-CAM | Heat-map of where the model looked |
| Mixed precision (AMP) | Using 16-bit maths where safe for faster GPU training |
