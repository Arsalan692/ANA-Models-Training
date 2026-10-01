"""
Central settings for the ANA Positive/Negative project.

Every other script imports its paths and settings from here, so if a folder
moves or a setting changes we edit ONE file instead of hunting through many.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
# Project root = the "Second Model Training" folder (parent of scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Raw AIDA images + expert labels (read-only: we never modify these files)
AIDA_RAW_DIR = PROJECT_ROOT.parent / "Model Training" / "aida_project_database"
AIDA_GROUND_TRUTH = PROJECT_ROOT / "Docs" / "AIDA_HEp-2_GroundTruth.xlsx"

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"      # resized PNGs (Phase 3)
MANIFEST_PATH = DATA_DIR / "manifest.csv"   # clean label table (Phase 1)
OUTPUTS_DIR = PROJECT_ROOT / "outputs"      # plots, metrics, models

# ---------------------------------------------------------------- settings
SEED = 42              # fixed random seed -> reproducible splits/training
N_FOLDS = 5            # 5-fold cross-validation (decided 1 Oct 2026)
SHORT_SIDE = 512       # processed images: shorter side resized to 512 px
THRESHOLD = 0.5        # probability cut-off for "Positive" (decided 1 Oct 2026)

# Standard manifest columns -- identical for every dataset (design rule)
MANIFEST_COLUMNS = ["image_path", "sample_id", "label", "source"]
