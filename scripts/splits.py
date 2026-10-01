"""
How one cross-validation ROUND is split into train / validation / test.

Used locally (make_folds.py, to check the splits) and copied unchanged into
the Colab training notebook, so both places produce exactly the same sets.

Round k (k = 0..4):
    test       = all images with fold == k          (~20% of samples)
    validation = 1/6 of the remaining samples       (~13% of samples)
    train      = everything else                    (~67% of samples)
Every split is made by sample_id (patient), stratified by label.

Validation patients are picked with numpy's legacy RandomState, whose random
sequence is guaranteed identical across numpy versions -- so the laptop and
Colab always produce the SAME split (scikit-learn's shuffling can change
between versions, which is what happened with the first version of this file).
"""
import numpy as np

VAL_PARTS = 6      # validation = 1 of 6 parts of the non-test patients


def get_round(manifest, k, seed=42):
    """Return (train_df, val_df, test_df) for cross-validation round k."""
    test = manifest[manifest["fold"] == k]
    rest = manifest[manifest["fold"] != k]

    # one row per patient, in a fixed (sorted) order
    patients = rest.drop_duplicates("sample_id")[["sample_id", "label"]].sort_values("sample_id")
    rng = np.random.RandomState(seed + k)   # seed + k -> different, reproducible val set per round
    val_ids = []
    for label in (0, 1):                     # stratified: same share of each class
        ids = patients.loc[patients["label"] == label, "sample_id"].to_numpy()
        ids = ids[rng.permutation(len(ids))]
        val_ids.extend(ids[: round(len(ids) / VAL_PARTS)])

    is_val = rest["sample_id"].isin(val_ids)
    return rest[~is_val], rest[is_val], test
