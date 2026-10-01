"""
Phase 4 -- assign every image to one of 5 cross-validation folds.

Rules
  * split by sample_id: all images of one patient land in the SAME fold
  * stratified by label: every fold has about the same Positive/Negative mix
  * fixed seed: running this again gives exactly the same folds

Adds a 'fold' column (0-4) to data/manifest.csv, prints/saves a report,
verifies there is no patient overlap, then rebuilds AIDA_processed.zip.

Run:  python scripts/make_folds.py
"""
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

import config
from preprocess import build_zip
from splits import get_round


def describe(df):
    """'1234 img (P 890 / N 344) | 600 samples (P 280 / N 320)'"""
    s = df.drop_duplicates("sample_id")
    return (f"{len(df):5d} img (P {int(df.label.sum()):4d} / N {int((df.label == 0).sum()):3d}) | "
            f"{len(s):3d} samples (P {int(s.label.sum()):3d} / N {int((s.label == 0).sum()):3d})")


def main():
    manifest = pd.read_csv(config.MANIFEST_PATH)[config.MANIFEST_COLUMNS]

    # ------------------------------------------------ assign folds
    sgkf = StratifiedGroupKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED)
    manifest["fold"] = -1
    for k, (_, test_idx) in enumerate(sgkf.split(manifest, manifest["label"], groups=manifest["sample_id"])):
        manifest.loc[manifest.index[test_idx], "fold"] = k

    # ------------------------------------------------ checks
    assert (manifest["fold"] >= 0).all(), "some image got no fold"
    folds_per_sample = manifest.groupby("sample_id")["fold"].nunique()
    assert (folds_per_sample == 1).all(), "LEAKAGE: a patient is split across folds"

    report = ["FOLDS (each is the test set of one round)"]
    for k in range(config.N_FOLDS):
        report.append(f"  fold {k}: {describe(manifest[manifest.fold == k])}")

    report.append("\nROUNDS (train / validation / test)")
    for k in range(config.N_FOLDS):
        train, val, test = get_round(manifest, k, config.SEED)
        sets = [set(x["sample_id"]) for x in (train, val, test)]
        assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]), \
            f"LEAKAGE in round {k}: a patient appears in two sets"
        assert len(train) + len(val) + len(test) == len(manifest)
        report.append(f"  round {k}")
        for name, part in (("train", train), ("val", val), ("test", test)):
            report.append(f"    {name:5s} {describe(part)}")
    report.append("\nChecks passed: every patient in exactly one fold; "
                  "no patient shared between train/val/test in any round.")

    # ------------------------------------------------ save + rebuild zip
    manifest.to_csv(config.MANIFEST_PATH, index=False)
    report.append(build_zip())
    (config.OUTPUTS_DIR / "phase4_folds_log.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
