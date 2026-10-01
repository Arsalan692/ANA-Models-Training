"""
Phase 1 -- AIDA adapter: expert spreadsheet  ->  clean data/manifest.csv

Reads AIDA_HEp-2_GroundTruth.xlsx, fixes its filename problems, matches each
row to a real image file, and writes our standard 4-column manifest:

    image_path, sample_id, label, source

This file is the ONLY place that knows AIDA's quirks. Every later phase reads
manifest.csv and never looks at the spreadsheet again.

Outputs
  data/manifest.csv          standard table (one row per image we will use)
  data/aida_file_map.csv     image_path -> original raw file (used by Phase 3)
  outputs/phase1_cleaning_log.txt   what was dropped/fixed and why

Run:  python scripts/prepare_manifest.py
"""
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

import config

SOURCE = "AIDA"


def split_name(filename):
    """'281_1.png' -> ('281_1', 'png').

    Also handles the spreadsheet's typos, where the dot before the extension
    was mistyped: '466_1_bmp' -> ('466_1', 'bmp'), '277_3,jpg' -> ('277_3', 'jpg').
    The 'stem' (name without extension) is what identifies an image.
    """
    name = str(filename).strip()
    match = re.match(r"^(.*)[.,_]([A-Za-z]{3,4})$", name)
    if match is None:
        return name.lower(), ""
    return match.group(1).lower(), match.group(2).lower()


def main():
    log = []  # human-readable record of every fix and drop

    # ------------------------------------------------ 1. read the expert labels
    sheet = pd.read_excel(config.AIDA_GROUND_TRUTH)
    sheet = sheet[["ImageFileName", "Fluorescence"]]  # nothing else is used
    n_total = len(sheet)

    empty = sheet["ImageFileName"].isna()
    sheet = sheet[~empty].copy()
    log.append(f"Spreadsheet rows: {n_total}; dropped {empty.sum()} empty rows -> {len(sheet)}")

    sheet["stem"], sheet["ext"] = zip(*sheet["ImageFileName"].map(split_name))

    # ------------------------------------------------ 2. index files on disk
    # stem -> {extension: actual filename}, e.g. '281_1' -> {'png': '281_1.png', 'tif': '281_1.tif'}
    disk = defaultdict(dict)
    for f in config.AIDA_RAW_DIR.iterdir():
        if f.is_file():
            stem, ext = split_name(f.name)
            disk[stem][ext] = f.name

    # ------------------------------------------------ 3. same image name listed twice
    # Two causes, both handled by keeping ONE row per image name:
    #  a) typing mistakes: e.g. '198_1.jpg' typed again where '198_4.jpg' was meant.
    #     The real file is recovered in step 5 using its well's label.
    #  b) well 281 is labelled as both .png and .tif (different pictures).
    #     Decision 1 Oct 2026: keep the .png version only.
    dup_stems = sheet["stem"][sheet["stem"].duplicated(keep=False)].unique()
    for stem in dup_stems:
        rows = sheet[sheet["stem"] == stem]
        keep = rows.index[rows["ext"] == "png"]
        keep = keep[0] if len(keep) else rows.index[0]
        dropped = rows.drop(index=keep)
        sheet = sheet.drop(index=dropped.index)
        log.append(f"Name listed twice: kept {rows.loc[keep, 'ImageFileName']}, "
                   f"dropped duplicate row(s) {', '.join(dropped['ImageFileName'])}")

    # ------------------------------------------------ 4. match each row to a file
    raw_files = []
    for _, row in sheet.iterrows():
        options = disk.get(row["stem"], {})
        if row["ext"] in options:              # exact extension exists on disk
            raw_files.append(options[row["ext"]])
        elif len(options) == 1:                # extension typo, but only one candidate
            actual = next(iter(options.values()))
            raw_files.append(actual)
            log.append(f"Filename typo fixed: '{row['ImageFileName']}' -> '{actual}'")
        else:
            raw_files.append(None)
            log.append(f"Missing on disk, dropped: '{row['ImageFileName']}'")
    sheet["raw_file"] = raw_files
    sheet = sheet[sheet["raw_file"].notna()].copy()

    # ------------------------------------------------ 5. files the sheet forgot to label
    # Every image of a well has the same label (checked in step 7), so an
    # unlabelled file like '198_4.jpg' safely inherits the label of well 198.
    # Not recovered: files whose name is already used by another format
    # (e.g. '281_1.tif' when '281_1.png' is kept) -- decision 1 Oct 2026.
    sheet["well"] = sheet["stem"].str.split("_").str[0]
    well_label = sheet.groupby("well")["Fluorescence"].first()
    used = set(sheet["raw_file"])
    used_stems = set(sheet["stem"])
    recovered = []
    for stem, exts in sorted(disk.items()):
        for name in sorted(exts.values()):
            if name in used:
                continue
            well = stem.split("_")[0]
            if stem not in used_stems and well in well_label.index:
                recovered.append({"ImageFileName": name, "Fluorescence": well_label[well],
                                  "stem": stem, "ext": split_name(name)[1],
                                  "raw_file": name, "well": well})
                used_stems.add(stem)
                log.append(f"Unlabelled file recovered with its well's label "
                           f"({well_label[well]}): '{name}'")
            else:
                log.append(f"Ignored (no label / other format kept): '{name}'")
    sheet = pd.concat([sheet, pd.DataFrame(recovered)], ignore_index=True)

    # ------------------------------------------------ 6. build the standard manifest
    manifest = pd.DataFrame({
        # where the processed PNG will live, relative to data/processed/ (Phase 3)
        "image_path": SOURCE + "/" + sheet["stem"] + ".png",
        # well number = the part before '_' ; prefixed so IDs never clash across datasets
        "sample_id": SOURCE + "_" + sheet["well"],
        "label": sheet["Fluorescence"].map({"Positive": 1, "Negative": 0}),
        "source": SOURCE,
    })[config.MANIFEST_COLUMNS]

    # ------------------------------------------------ 7. sanity checks (stop if wrong)
    assert manifest["label"].isin([0, 1]).all(), "unknown label value"
    assert manifest["image_path"].is_unique, "two rows point to the same image"
    assert all((config.AIDA_RAW_DIR / f).is_file() for f in sheet["raw_file"])
    per_sample = manifest.groupby("sample_id")["label"].nunique()
    assert (per_sample == 1).all(), "a sample has images with different labels"

    # ------------------------------------------------ 8. save
    config.DATA_DIR.mkdir(exist_ok=True)
    config.OUTPUTS_DIR.mkdir(exist_ok=True)
    manifest.to_csv(config.MANIFEST_PATH, index=False)
    pd.DataFrame({"image_path": manifest["image_path"], "raw_file": sheet["raw_file"]}) \
        .to_csv(config.DATA_DIR / "aida_file_map.csv", index=False)

    samples = manifest.drop_duplicates("sample_id")
    summary = [
        "",
        "================ SUMMARY ================",
        f"Images kept : {len(manifest)}  "
        f"(Positive {int((manifest.label == 1).sum())}, Negative {int((manifest.label == 0).sum())})",
        f"Samples     : {len(samples)}  "
        f"(Positive {int((samples.label == 1).sum())}, Negative {int((samples.label == 0).sum())})",
        f"Images per sample: {manifest.groupby('sample_id').size().value_counts().sort_index().to_dict()}",
    ]
    log += summary
    log_path = config.OUTPUTS_DIR / "phase1_cleaning_log.txt"
    log_path.write_text("\n".join(log), encoding="utf-8")

    print("\n".join(log))
    print(f"\nSaved: {config.MANIFEST_PATH}\nSaved: {log_path}")


if __name__ == "__main__":
    main()
