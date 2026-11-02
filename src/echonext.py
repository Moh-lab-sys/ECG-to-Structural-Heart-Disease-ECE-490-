"""Load the EchoNext files into datasets for the waveform model.

EchoNext ships, for each split (train, val, test, no_split), a waveform file
EchoNext_<split>_waveforms.npy of shape (N, 1, 2500, 12): a 10 s, 12-lead
ECG at 250 Hz. The dataset authors already median-filtered each lead,
clipped it at the 0.1st and 99.9th percentiles and normalised it, so no
further signal processing is done here.

One metadata table, EchoNext_metadata_100k.csv, covers all splits through
its `split` column. Its rows for a split, in file order, line up with the
rows of that split's waveform file.

This module:
1. picks the metadata rows of a split and checks they match the waveform file;
2. maps the 12 echo flags to the 4 project labels (docs/preprocessing.md,
   section 1), leaving a label missing (NaN) when it cannot be decided;
3. turns age and sex into model inputs, scaling age with training-split
   statistics only;
4. reorders each ECG to (12, 2500), the layout ECGResNet expects.

The no_split ECGs are never loaded. Waveform files are memory-mapped, so
ECGs are read from disk one at a time instead of all at once.
"""

import os

import numpy as np
import pandas as pd

from train import ECGDataset
from waveform_model import LABELS

METADATA_FILE = "EchoNext_metadata_100k.csv"
WAVEFORM_FILE = "EchoNext_{split}_waveforms.npy"

# Echo flags behind each project label (docs/preprocessing.md, section 1).
LABEL_SOURCES = {
    "StructuralDisease": [
        "shd_moderate_or_greater_flag",
        "rv_systolic_dysfunction_moderate_or_greater_flag",
        "pasp_gte_45_flag",
        "pericardial_effusion_moderate_large_flag",
        "tr_max_gte_32_flag",
    ],
    "ReducedEF": ["lvef_lte_45_flag"],
    "ValvularDisease": [
        "aortic_stenosis_moderate_or_greater_flag",
        "aortic_regurgitation_moderate_or_greater_flag",
        "mitral_regurgitation_moderate_or_greater_flag",
        "tricuspid_regurgitation_moderate_or_greater_flag",
        "pulmonary_regurgitation_moderate_or_greater_flag",
    ],
    "LVH": ["lvwt_gte_13_flag"],
}


def find_file(data_dir, name):
    """Path of `name` inside data_dir, ignoring upper/lower case.

    Colab runs on Linux, where file names are case-sensitive.
    """
    for f in os.listdir(data_dir):
        if f.lower() == name.lower():
            return os.path.join(data_dir, f)
    raise FileNotFoundError(f"{name} not found in {data_dir}")


def make_labels(meta):
    """Map the 12 echo flags to the 4 labels: array of shape (N, 4) with 0, 1 or NaN.

    A label is 1 if any of its source flags is 1, and 0 if all of them are 0.
    If none is 1 but some are missing, the label is NaN (unknown), so the
    training loss skips it instead of counting it as a negative.
    """
    cols = []
    for name in LABELS:
        flags = meta[LABEL_SOURCES[name]]
        positive = (flags == 1).any(axis=1)
        label = positive.astype(float)
        label[~positive & flags.isna().any(axis=1)] = np.nan
        cols.append(label.to_numpy())
    return np.stack(cols, axis=1).astype(np.float32)


def age_sex(meta, age_mean, age_std, age_median):
    """Age and sex as model inputs, in the order of TABULAR_FEATURES.

    Age is standardised with the training-split mean and standard deviation
    (a missing age gets the training median). Sex becomes 0 (female) or 1 (male).
    """
    age = meta["age_at_ecg"].fillna(age_median)
    codes = {"female": 0.0, "male": 1.0, "0": 0.0, "1": 1.0}
    sex = meta["sex"].astype(str).str.strip().str.lower().map(codes)
    if sex.isna().any():
        raise ValueError("unexpected values in the sex column: expected male/female")
    return np.stack([(age - age_mean) / age_std, sex], axis=1).astype(np.float32)


class LeadsFirst:
    """View of an (N, 1, 2500, 12) waveform array that returns each ECG as (12, 2500).

    Reorders one ECG at a time when it is read, so a memory-mapped file is
    never copied as a whole.
    """

    def __init__(self, waveforms):
        self.w = waveforms

    def __len__(self):
        return self.w.shape[0]

    def __getitem__(self, i):
        # (1, 2500, 12) -> (2500, 12) -> (12, 2500)
        return self.w[i, 0].T


def load_split(data_dir, split, meta):
    """Metadata rows and memory-mapped waveforms of one split, checked to line up."""
    rows = meta[meta["split"] == split].reset_index(drop=True)
    waves = np.load(find_file(data_dir, WAVEFORM_FILE.format(split=split)), mmap_mode="r")
    if waves.shape[0] != len(rows) or waves.shape[1:] != (1, 2500, 12):
        raise ValueError(f"{split}: {len(rows)} metadata rows but waveforms of shape {waves.shape}")
    return rows, waves


def build_datasets(data_dir, use_tabular=False):
    """ECGDatasets for train, val and test, ready for train.fit().

    With use_tabular=True, each ECG also carries scaled age and sex, for a
    model built with ECGResNet(n_tabular=2).
    """
    meta = pd.read_csv(find_file(data_dir, METADATA_FILE), low_memory=False)
    # Age statistics come from the training split only and are reused for val and test.
    train_age = meta.loc[meta["split"] == "train", "age_at_ecg"]
    age_stats = (train_age.mean(), train_age.std(), train_age.median())

    datasets = []
    for split in ("train", "val", "test"):
        rows, waves = load_split(data_dir, split, meta)
        tab = age_sex(rows, *age_stats) if use_tabular else None
        datasets.append(ECGDataset(LeadsFirst(waves), make_labels(rows), tab))
    return tuple(datasets)


if __name__ == "__main__":
    # Smoke test on fake files in the EchoNext format: checks file loading,
    # row alignment, label mapping and shapes. No real data is used.
    import tempfile

    rng = np.random.default_rng(0)
    sizes = {"train": 40, "val": 10, "test": 10, "no_split": 5}
    # Splits in mixed order, to check that rows are picked by the split column.
    splits = np.concatenate([[s] * n for s, n in sizes.items()])
    rng.shuffle(splits)
    meta = pd.DataFrame({
        "split": splits,
        "age_at_ecg": rng.integers(18, 91, len(splits)).astype(float),
        "sex": rng.choice(["male", "female"], len(splits)),
    })
    for col in sorted({c for cols in LABEL_SOURCES.values() for c in cols}):
        meta[col] = (rng.random(len(splits)) < 0.2).astype(float)
    meta.loc[meta.index[:8], "lvef_lte_45_flag"] = np.nan  # a few missing flags

    waves = {s: rng.standard_normal((n, 1, 2500, 12)).astype(np.float32) for s, n in sizes.items()}
    with tempfile.TemporaryDirectory() as d:
        meta.to_csv(os.path.join(d, METADATA_FILE), index=False)
        for s, w in waves.items():
            np.save(os.path.join(d, WAVEFORM_FILE.format(split=s)), w)

        train_ds, val_ds, test_ds = build_datasets(d, use_tabular=True)
        x, tab, y = train_ds[0]
        assert np.array_equal(x.numpy(), waves["train"][0, 0].T), "leads reordered wrongly"
        print(f"sizes: train {len(train_ds)}, val {len(val_ds)}, test {len(test_ds)}")
        print(f"one ECG {tuple(x.shape)}, age/sex {tuple(tab.shape)}, labels {tuple(y.shape)}")
        print(f"missing ReducedEF labels in train: {int(np.isnan(train_ds.y[:, 1]).sum())}")
        # Close the memory-mapped files before the folder is deleted (needed on Windows).
        del train_ds, val_ds, test_ds
