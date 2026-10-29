# ECG to Structural Heart Disease Screening

Structural heart disease is usually confirmed by echocardiography, which is expensive and not available everywhere. An ECG costs almost nothing by comparison. This project asks how much of the echo finding can be recovered from the 12-lead ECG alone, and whether the resulting model is good enough to work as a **low-cost triage step**. It is not meant to replace diagnosis.

## Objectives

- Predict four echo-confirmed targets from the 12-lead ECG:
  - Any structural heart disease
  - Reduced left-ventricular ejection fraction
  - Valvular disease
  - Left-ventricular hypertrophy (LVH)
- Treat the task as **multi-label** and report results for each condition separately.
- Frame the evaluation as **screening**: fix a sensitivity a clinic would accept (e.g. 90%), then report the resulting **positive predictive value (PPV)** and **referral volume**.
- Compare a **waveform model** (deep learning on the raw signal) against a **feature model** built on standard ECG measurements (intervals, axes, amplitudes).

## Dataset

**EchoNext** ([PhysioNet](https://physionet.org/)): about 100,000 12-lead ECGs paired with labels confirmed by echocardiography.

Access requires a PhysioNet credentialed account and acceptance of the data use agreement. Raw data is **not** committed to this repository; place it under `data/` locally.

## Approach

| Component | Waveform model | Measurement model |
|---|---|---|
| Input | Raw 12-lead signal | Standard ECG measurements + age/sex |
| Model | 1D CNN / ResNet | Gradient boosting / logistic regression |
| Output | 4 sigmoid heads (multi-label) | One classifier per label |

**Evaluation (per label):**
- AUROC and AUPRC
- Operating threshold chosen on validation data to hit the target sensitivity
- At that threshold, on the test set: PPV, specificity, and referral volume (the share of patients flagged for echo)
- Calibration
- Patient-level train/val/test splits, so no patient appears in more than one split

## Setup

```bash
pip install -r requirements.txt
```

## Repository structure

```
data/        # local only, not tracked (EchoNext files)
notebooks/   # exploration and analysis
src/         # data loading, models, training, evaluation
results/     # metrics tables and figures
```

## Disclaimer

This is a research project. It is not a medical device and must not be used for clinical decisions.
