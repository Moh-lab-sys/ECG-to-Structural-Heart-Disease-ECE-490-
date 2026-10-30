# Preprocessing

This document records how the EchoNext metadata table is turned into model-ready features and labels for the **measurement model**. The waveform model's signal preprocessing will be documented separately.

Status key: ✅ done · 🔜 decided, not yet applied · ❓ open question

---

## 1. Target labels ✅

EchoNext ships 12 binary echo-derived flags. Each was mapped to one of the project's four targets.

| Source flag | Target |
|---|---|
| `shd_moderate_or_greater_flag` | StructuralDisease |
| `rv_systolic_dysfunction_moderate_or_greater_flag` | StructuralDisease |
| `pasp_gte_45_flag` | StructuralDisease |
| `pericardial_effusion_moderate_large_flag` | StructuralDisease |
| `tr_max_gte_32_flag` | StructuralDisease |
| `lvef_lte_45_flag` | ReducedEF |
| `aortic_stenosis_moderate_or_greater_flag` | ValvularDisease |
| `aortic_regurgitation_moderate_or_greater_flag` | ValvularDisease |
| `mitral_regurgitation_moderate_or_greater_flag` | ValvularDisease |
| `tricuspid_regurgitation_moderate_or_greater_flag` | ValvularDisease |
| `pulmonary_regurgitation_moderate_or_greater_flag` | ValvularDisease |
| `lvwt_gte_13_flag` | LVH |

Mapping notes:

- **StructuralDisease is not a peer of the other three.** `shd_moderate_or_greater_flag` is a composite of all the other findings, so every ReducedEF, ValvularDisease and LVH positive should also be a StructuralDisease positive.
- **`tr_max_gte_32_flag` is not a valvular label.** Despite the "TR" in its name, it is a jet velocity used to estimate pulmonary pressure, not a grade of tricuspid regurgitation, so it sits with `pasp_gte_45_flag`.
- **LVH is a wall-thickness proxy.** `lvwt_gte_13_flag` uses a single 1.3 cm cutoff, not LV mass index, and the same cutoff for both sexes.
- **MR and TR are often secondary** to LV or RV dysfunction, so ValvularDisease will correlate with ReducedEF.

Code:

```python
df['StructuralDisease'] = (
    (df['shd_moderate_or_greater_flag'] == 1) |
    (df['rv_systolic_dysfunction_moderate_or_greater_flag'] == 1) |
    (df['pasp_gte_45_flag'] == 1) |
    (df['pericardial_effusion_moderate_large_flag'] == 1) |
    (df['tr_max_gte_32_flag'] == 1)
).astype(int)

df['ReducedEF'] = (df['lvef_lte_45_flag'] == 1).astype(int)

df['ValvularDisease'] = (
    (df['aortic_stenosis_moderate_or_greater_flag'] == 1) |
    (df['aortic_regurgitation_moderate_or_greater_flag'] == 1) |
    (df['mitral_regurgitation_moderate_or_greater_flag'] == 1) |
    (df['tricuspid_regurgitation_moderate_or_greater_flag'] == 1) |
    (df['pulmonary_regurgitation_moderate_or_greater_flag'] == 1)
).astype(int)

df['LVH'] = (df['lvwt_gte_13_flag'] == 1).astype(int)
```

The 12 original flag columns were then dropped.

Caveats:

- `== 1` turns NaN into 0, so a missing flag becomes a negative. Check `isna().sum()` on the source flags.
- Some severity columns contain **"presumed none"**: the finding was not mentioned in the echo report and is assumed absent. These rows are negatives (0), consistent with the dataset's own labels, but they are weaker negatives than an explicit "none". A sensitivity analysis excluding them is planned (section 6).

## 2. Categorical encoding

| Column | Type | Encoding | Status |
|---|---|---|---|
| `sex` | Binary | Single 0/1 column | 🔜 verify values with `value_counts(dropna=False)` |
| `location_setting` | Unordered, 4 levels | One-hot: `emergency`, `inpatient`, `outpatient`, `procedural` | ✅ |
| `race_ethnicity` | Unordered, 6 levels | One-hot: `asian`, `black`, `hispanic`, `other`, `unknown`, `white` | ✅ |

```python
df = pd.get_dummies(df, columns=['location_setting', 'race_ethnicity'], dtype=int)
```

Notes:

- All categories were kept (no `drop_first`). That is fine for tree models, neural networks and regularized logistic regression.
- Missing race/ethnicity is kept as its own `unknown` category, not dropped.
- Encoding sex as one 0/1 column does not introduce bias: with two categories it carries the same information as one-hot. Sex bias is evaluated through per-sex results (section 6).
- **Location setting is a strong shortcut.** Inpatients and ED patients are much sicker than outpatients, so the model can partly predict disease from location alone. A clinic screening population is mostly outpatient.
- ❓ **Whether race/ethnicity is a model input is still open.** The leading option is to keep it out of the inputs and use it only for subgroup evaluation.

Ordinal severity columns (e.g. pericardial effusion) map in clinical order: `none` / `presumed none` = 0, `trace` = 1, `small` = 2, `moderate` = 3, `large` = 4. This is for exploration and label checks only, since these columns come from the echo and are excluded from model inputs (section 3).

## 3. Columns excluded from model inputs 🔜

| Column(s) | Reason |
|---|---|
| `aortic_stenosis_value`, `aortic_regurgitation_value`, `mitral_regurgitation_value`, `tricuspid_regurgitation_value`, `pulmonary_regurgitation_value`, `rv_systolic_function_value`, `pericardial_effusion_value`, `ivs_measurement`, `lvpw_measurement`, `pasp_value`, `tr_max_velocity_value`, `lvef_value` | **Label leakage.** These are echo measurements: the sources of the labels, unavailable at ECG screening time. |
| `acquisition_year` | Captures equipment and practice drift, not physiology, and extrapolates badly to future years. Used for a temporal split instead. |
| `most_recent_ecg` | Dataset bookkeeping, not physiology. Indicates patients can have multiple ECGs (see section 5). |
| `StructuralDisease`, `ReducedEF`, `ValvularDisease`, `LVH` | Targets. |

## 4. Feature scaling 🔜

Scaled with `StandardScaler`, fitted on the **training split only**:

`age_at_ecg`, `ventricular_rate`, `atrial_rate`, `pr_interval`, `qrs_duration`, `qt_corrected`

Not scaled: `sex`, the one-hot columns, and the targets.

```python
from sklearn.preprocessing import StandardScaler

scale_cols = ['age_at_ecg', 'ventricular_rate', 'atrial_rate',
              'pr_interval', 'qrs_duration', 'qt_corrected']

scaler = StandardScaler()
X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
X_val[scale_cols]   = scaler.transform(X_val[scale_cols])
X_test[scale_cols]  = scaler.transform(X_test[scale_cols])
```

Notes:

- Age stays **continuous**, not binned. It is available at screening time and is a genuine predictor.
- Scaling matters for linear models and neural networks (including the age + sex baseline). Tree models are unaffected.
- **Before scaling:** `pr_interval` and `atrial_rate` are undefined in atrial fibrillation, so their NaNs carry clinical meaning. Add a 0/1 missing indicator for each, then impute with the training-set median.

## 5. Splitting 🔜

- **Patient-level splits.** `most_recent_ecg` suggests multiple ECGs per patient. If a patient ID exists, split by patient; otherwise keep only `most_recent_ecg == 1` to get one ECG per patient.
- **Predefined split.** If EchoNext ships a split column, use it for comparability with published results.
- **Temporal validation.** Train on earlier `acquisition_year` values and test on later ones as an additional analysis.
- All encoders, imputers and scalers are fitted on the training split only.

## 6. Planned checks and analyses

- [ ] Confirm `shd_moderate_or_greater_flag` equals the OR of the other 11 flags.
- [ ] Count missing values in the source flags and in the six scaled features.
- [ ] Verify `sex` values before encoding.
- [ ] Decide whether race/ethnicity is a model input.
- [ ] **Age + sex baseline:** logistic regression on demographics only; the ECG models must clearly beat it.
- [ ] Give age and sex to both the waveform and measurement models, so the comparison is fair.
- [ ] Train the measurement model with and without `location_setting`; report the outpatient subset separately.
- [ ] Report sensitivity, PPV and referral volume per sex, per race/ethnicity group and per age group (< 65 vs. ≥ 65).
- [ ] Sensitivity analysis excluding "presumed none" negatives.
