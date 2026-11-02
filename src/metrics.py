"""Screening metrics for the multi-label ECG models.

The models are judged as a screening step, not a diagnosis. For each label:

1. Pick the decision threshold on the validation split so that the model
   catches a target share of true cases (sensitivity, 90% by default).
2. Apply that fixed threshold to the test split and report what a clinic
   would see: sensitivity, specificity, PPV (share of flagged patients who
   really have the condition), NPV, and referral volume (share of patients
   flagged for an echocardiogram).

AUROC, AUPRC and the Brier score (calibration) are reported alongside.
AUPRC should be read against the prevalence, which is its chance level.

The functions take predicted probabilities and labels only, so the
waveform model and the measurement model are scored the same way.
Missing (NaN) labels are skipped.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve


def threshold_for_sensitivity(y_true, y_prob, target=0.90):
    """Highest threshold whose sensitivity on (y_true, y_prob) is at least target.

    Use validation data here; the threshold is then fixed for the test set.
    """
    _, tpr, thresholds = roc_curve(y_true, y_prob)
    # roc_curve lists thresholds from high to low, so sensitivity (tpr) only
    # rises; the first one that reaches the target is the strictest that works.
    idx = int(np.argmax(tpr >= target))
    return float(thresholds[idx])


def screening_stats(y_true, y_prob, threshold):
    """Confusion-matrix figures for a fixed threshold (flag when y_prob >= threshold)."""
    y_true = np.asarray(y_true).astype(int)
    flagged = np.asarray(y_prob) >= threshold
    tp = int(np.sum(flagged & (y_true == 1)))
    fp = int(np.sum(flagged & (y_true == 0)))
    fn = int(np.sum(~flagged & (y_true == 1)))
    tn = int(np.sum(~flagged & (y_true == 0)))
    n = len(y_true)

    def ratio(a, b):
        return a / b if b else float("nan")

    return {
        "sensitivity": ratio(tp, tp + fn),
        "specificity": ratio(tn, tn + fp),
        "ppv": ratio(tp, tp + fp),
        "npv": ratio(tn, tn + fn),
        "referral_volume": ratio(tp + fp, n),
        # The same result per 1,000 patients screened, for the report text.
        "referred_per_1000": 1000 * ratio(tp + fp, n),
        "found_per_1000": 1000 * ratio(tp, n),
        "missed_per_1000": 1000 * ratio(fn, n),
    }


def ppv_at_prevalence(sensitivity, specificity, prevalence):
    """PPV the same model would reach in a population with another prevalence.

    EchoNext comes from a hospital, where disease is more common than in a
    clinic screening population, so its PPV is optimistic. Bayes' rule gives
    the PPV at an assumed screening prevalence.
    """
    true_pos = sensitivity * prevalence
    false_pos = (1 - specificity) * (1 - prevalence)
    return true_pos / (true_pos + false_pos)


def screening_report(val_probs, val_targets, test_probs, test_targets, labels,
                     target_sensitivity=0.90, screening_prevalence=None):
    """One row per label: ranking metrics plus screening figures on the test split.

    Args:
        val_probs, val_targets: arrays of shape (N_val, n_labels).
        test_probs, test_targets: arrays of shape (N_test, n_labels).
        labels: label names, in column order.
        target_sensitivity: sensitivity the threshold must reach on validation.
        screening_prevalence: optional assumed prevalence in a clinic, either
            one number for all labels or a dict {label: prevalence}. Adds a
            column with the PPV expected at that prevalence.
    """
    rows = []
    for j, name in enumerate(labels):
        v_keep = ~np.isnan(val_targets[:, j])
        t_keep = ~np.isnan(test_targets[:, j])
        yv, pv = val_targets[v_keep, j], val_probs[v_keep, j]
        yt, pt = test_targets[t_keep, j], test_probs[t_keep, j]

        threshold = threshold_for_sensitivity(yv, pv, target_sensitivity)
        row = {
            "label": name,
            "n_test": len(yt),
            "prevalence": float(yt.mean()),
            "auroc": roc_auc_score(yt, pt),
            "auprc": average_precision_score(yt, pt),
            "brier": brier_score_loss(yt, pt),
            "threshold": threshold,
        }
        # Sensitivity on test can land slightly off the target, because the
        # threshold was chosen on validation data. That gap is reported as is.
        row.update(screening_stats(yt, pt, threshold))

        if screening_prevalence is not None:
            prev = (screening_prevalence.get(name) if isinstance(screening_prevalence, dict)
                    else screening_prevalence)
            row["ppv_at_screening_prev"] = ppv_at_prevalence(row["sensitivity"], row["specificity"], prev)
        rows.append(row)
    return pd.DataFrame(rows).set_index("label")


if __name__ == "__main__":
    # Smoke test on synthetic predictions: positives score higher on average,
    # so the metrics should be clearly better than chance. No real data used.
    rng = np.random.default_rng(0)
    labels = ["StructuralDisease", "ReducedEF", "ValvularDisease", "LVH"]

    def fake_split(n):
        y = (rng.random((n, len(labels))) < 0.15).astype(float)
        probs = 1 / (1 + np.exp(-(2.0 * y - 1.5 + rng.normal(0, 1, y.shape))))
        y[rng.random(y.shape) < 0.03] = np.nan  # a few missing labels
        return probs, y

    val_probs, val_y = fake_split(2000)
    test_probs, test_y = fake_split(2000)
    report = screening_report(val_probs, val_y, test_probs, test_y, labels,
                              target_sensitivity=0.90, screening_prevalence=0.05)
    pd.set_option("display.width", 200)
    print(report.round(3).T)
