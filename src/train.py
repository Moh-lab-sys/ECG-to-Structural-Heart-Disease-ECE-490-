"""Training loop for the waveform model.

Trains ECGResNet on multi-label ECG data and keeps the weights that score
best on the validation split. Two details matter for this dataset:

- Class imbalance: each condition is much rarer than its absence, so
  positives get a higher weight in the loss (pos_weight = negatives /
  positives per label, computed on the training split only).
- Missing labels: a label can be missing (NaN) for some ECGs. Those entries
  are left out of the loss instead of being counted as negatives.

Model selection uses the mean AUROC over the labels on the validation split.
The test split is never used here.
"""

import copy
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from waveform_model import LABELS, ECGResNet


def set_seed(seed=42):
    """Fix the random seeds so a run can be repeated."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ECGDataset(Dataset):
    """ECG waveforms, labels and (optionally) tabular features.

    Args:
        waveforms: array of shape (N, 12, n_samples). Can be a memory-mapped
            file from np.load(..., mmap_mode="r"): ECGs are read one at a
            time, so the full set never has to fit in RAM.
        labels: array of shape (N, n_labels) with values 0, 1 or NaN (missing).
        tabular: optional array of shape (N, n_tabular), already scaled.
    """

    def __init__(self, waveforms, labels, tabular=None):
        self.x = waveforms
        self.y = np.asarray(labels, dtype=np.float32)
        self.tab = None if tabular is None else np.asarray(tabular, dtype=np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = torch.from_numpy(np.array(self.x[i], dtype=np.float32))
        y = torch.from_numpy(self.y[i])
        # An empty tensor stands for "no tabular input" so batching still works.
        tab = torch.from_numpy(self.tab[i]) if self.tab is not None else torch.empty(0)
        return x, tab, y


def compute_pos_weight(labels):
    """Weight of the positive class for each label: negatives / positives.

    Missing (NaN) labels are ignored. Pass the training labels only.
    """
    labels = np.asarray(labels, dtype=np.float32)
    pos = np.nansum(labels, axis=0)
    neg = np.sum(labels == 0, axis=0)
    return torch.tensor(neg / np.maximum(pos, 1), dtype=torch.float32)


def masked_bce(logits, targets, pos_weight):
    """Binary cross-entropy on logits that skips missing (NaN) labels."""
    mask = ~torch.isnan(targets)
    # Missing labels get a placeholder 0; the mask then removes their loss.
    loss = F.binary_cross_entropy_with_logits(
        logits, torch.nan_to_num(targets), pos_weight=pos_weight, reduction="none")
    return (loss * mask).sum() / mask.sum().clamp(min=1)


def run_epoch(model, loader, pos_weight, device, optimizer=None):
    """One pass over the data: trains if an optimiser is given, else evaluates.

    Returns the mean loss, the predicted probabilities and the labels.
    """
    training = optimizer is not None
    model.train(training)
    total_loss, n_batches, probs, targets = 0.0, 0, [], []
    with torch.set_grad_enabled(training):
        for x, tab, y in loader:
            x, y = x.to(device), y.to(device)
            tab = tab.to(device) if tab.shape[1] > 0 else None
            logits = model(x, tab)
            loss = masked_bce(logits, y, pos_weight)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            n_batches += 1
            probs.append(torch.sigmoid(logits).detach().cpu())
            targets.append(y.cpu())
    return total_loss / max(n_batches, 1), torch.cat(probs).numpy(), torch.cat(targets).numpy()


def per_label_auroc(probs, targets):
    """AUROC for each label, ignoring missing labels.

    A label with only one class present gets NaN, since AUROC is undefined.
    """
    scores = []
    for j in range(targets.shape[1]):
        keep = ~np.isnan(targets[:, j])
        y, p = targets[keep, j], probs[keep, j]
        scores.append(roc_auc_score(y, p) if len(np.unique(y)) == 2 else float("nan"))
    return np.array(scores)


def fit(model, train_ds, val_ds, epochs=30, batch_size=64, lr=1e-3, weight_decay=1e-4,
        patience=5, save_path="results/waveform_best.pt", num_workers=0):
    """Train the model and keep the weights with the best mean validation AUROC.

    Stops early once the validation AUROC has not improved for `patience`
    epochs. Saves the best weights to save_path and loads them back into the
    model at the end. Returns the per-epoch history.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    pos_weight = compute_pos_weight(train_ds.y).to(device)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    best_auroc, best_state, bad_epochs, history = -1.0, None, 0, []
    for epoch in range(1, epochs + 1):
        train_loss, _, _ = run_epoch(model, train_loader, pos_weight, device, optimizer)
        val_loss, val_probs, val_targets = run_epoch(model, val_loader, pos_weight, device)
        aurocs = per_label_auroc(val_probs, val_targets)
        mean_auroc = float(np.nanmean(aurocs))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "val_auroc": dict(zip(LABELS, aurocs.tolist())), "val_mean_auroc": mean_auroc})
        print(f"epoch {epoch:2d}  train loss {train_loss:.4f}  val loss {val_loss:.4f}  "
              f"val mean AUROC {mean_auroc:.3f}")

        if mean_auroc > best_auroc:
            best_auroc, bad_epochs = mean_auroc, 0
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, save_path)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"no improvement for {patience} epochs, stopping early")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


if __name__ == "__main__":
    # Smoke test on synthetic data: checks that the loop runs end to end and
    # that the model can pick up a planted signal. No real ECGs are used.
    set_seed(0)
    n, n_samples = 600, 2500
    waveforms = np.random.randn(n, 12, n_samples).astype(np.float32)
    labels = (np.random.rand(n, len(LABELS)) < 0.2).astype(np.float32)
    # Planted signal: label j is positive when lead j is shifted upwards.
    for j in range(len(LABELS)):
        waveforms[labels[:, j] == 1, j, :] += 0.5
    # Mark 5% of the labels as missing to exercise the masking.
    labels[np.random.rand(n, len(LABELS)) < 0.05] = np.nan
    age_sex = np.random.randn(n, 2).astype(np.float32)

    train_ds = ECGDataset(waveforms[:400], labels[:400], age_sex[:400])
    val_ds = ECGDataset(waveforms[400:], labels[400:], age_sex[400:])
    model = ECGResNet(n_tabular=2)
    fit(model, train_ds, val_ds, epochs=3, batch_size=32, save_path="results/smoke_test.pt")
