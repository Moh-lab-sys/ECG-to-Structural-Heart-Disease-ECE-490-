"""Waveform model for ECG -> structural heart disease screening.

A 1D residual convolutional network (ResNet) that reads a raw 12-lead ECG
and outputs one score per target label. The task is multi-label: a patient
can have several conditions at once, so each label gets its own independent
output instead of a softmax over classes.

Design follows residual networks (He et al., 2016), adapted to 1D signals
as in Ribeiro et al. (2020) for 12-lead ECG classification.

Input:  tensor of shape (batch, 12, n_samples), one row per lead.
Output: tensor of shape (batch, 4) of logits, one per label in LABELS.
        Apply torch.sigmoid() to turn them into probabilities.
"""

import torch
import torch.nn as nn

# Order of the output columns. Must match the label columns in the data.
LABELS = ["StructuralDisease", "ReducedEF", "ValvularDisease", "LVH"]


class ResBlock1d(nn.Module):
    """Residual block: two 1D convolutions plus a shortcut connection.

    The block learns a correction (the "residual") that is added to its
    input. The shortcut lets gradients flow straight through, which keeps
    deeper networks trainable.

    Args:
        in_ch: number of input channels.
        out_ch: number of output channels.
        stride: 2 halves the time length, 1 keeps it.
        kernel_size: width of each convolution, in samples.
        dropout: dropout rate between the two convolutions.
    """

    def __init__(self, in_ch, out_ch, stride=1, kernel_size=7, dropout=0.2):
        super().__init__()
        # Padding of kernel_size // 2 keeps the length unchanged when stride is 1.
        pad = kernel_size // 2
        # bias=False because the BatchNorm right after has its own shift term.
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

        # Shortcut path. If the block changes the length (stride) or the
        # number of channels, the input no longer matches the output shape,
        # so a 1x1 convolution reshapes it before the addition.
        self.skip = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x):
        # Main path: conv -> BN -> ReLU -> dropout -> conv -> BN
        out = self.drop(self.relu(self.bn1(self.conv1(x))))
        out = self.bn2(self.conv2(out))
        # Add the shortcut, then the final ReLU.
        return self.relu(out + self.skip(x))


class ECGResNet(nn.Module):
    """1D ResNet for multi-label classification of 12-lead ECGs.

    Structure: stem -> residual blocks -> global average pool -> linear head.

    Args:
        n_leads: number of input leads (channels).
        n_labels: number of outputs, one per target label.
        widths: channels of each residual block. Every block after the first
            halves the time length, so more widths means a deeper model.
        dropout: dropout rate inside each residual block.
    """

    def __init__(self, n_leads=12, n_labels=len(LABELS), widths=(32, 64, 128, 256), dropout=0.2):
        super().__init__()
        # Stem: one wide convolution (15 samples) to pick up waveform shape,
        # then two halvings of the time length (conv stride + max pool) to cut
        # the cost of the layers that follow.
        self.stem = nn.Sequential(
            nn.Conv1d(n_leads, widths[0], 15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(widths[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1),
        )

        # One residual block per entry in widths. The first keeps the length;
        # each later one halves it while the channel count grows.
        blocks, in_ch = [], widths[0]
        for i, w in enumerate(widths):
            blocks.append(ResBlock1d(in_ch, w, stride=1 if i == 0 else 2, dropout=dropout))
            in_ch = w
        self.blocks = nn.Sequential(*blocks)

        # Average over time, so the output size does not depend on recording length.
        self.pool = nn.AdaptiveAvgPool1d(1)
        # One output per label. No sigmoid here: training uses
        # BCEWithLogitsLoss, which applies it internally and is more stable.
        self.head = nn.Linear(in_ch, n_labels)

    def forward(self, x):
        # Shapes for a 10 s ECG at 250 Hz (2500 samples) and default widths:
        # (B, 12, 2500) -> stem -> (B, 32, 625) -> blocks -> (B, 256, 79)
        x = self.blocks(self.stem(x))
        # (B, 256, 79) -> pool -> (B, 256, 1) -> squeeze -> (B, 256) -> head -> (B, 4)
        return self.head(self.pool(x).squeeze(-1))


if __name__ == "__main__":
    # Smoke test on random data: checks the shapes, not the learning.
    model = ECGResNet()
    x = torch.randn(8, 12, 2500)  # batch of 8 fake ECGs
    print("output shape:", tuple(model(x).shape))  # expect (8, 4)
    print("parameters:", sum(p.numel() for p in model.parameters()))
