import torch
from torch import nn


class EEGNetBackbone(nn.Module):
    """Compact EEGNet-style backbone with a 3D-input interface adapted to [B, C, T]."""

    def __init__(
        self,
        in_chans: int,
        n_classes: int,
        input_time_length: int,
        f1: int = 8,
        d: int = 2,
        f2: int = 16,
        kernel_length: int = 64,
        dropout: float = 0.25,
        **kwargs,
    ):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv2d(
                1,
                f1,
                kernel_size=(1, kernel_length),
                padding=(0, kernel_length // 2),
                bias=False,
            ),
            nn.BatchNorm2d(f1),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(
                f1,
                f1 * d,
                kernel_size=(in_chans, 1),
                groups=f1,
                bias=False,
            ),
            nn.BatchNorm2d(f1 * d),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)),
            nn.Dropout(dropout),
        )
        self.separable = nn.Sequential(
            nn.Conv2d(
                f1 * d,
                f1 * d,
                kernel_size=(1, 16),
                padding=(0, 8),
                groups=f1 * d,
                bias=False,
            ),
            nn.Conv2d(f1 * d, f2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)),
            nn.Dropout(dropout),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, in_chans, input_time_length)
            emb = self._forward_features(dummy)
        flat_dim = emb.flatten(1).shape[1]
        self.head = nn.Linear(flat_dim, n_classes)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.separable(x)
        return x

    def forward(self, x: torch.Tensor):
        embedding = self._forward_features(x)
        logits = self.head(embedding.flatten(1))
        return logits, {"embedding": embedding}
