import torch
from torch import nn


class LiteEEGBackbone(nn.Module):
    """A compact EEG decoder for low-channel IoT settings."""

    def __init__(
        self,
        in_chans: int,
        n_classes: int,
        input_time_length: int,
        hidden_dim: int = 32,
        dropout: float = 0.25,
        **kwargs,
    ):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv1d(in_chans, hidden_dim, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.AvgPool1d(kernel_size=4, stride=4),
            nn.Dropout(dropout),
        )
        self.depthwise = nn.Sequential(
            nn.Conv1d(
                hidden_dim,
                hidden_dim,
                kernel_size=9,
                padding=4,
                groups=hidden_dim,
                bias=False,
            ),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.AvgPool1d(kernel_size=4, stride=4),
            nn.Dropout(dropout),
        )
        pooled_time = max(1, input_time_length // 16)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim * pooled_time, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor):
        x = self.temporal(x)
        x = self.depthwise(x)
        logits = self.head(x)
        return logits, {"embedding": x}
