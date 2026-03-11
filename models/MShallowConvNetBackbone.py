import torch
from torch import nn


class _LogPowerPool(nn.Module):
    def __init__(self, kernel_size: int, stride: int):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=(1, kernel_size), stride=(1, stride))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(self.pool(x.pow(2)).clamp_min(1e-6))


class MShallowConvNetBackbone(nn.Module):
    """Compact adaptation of M-ShallowConvNet-style defaults for MI EEG."""

    def __init__(
        self,
        in_chans: int,
        n_classes: int,
        input_time_length: int,
        sampling_rate: int = 250,
        depth: int = 24,
        temporal_kernel_seconds: float = 0.12,
        temporal_stride: int = 2,
        pool_length: int = 75,
        pool_stride: int = 15,
        dropout: float = 0.5,
        **kwargs,
    ):
        super().__init__()
        kernel_length = max(3, int(round(sampling_rate * temporal_kernel_seconds)))
        self.temporal = nn.Conv2d(
            1,
            depth,
            kernel_size=(1, kernel_length),
            stride=(1, temporal_stride),
            padding=(0, kernel_length // 2),
            bias=False,
        )
        self.spatial = nn.Conv2d(depth, depth, kernel_size=(in_chans, 1), bias=False, groups=1)
        self.bn = nn.BatchNorm2d(depth, momentum=0.01, eps=1e-3)
        self.activation = nn.ELU()
        self.pool = _LogPowerPool(kernel_size=pool_length, stride=pool_stride)
        self.dropout = nn.Dropout(dropout)

        with torch.no_grad():
            dummy = torch.zeros(1, in_chans, input_time_length)
            emb = self._forward_features(dummy)
        self.head = nn.Linear(emb.flatten(1).shape[1], n_classes)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x

    def forward(self, x: torch.Tensor):
        embedding = self._forward_features(x)
        logits = self.head(embedding.flatten(1))
        return logits, {"embedding": embedding}
