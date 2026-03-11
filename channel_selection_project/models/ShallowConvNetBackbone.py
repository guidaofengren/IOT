import torch
from torch import nn


class LogVarPool(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.clamp(x.var(dim=-1, keepdim=True, unbiased=False), min=1e-6))


class ShallowConvNetBackbone(nn.Module):
    """Compact ShallowConvNet-style backbone for low-channel MI-EEG decoding."""

    def __init__(
        self,
        in_chans: int,
        n_classes: int,
        input_time_length: int,
        temporal_filters: int = 16,
        spatial_filters: int = 32,
        kernel_length: int = 25,
        pool_stride: int = 15,
        dropout: float = 0.25,
        **kwargs,
    ):
        super().__init__()
        self.temporal = nn.Conv2d(
            1,
            temporal_filters,
            kernel_size=(1, kernel_length),
            padding=(0, kernel_length // 2),
            bias=False,
        )
        self.spatial = nn.Conv2d(
            temporal_filters,
            spatial_filters,
            kernel_size=(in_chans, 1),
            bias=False,
        )
        self.bn = nn.BatchNorm2d(spatial_filters)
        self.activation = nn.ELU()
        self.pool = nn.AvgPool2d(kernel_size=(1, 35), stride=(1, pool_stride), padding=(0, 17))
        self.logvar = LogVarPool()
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
        x = x.pow(2)
        x = self.pool(x)
        x = self.logvar(x)
        x = self.dropout(x)
        return x

    def forward(self, x: torch.Tensor):
        embedding = self._forward_features(x)
        logits = self.head(embedding.flatten(1))
        return logits, {"embedding": embedding}
