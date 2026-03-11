import torch
from torch import nn

from .NexusNet import NexusNet


class ChannelSelectionNexusNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        in_chans = kwargs["in_chans"]
        self.selector_logits = nn.Parameter(torch.zeros(in_chans))
        self.backbone = NexusNet(*args, **kwargs)

    def channel_weights(self):
        return torch.sigmoid(self.selector_logits)

    def forward(self, x):
        weights = self.channel_weights().view(1, -1, 1)
        gated_x = x * weights
        logits, features = self.backbone(gated_x)
        aux = {
            "channel_weights": weights.squeeze(0).squeeze(-1),
            "backbone_features": features,
        }
        return logits, aux
