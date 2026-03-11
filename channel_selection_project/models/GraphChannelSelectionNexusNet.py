import torch
from torch import nn

from .NexusNet import NexusNet
from tools.eeg_graph_features import extract_node_features


class GraphSelectorLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))
        self.msg = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        agg = torch.matmul(adj, x)
        out = self.msg((1.0 + self.eps) * x + agg)
        return self.norm(out + x)


class GraphChannelSelector(nn.Module):
    def __init__(self, in_chans: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(7, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layers = nn.ModuleList(
            [
                GraphSelectorLayer(hidden_dim, dropout),
                GraphSelectorLayer(hidden_dim, dropout),
            ]
        )
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.in_chans = in_chans

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        node_feat = extract_node_features(x)
        h = self.node_encoder(node_feat)
        for layer in self.layers:
            h = layer(h, adj)
        logits = self.score(h).squeeze(-1)
        return torch.sigmoid(logits)


class GraphChannelSelectionNexusNet(nn.Module):
    def __init__(self, *args, hidden_dim: int = 64, selector_dropout: float = 0.1, **kwargs):
        super().__init__()
        in_chans = kwargs["in_chans"]
        adj = kwargs["Adj"]
        if not torch.is_tensor(adj):
            adj = torch.tensor(adj, dtype=torch.float32)
        adj = adj.float()
        adj = adj + torch.eye(adj.size(0), dtype=adj.dtype)
        degree = adj.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        adj = degree.rsqrt() * adj * degree.rsqrt().transpose(0, 1)
        self.register_buffer("selector_adj", adj)

        self.selector = GraphChannelSelector(
            in_chans=in_chans,
            hidden_dim=hidden_dim,
            dropout=selector_dropout,
        )
        self.backbone = NexusNet(*args, **kwargs)

    def channel_weights(self, x: torch.Tensor) -> torch.Tensor:
        adj = self.selector_adj.unsqueeze(0).expand(x.size(0), -1, -1).to(x.device)
        return self.selector(x, adj)

    def forward(self, x: torch.Tensor):
        weights = self.channel_weights(x)
        gated_x = x * weights.unsqueeze(-1)
        logits, features = self.backbone(gated_x)
        aux = {
            "channel_weights": weights,
            "backbone_features": features,
        }
        return logits, aux
