import math

import torch
from torch import nn
import torch.nn.functional as F

from tools.eeg_graph_features import build_dynamic_adj, extract_node_features


class TemporalFilterEncoder(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        mid_dim = hidden_dim // 2
        self.net = nn.Sequential(
            nn.Conv1d(1, mid_dim, kernel_size=31, padding=15, bias=False),
            nn.BatchNorm1d(mid_dim),
            nn.GELU(),
            nn.Conv1d(mid_dim, hidden_dim, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.score = nn.Conv1d(hidden_dim, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        batch_size, num_nodes, num_steps = x.shape
        x = x.reshape(batch_size * num_nodes, 1, num_steps)
        feat = self.net(x)
        score = torch.softmax(self.score(feat), dim=-1)
        pooled = (feat * score).sum(dim=-1)
        return pooled.view(batch_size, num_nodes, -1)


class BatchGraphAttention(nn.Module):
    def __init__(self, hidden_dim: int, heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, hidden_dim = x.shape
        q = self.q_proj(x).view(batch_size, num_nodes, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, num_nodes, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, num_nodes, self.heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = adj.unsqueeze(1)
        attn = attn.masked_fill(mask <= 1e-8, -1e4)
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch_size, num_nodes, hidden_dim)
        out = self.out_proj(out)
        return self.norm(out + x)


class GraphBlock(nn.Module):
    def __init__(self, hidden_dim: int, heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.attn = BatchGraphAttention(hidden_dim, heads=heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = self.attn(x, adj)
        return self.norm(self.ffn(x) + x)


class GraphStack(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.layers = nn.ModuleList([GraphBlock(hidden_dim, heads=heads, dropout=dropout) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        states = []
        for layer in self.layers:
            x = layer(x, adj)
            states.append(x)
        return torch.stack(states, dim=0).mean(dim=0)


class RegionPool(nn.Module):
    def __init__(self, in_chans: int):
        super().__init__()
        if in_chans == 22:
            groups = [[0], [1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17], [18, 19, 20], [21]]
        else:
            groups = [[i] for i in range(in_chans)]

        pool = torch.zeros(len(groups), in_chans, dtype=torch.float32)
        for ridx, members in enumerate(groups):
            pool[ridx, members] = 1.0 / len(members)
        self.register_buffer("pool", pool)
        self.register_buffer("region_adj", torch.ones(len(groups), len(groups), dtype=torch.float32))

    def down(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("rc,bch->brh", self.pool, x)

    def up(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("cr,brh->bch", self.pool.t(), x)


class FusionGNNNet(nn.Module):
    def __init__(
        self,
        flag,
        Adj,
        eu_adj,
        centrality,
        in_chans,
        n_classes,
        input_time_length=None,
        hidden_dim=128,
        num_layers=2,
        drop_prob=0.2,
        topk=8,
        dataset=None,
        **kwargs,
    ):
        super().__init__()
        del flag, eu_adj, centrality, dataset, input_time_length, kwargs

        static_adj = torch.as_tensor(Adj, dtype=torch.float32)
        static_adj = static_adj + torch.eye(static_adj.size(0), dtype=static_adj.dtype)
        self.register_buffer("static_adj", static_adj)

        self.topk = topk
        self.temporal_encoder = TemporalFilterEncoder(hidden_dim=hidden_dim, dropout=drop_prob)
        self.handcrafted_encoder = nn.Sequential(
            nn.Linear(7, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(drop_prob),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
        )
        self.node_fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim + hidden_dim // 2),
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_prob),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.static_graph = GraphStack(hidden_dim, num_layers=num_layers, heads=4, dropout=drop_prob)
        self.dynamic_graph = GraphStack(hidden_dim, num_layers=num_layers, heads=4, dropout=drop_prob)
        self.region_pool = RegionPool(in_chans)
        self.region_graph = GraphStack(hidden_dim, num_layers=1, heads=2, dropout=drop_prob)

        self.branch_gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_prob),
            nn.Linear(hidden_dim, 3),
        )
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(drop_prob),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_prob),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, inputs: torch.Tensor):
        temporal = self.temporal_encoder(inputs)
        handcrafted = self.handcrafted_encoder(extract_node_features(inputs))
        node_feat = self.node_fusion(torch.cat([temporal, handcrafted], dim=-1))

        static_adj = self.static_adj.unsqueeze(0).expand(inputs.size(0), -1, -1)
        dynamic_adj = build_dynamic_adj(inputs, topk=self.topk)

        h_static = self.static_graph(node_feat, static_adj)
        h_dynamic = self.dynamic_graph(node_feat, dynamic_adj)

        region_feat = self.region_pool.down(node_feat)
        region_adj = self.region_pool.region_adj.unsqueeze(0).expand(inputs.size(0), -1, -1)
        h_region = self.region_graph(region_feat, region_adj)
        h_region = self.region_pool.up(h_region)

        gate = torch.softmax(self.branch_gate(torch.cat([h_static, h_dynamic, h_region], dim=-1)), dim=-1)
        fused = gate[..., 0:1] * h_static + gate[..., 1:2] * h_dynamic + gate[..., 2:3] * h_region

        node_score = torch.softmax(self.readout(fused), dim=1)
        graph_repr = (node_score * fused).sum(dim=1)
        logits = self.classifier(graph_repr)

        aux = {
            "gate_mean": gate.mean(dim=1).detach(),
            "node_score_mean": node_score.mean(dim=0).detach(),
        }
        return logits, aux
