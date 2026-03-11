from typing import Dict, Optional

import torch
from torch import nn

from .LGGNetBackbone import LGGNetBackbone
from .MShallowConvNetBackbone import MShallowConvNetBackbone
from .NexusNet import NexusNet
from tools.eeg_graph_features import build_dynamic_adj, extract_node_features


class StraightThroughTopK(nn.Module):
    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        if self.k <= 0 or self.k >= scores.size(1):
            return torch.ones_like(scores)
        _, indices = scores.topk(self.k, dim=1)
        hard = torch.zeros_like(scores)
        hard.scatter_(1, indices, 1.0)
        # Use the dense scores as the gradient carrier so the budget loss
        # can meaningfully regularize the selector before the hard top-k is fixed.
        soft = scores
        return hard - soft.detach() + soft


class GraphMessageLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        propagated = torch.matmul(adj, x)
        updated = self.message(x + propagated)
        return self.norm(updated + x)


class GraphGuidedChannelSelector(nn.Module):
    def __init__(
        self,
        in_chans: int,
        static_adj: torch.Tensor,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        topk: int = 5,
        use_dynamic_graph: bool = True,
    ):
        super().__init__()
        adj = static_adj.float()
        adj = adj + torch.eye(adj.size(0), dtype=adj.dtype)
        degree = adj.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        norm_adj = degree.rsqrt() * adj * degree.rsqrt().transpose(0, 1)
        self.register_buffer("static_adj", norm_adj)

        self.use_dynamic_graph = use_dynamic_graph
        self.topk = topk
        self.stat_encoder = nn.Sequential(
            nn.Linear(7, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(1, hidden_dim // 2, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=7, padding=3, bias=False),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layers = nn.ModuleList(
            [GraphMessageLayer(hidden_dim, dropout) for _ in range(num_layers)]
        )
        self.stat_score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.temp_score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.graph_score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.score_fusion = nn.Sequential(
            nn.Linear(3, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.masker = StraightThroughTopK(topk)
        self.in_chans = in_chans
        self.target_budget = 1.0 if topk <= 0 else min(topk, in_chans) / float(in_chans)
        self.temperature = 1.0

    def set_temperature(self, temperature: float):
        self.temperature = max(0.1, float(temperature))

    def _compose_adj(self, x: torch.Tensor) -> torch.Tensor:
        static = self.static_adj.unsqueeze(0).expand(x.size(0), -1, -1).to(x.device)
        if not self.use_dynamic_graph:
            return static
        dynamic = build_dynamic_adj(x, topk=min(max(self.topk * 2, 4), self.in_chans))
        mixed = 0.5 * static + 0.5 * dynamic
        degree = mixed.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return degree.rsqrt() * mixed * degree.rsqrt().transpose(1, 2)

    def forward(self, x: torch.Tensor, use_hard_mask: bool = True) -> Dict[str, torch.Tensor]:
        node_features = extract_node_features(x)
        adj = self._compose_adj(x)
        stat_hidden = self.stat_encoder(node_features)
        batch_size, num_nodes, time_len = x.shape
        temporal_hidden = self.temporal_encoder(
            x.reshape(batch_size * num_nodes, 1, time_len)
        ).reshape(batch_size, num_nodes, -1)
        hidden = self.feature_fusion(torch.cat([stat_hidden, temporal_hidden], dim=-1))
        for layer in self.layers:
            hidden = layer(hidden, adj)
        stat_logits = self.stat_score(stat_hidden).squeeze(-1)
        temp_logits = self.temp_score(temporal_hidden).squeeze(-1)
        graph_logits = self.graph_score(hidden).squeeze(-1)
        logits = self.score_fusion(
            torch.stack([stat_logits, temp_logits, graph_logits], dim=-1)
        ).squeeze(-1)
        logits = logits - logits.mean(dim=1, keepdim=True)
        logits = logits / logits.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
        scores = torch.sigmoid(logits / self.temperature)
        mask = self.masker(scores) if use_hard_mask else scores
        gated_x = x * mask.unsqueeze(-1)
        return {
            "scores": scores,
            "logits": logits,
            "view_scores": {
                "stat": stat_logits,
                "temp": temp_logits,
                "graph": graph_logits,
            },
            "mask": mask,
            "gated_x": gated_x,
            "node_features": hidden,
            "adj": adj,
            "target_budget": torch.tensor(self.target_budget, device=x.device),
        }


class ModelAgnosticChannelSelectionWrapper(nn.Module):
    def __init__(
        self,
        selector: GraphGuidedChannelSelector,
        backbone: nn.Module,
        graph_decoder_dim: int = 64,
        num_classes: int = 4,
    ):
        super().__init__()
        self.selector = selector
        self.backbone = backbone
        self.graph_decoder = nn.Sequential(
            nn.Linear(graph_decoder_dim, graph_decoder_dim),
            nn.GELU(),
            nn.Linear(graph_decoder_dim, num_classes),
        )
        self.fusion_alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor, use_hard_mask: bool = True):
        selector_out = self.selector(x, use_hard_mask=use_hard_mask)
        backbone_logits, backbone_aux = self.backbone(selector_out["gated_x"])
        graph_repr = selector_out["node_features"].mean(dim=1)
        graph_logits = self.graph_decoder(graph_repr)
        alpha = torch.sigmoid(self.fusion_alpha)
        logits = alpha * backbone_logits + (1.0 - alpha) * graph_logits
        aux = {
            "channel_scores": selector_out["scores"],
            "channel_logits": selector_out["logits"],
            "view_scores": selector_out["view_scores"],
            "channel_mask": selector_out["mask"],
            "target_budget": selector_out["target_budget"],
            "selector_adj": selector_out["adj"],
            "selector_features": selector_out["node_features"],
            "backbone_aux": backbone_aux,
            "fusion_alpha": alpha.detach(),
        }
        return logits, aux


def build_backbone(
    backbone_name: str,
    *,
    num_classes: int,
    in_chans: int,
    input_time_length: int,
    backbone_kwargs: Optional[Dict] = None,
):
    backbone_kwargs = backbone_kwargs or {}
    if backbone_name == "nexusnet":
        return NexusNet(
            flag=[1, 1, 1, 1],
            n_classes=num_classes,
            in_chans=in_chans,
            input_time_length=input_time_length,
            **backbone_kwargs,
        )
    if backbone_name == "lggnet":
        return LGGNetBackbone(
            in_chans=in_chans,
            n_classes=num_classes,
            input_time_length=input_time_length,
            **backbone_kwargs,
        )
    if backbone_name == "mshallowconvnet":
        return MShallowConvNetBackbone(
            in_chans=in_chans,
            n_classes=num_classes,
            input_time_length=input_time_length,
            **backbone_kwargs,
        )
    raise ValueError(f"Unsupported backbone: {backbone_name}")
