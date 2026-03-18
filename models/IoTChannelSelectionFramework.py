from typing import Dict, Optional

import numpy as np
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
        backbone_name: str,
        backbone_kwargs: Optional[Dict] = None,
        input_time_length: Optional[int] = None,
        graph_decoder_dim: int = 64,
        num_classes: int = 4,
        adaptive_subset_eval: bool = False,
    ):
        super().__init__()
        self.selector = selector
        self.backbone = backbone
        self.backbone_name = backbone_name
        self.backbone_kwargs = dict(backbone_kwargs or {})
        self.input_time_length = input_time_length
        self.num_classes = num_classes
        self.adaptive_subset_eval = adaptive_subset_eval
        self.disable_graph_fusion_in_subset = False
        self.graph_decoder = nn.Sequential(
            nn.Linear(graph_decoder_dim, graph_decoder_dim),
            nn.GELU(),
            nn.Linear(graph_decoder_dim, num_classes),
        )
        self.fusion_alpha = nn.Parameter(torch.tensor(0.5))
        self._subset_backbone_cache = {}
        self.fixed_subset_indices: Optional[torch.Tensor] = None

    def set_adaptive_subset_eval(self, enabled: bool) -> None:
        self.adaptive_subset_eval = bool(enabled)

    def set_subset_graph_fusion(self, enabled: bool) -> None:
        self.disable_graph_fusion_in_subset = not bool(enabled)

    def set_fixed_subset_indices(self, indices: Optional[torch.Tensor]) -> None:
        if indices is None:
            self.fixed_subset_indices = None
            return
        self.fixed_subset_indices = indices.detach().cpu().to(dtype=torch.long)

    def _select_subset_indices(self, selector_out: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.fixed_subset_indices is not None:
            return self.fixed_subset_indices.to(selector_out["scores"].device)
        scores = selector_out["scores"].mean(dim=0)
        topk = min(self.selector.topk, scores.numel())
        if topk <= 0:
            return torch.arange(scores.numel(), device=scores.device, dtype=torch.long)
        return scores.topk(topk, dim=0).indices.to(dtype=torch.long)

    def _build_subset_backbone(self, indices: torch.Tensor, device: torch.device) -> nn.Module:
        key = tuple(int(idx) for idx in indices.detach().cpu().tolist())
        cached = self._subset_backbone_cache.get(key)
        if cached is not None:
            return cached.to(device)

        subset_kwargs = dict(self.backbone_kwargs)
        subset_kwargs["channel_indices"] = list(key)
        index_tensor = indices.detach().cpu()
        for tensor_key in ["Adj", "eu_adj"]:
            if tensor_key not in subset_kwargs:
                continue
            tensor_value = subset_kwargs[tensor_key]
            if isinstance(tensor_value, torch.Tensor):
                tensor_value = tensor_value.detach().cpu()
                subset_kwargs[tensor_key] = tensor_value.index_select(0, index_tensor).index_select(1, index_tensor)
            elif isinstance(tensor_value, np.ndarray):
                index_array = index_tensor.numpy()
                subset_kwargs[tensor_key] = tensor_value[np.ix_(index_array, index_array)]
        if "centrality" in subset_kwargs and isinstance(subset_kwargs["centrality"], torch.Tensor):
            subset_kwargs["centrality"] = subset_kwargs["centrality"].detach().cpu().index_select(0, index_tensor)
        subset_backbone = build_backbone(
            self.backbone_name,
            num_classes=self.num_classes,
            in_chans=len(key),
            input_time_length=self.input_time_length,
            backbone_kwargs=subset_kwargs,
        )
        transfer_backbone_weights(self.backbone_name, self.backbone, subset_backbone, key)
        subset_backbone = subset_backbone.to(device)
        self._subset_backbone_cache[key] = subset_backbone
        return subset_backbone

    def _forward_with_subset_backbone(self, x: torch.Tensor, selector_out: Dict[str, torch.Tensor]):
        indices = self._select_subset_indices(selector_out)
        subset_x = x.index_select(dim=1, index=indices)
        subset_backbone = self._build_subset_backbone(indices, x.device)
        subset_backbone.train(self.training)
        backbone_logits, backbone_aux = subset_backbone(subset_x)
        return backbone_logits, backbone_aux

    def forward(self, x: torch.Tensor, use_hard_mask: bool = True):
        selector_out = self.selector(x, use_hard_mask=use_hard_mask)
        if self.adaptive_subset_eval and use_hard_mask:
            backbone_logits, backbone_aux = self._forward_with_subset_backbone(x, selector_out)
        else:
            backbone_logits, backbone_aux = self.backbone(selector_out["gated_x"])
        graph_repr = selector_out["node_features"].mean(dim=1)
        graph_logits = self.graph_decoder(graph_repr)
        if self.adaptive_subset_eval and use_hard_mask and self.disable_graph_fusion_in_subset:
            alpha = torch.ones((), device=backbone_logits.device, dtype=backbone_logits.dtype)
            logits = backbone_logits
        else:
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


def _copy_matching_state(source: nn.Module, target: nn.Module):
    source_state = source.state_dict()
    target_state = target.state_dict()
    matched = {
        key: value
        for key, value in source_state.items()
        if key in target_state and target_state[key].shape == value.shape
    }
    target_state.update(matched)
    target.load_state_dict(target_state, strict=False)


def _transfer_nexusnet_weights(source: nn.Module, target: nn.Module, indices) -> None:
    _copy_matching_state(source, target)
    index_tensor = torch.tensor(indices, dtype=torch.long)

    with torch.no_grad():
        if hasattr(source, "spatial_conv") and hasattr(target, "spatial_conv"):
            source_weight = source.spatial_conv[0].weight.data
            weight_index = index_tensor.to(source_weight.device)
            target.spatial_conv[0].weight.data.copy_(
                source_weight.index_select(2, weight_index).to(target.spatial_conv[0].weight.data.device)
            )

        if hasattr(source, "centrality") and hasattr(target, "centrality"):
            centrality_index = index_tensor.to(source.centrality.device)
            target.centrality = source.centrality.index_select(0, centrality_index).clone().to(target.centrality.device)

        if hasattr(source, "Adj") and hasattr(target, "Adj"):
            adj_index = index_tensor.to(source.Adj.device)
            target.Adj = source.Adj.index_select(0, adj_index).index_select(1, adj_index).clone().to(target.Adj.device)

        if hasattr(source, "eu_adj") and hasattr(target, "eu_adj"):
            if isinstance(source.eu_adj, torch.Tensor):
                eu_index = index_tensor.to(source.eu_adj.device)
                target.eu_adj = source.eu_adj.index_select(0, eu_index).index_select(1, eu_index).clone()
            else:
                index_array = index_tensor.cpu().numpy()
                target.eu_adj = np.array(source.eu_adj)[np.ix_(index_array, index_array)].copy()

        if hasattr(source, "nexus") and hasattr(target, "nexus"):
            source_nexus = source.nexus[1]
            target_nexus = target.nexus[1]
            if hasattr(source_nexus, "edge_weight") and hasattr(target_nexus, "edge_weight"):
                full_neighbor = torch.zeros(
                    source_nexus.n_nodes,
                    source_nexus.n_nodes,
                    device=source_nexus.edge_weight.device,
                    dtype=source_nexus.edge_weight.dtype,
                )
                full_neighbor[source_nexus.xs, source_nexus.ys] = source_nexus.edge_weight.data
                full_neighbor = full_neighbor + full_neighbor.transpose(0, 1)
                neighbor_index = index_tensor.to(full_neighbor.device)
                subset_neighbor = full_neighbor.index_select(0, neighbor_index).index_select(1, neighbor_index)
                xs, ys = torch.tril_indices(len(indices), len(indices), offset=-1)
                target_nexus.edge_weight.data.copy_(subset_neighbor[xs, ys])


def _transfer_lggnet_weights(source: nn.Module, target: nn.Module, indices) -> None:
    _copy_matching_state(source, target)
    index_tensor = torch.tensor(indices, dtype=torch.long)
    with torch.no_grad():
        if hasattr(source, "local_filter_weight") and hasattr(target, "local_filter_weight"):
            weight_index = index_tensor.to(source.local_filter_weight.device)
            target.local_filter_weight.data.copy_(
                source.local_filter_weight.data.index_select(0, weight_index).to(target.local_filter_weight.device)
            )
        if hasattr(source, "local_filter_bias") and hasattr(target, "local_filter_bias"):
            bias_index = index_tensor.to(source.local_filter_bias.device)
            target.local_filter_bias.data.copy_(
                source.local_filter_bias.data.index_select(1, bias_index).to(target.local_filter_bias.device)
            )


def _transfer_mshallow_weights(source: nn.Module, target: nn.Module, indices) -> None:
    _copy_matching_state(source, target)
    index_tensor = torch.tensor(indices, dtype=torch.long)
    with torch.no_grad():
        if hasattr(source, "spatial") and hasattr(target, "spatial"):
            target.spatial.weight.data.copy_(source.spatial.weight.data.index_select(2, index_tensor))


def transfer_backbone_weights(backbone_name: str, source: nn.Module, target: nn.Module, indices) -> None:
    if backbone_name == "nexusnet":
        _transfer_nexusnet_weights(source, target, indices)
        return
    if backbone_name == "lggnet":
        _transfer_lggnet_weights(source, target, indices)
        return
    if backbone_name == "mshallowconvnet":
        _transfer_mshallow_weights(source, target, indices)
        return
    _copy_matching_state(source, target)
