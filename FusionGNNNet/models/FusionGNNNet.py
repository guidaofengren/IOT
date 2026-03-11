import torch
from torch import nn

from tools.eeg_graph_features import build_dynamic_adj, extract_node_features
from models.utils import Conv2dWithConstraint

# ==============================================================================
# 1. EEGNet-style CNN Frontend
# ==============================================================================
class EEGNetFrontend(nn.Module):
    """
    EEGNet-style temporal and spatial feature extractor.
    Operates on [B, 1, C, T] and extracts per-channel features.
    Matches NexusNet temporal_conv + spatial_conv dimension behavior.
    """
    def __init__(self, in_chans: int, temporal_filters: int = 8, spatial_filters: int = 2, kernel_length: int = 64, drop_prob: float = 0.3):
        super().__init__()
        self.F1 = temporal_filters
        self.D = spatial_filters
        self.F2 = self.F1 * self.D

        # 1. Temporal Convolution (across time, independent per channel)
        self.temporal_conv = nn.Sequential(
            Conv2dWithConstraint(
                in_channels=1, 
                out_channels=self.F1,
                kernel_size=(1, kernel_length),
                max_norm=None,
                stride=1,
                bias=False,
                padding=(0, kernel_length // 2)
            ),
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),
        )

        # 2. Depthwise Spatial Convolution (mixes channel info locally)
        # Note: NexusNet maps C channels down to 1 virtual channel.
        # But for GNN, we want features PRESERVED PER CHANNEL (N nodes).
        # We modify spatial conv to be a 1x1 conv across channels or omit it if we want independent nodal features.
        # Since we are building a Graph of channels, the node features should represent that channel.
        # Therefore, we just use temporal conv + point-wise feature aggregation.
        
        self.pointwise_conv = nn.Sequential(
            nn.Conv2d(self.F1, self.F2, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)),
            nn.Dropout(drop_prob)
        )
        # 3. Separable Convolution (cross-time feature mixing)
        # Operating on [B, F2, C, T]
        # Depthwise across time: kernel=(1, 16), groups=F2
        self.separable_conv = nn.Sequential(
            nn.Dropout(drop_prob),
            nn.Conv2d(self.F2, self.F2, kernel_size=(1, 16), stride=1, bias=False, groups=self.F2, padding=(0, 7)),
            nn.Conv2d(self.F2, self.F2, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8))
        )

        # We need a temporal projection because the sequence length after pool is ~ 1125/8/8 = 17
        self.temp_pool = nn.AdaptiveAvgPool2d((in_chans, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        x = x.unsqueeze(1) # [B, 1, C, T]

        # Temporal conv: [B, F1, C, T]
        x = self.temporal_conv(x)
        
        # Pointwise & pool: [B, F2, C, T//8]
        x = self.pointwise_conv(x)
        
        # Separable & pool: [B, F2, C, T//64]
        x = self.separable_conv(x)
        
        # We want node features: [B, C, cnn_dim]
        # Currently x is [B, F2, C, T']
        # Instead of generic mean, we use the Adaptive Pool to get [B, F2, C, 1]
        x = self.temp_pool(x)
        x = x.squeeze(-1) # [B, F2, C]
        
        # Return as [B, C, F2]
        x = x.transpose(1, 2)
        return x

# ==============================================================================
# 2. GNN Components (GIN + JK-Net)
# ==============================================================================
class GINLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D], adj: [B, N, N]
        agg = torch.matmul(adj, x)
        out = (1.0 + self.eps) * x + agg

        B, N, D = out.shape
        out = out.reshape(B * N, D)
        out = self.mlp(out)
        out = out.reshape(B, N, D)

        return self.norm(out + x)


class GINEncoder(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([GINLayer(hidden_dim, dropout) for _ in range(num_layers)])
        # JK-Net attention for layer aggregation
        self.jk_attn = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        outputs = []
        for layer in self.layers:
            x = layer(x, adj)
            outputs.append(x)

        stack = torch.stack(outputs, dim=-2)  # [B, N, num_layers, D]
        weights = self.jk_attn(stack)         # [B, N, num_layers, 1]
        out = (weights * stack).sum(dim=-2)   # [B, N, D]
        return out

# ==============================================================================
# 3. Main Model
# ==============================================================================
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
        hidden_dim=64,
        num_layers=2,
        drop_prob=0.3,
        topk=8,
        cnn_dim=16, # matching F1*D from NexusNet (8*2)
        dataset=None,
        **kwargs,
    ):
        super().__init__()
        del flag, eu_adj, centrality, input_time_length, dataset, kwargs

        self.hidden_dim = hidden_dim
        self.topk = topk

        # ---- Static adjacency: symmetric normalization ----
        static_adj = torch.as_tensor(Adj, dtype=torch.float32)
        static_adj = static_adj + torch.eye(static_adj.size(0), dtype=static_adj.dtype)
        degree = static_adj.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        inv_sqrt = degree.rsqrt()
        static_adj = inv_sqrt * static_adj * inv_sqrt.transpose(0, 1)
        self.register_buffer("static_adj", static_adj)

        # ---- CNN frontend: extract temporal features per channel ----
        self.temporal_cnn = EEGNetFrontend(in_chans=in_chans, temporal_filters=8, spatial_filters=2, drop_prob=drop_prob)

        # ---- Node projection: CNN features (F1*D) + hand-crafted (15) → hidden_dim ----
        cnn_out_dim = self.temporal_cnn.F2
        feat_input_dim = cnn_out_dim + 15
        self.node_proj = nn.Sequential(
            nn.Linear(feat_input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(drop_prob),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # ---- Dual-branch GIN encoders ----
        self.static_encoder = GINEncoder(hidden_dim, num_layers, drop_prob)
        self.dynamic_encoder = GINEncoder(hidden_dim, num_layers, drop_prob)

        # ---- Gated fusion ----
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

        # ---- Classifier (multi-pooling readout) ----
        # Concat Mean and Max => 2 * hidden_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(drop_prob),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_prob),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, inputs):
        # inputs: [B, C, T]
        x = inputs

        # 1. Extract features (Deep Learned + Hand-crafted)
        cnn_features = self.temporal_cnn(x)              # [B, C, 16]
        handcrafted_features = extract_node_features(x)  # [B, C, 15]
        
        node_features = torch.cat([cnn_features, handcrafted_features], dim=-1)  # [B, C, 31]
        node_features = self.node_proj(node_features)    # [B, C, hidden_dim]

        # 2. Build Graphs
        static_adj = self.static_adj.unsqueeze(0).expand(x.size(0), -1, -1)
        dynamic_adj = build_dynamic_adj(x, topk=self.topk)

        # 3. Dual-branch GIN Propagation
        h_static = self.static_encoder(node_features, static_adj)
        h_dynamic = self.dynamic_encoder(node_features, dynamic_adj)

        # 4. Node-level Gated Fusion
        gates = torch.softmax(self.gate(torch.cat([h_static, h_dynamic], dim=-1)), dim=-1)
        h_fused = gates[..., :1] * h_static + gates[..., 1:] * h_dynamic

        # 5. Multi-pooling Readout
        graph_repr_mean = h_fused.mean(dim=1)
        graph_repr_max, _ = h_fused.max(dim=1)
        graph_repr = torch.cat([graph_repr_mean, graph_repr_max], dim=-1)

        # 6. Classification
        logits = self.classifier(graph_repr)

        features = {
            "node_features": node_features.detach(),
            "gate_mean": gates.mean(dim=1).detach(),
        }
        return logits, features
