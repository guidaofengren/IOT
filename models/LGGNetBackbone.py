import torch
import torch.nn.functional as F
from torch import nn


class _PowerLayer(nn.Module):
    def __init__(self, length: int, step: int):
        super().__init__()
        self.pooling = nn.AvgPool2d(kernel_size=(1, length), stride=(1, step))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(self.pooling(x.pow(2)).clamp_min(1e-6))


class _GraphConvolution(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros((1, 1, out_features), dtype=torch.float32))
        nn.init.xavier_uniform_(self.weight, gain=1.414)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        support = torch.matmul(x, self.weight) - self.bias
        return F.relu(torch.matmul(adj, support))


class _Aggregator:
    def __init__(self, group_sizes):
        self.group_sizes = group_sizes
        idx = [0]
        for size in group_sizes:
            idx.append(idx[-1] + size)
        self.idx = idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chunks = []
        for i, size in enumerate(self.group_sizes):
            start = self.idx[i]
            end = self.idx[i + 1]
            chunks.append(x[:, start:end, :].mean(dim=1))
        return torch.stack(chunks, dim=1)


class LGGNetBackbone(nn.Module):
    """Adapted from the public LGG repository for BNCI2014001/4004-style inputs."""

    REGION_SIZES = {
        22: [1, 5, 7, 5, 3, 1],
        3: [3],
    }

    def __init__(
        self,
        in_chans: int,
        n_classes: int,
        input_time_length: int,
        sampling_rate: int = 250,
        num_t_filters: int = 16,
        out_graph: int = 16,
        dropout: float = 0.25,
        pool: int = 16,
        pool_step_rate: float = 0.25,
        **kwargs,
    ):
        super().__init__()
        region_sizes = self.REGION_SIZES.get(in_chans)
        if region_sizes is None:
            raise ValueError(f"Unsupported channel count for LGGNetBackbone: {in_chans}")
        self.region_sizes = region_sizes
        self.num_regions = len(region_sizes)

        self.window = [0.5, 0.25, 0.125]
        self.temporal_learners = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(1, num_t_filters, kernel_size=(1, int(window * sampling_rate)), stride=(1, 1)),
                    _PowerLayer(length=pool, step=max(1, int(pool_step_rate * pool))),
                )
                for window in self.window
            ]
        )
        self.bn_t = nn.BatchNorm2d(num_t_filters)
        self.bn_t2 = nn.BatchNorm2d(num_t_filters)
        self.one_by_one = nn.Sequential(
            nn.Conv2d(num_t_filters, num_t_filters, kernel_size=(1, 1), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.AvgPool2d((1, 2)),
        )

        feature_dim = self._infer_feature_dim(in_chans, input_time_length)
        self.local_filter_weight = nn.Parameter(torch.empty(in_chans, feature_dim))
        nn.init.xavier_uniform_(self.local_filter_weight)
        self.local_filter_bias = nn.Parameter(torch.zeros((1, in_chans, 1), dtype=torch.float32))

        self.aggregate = _Aggregator(region_sizes)
        self.global_adj = nn.Parameter(torch.empty(self.num_regions, self.num_regions))
        nn.init.xavier_uniform_(self.global_adj)
        self.bn_g1 = nn.BatchNorm1d(self.num_regions)
        self.bn_g2 = nn.BatchNorm1d(self.num_regions)
        self.gcn = _GraphConvolution(feature_dim, out_graph)
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.num_regions * out_graph, n_classes),
        )

    def _temporal_forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [learner(x) for learner in self.temporal_learners]
        out = torch.cat(outputs, dim=-1)
        out = self.bn_t(out)
        out = self.one_by_one(out)
        out = self.bn_t2(out)
        out = out.permute(0, 2, 1, 3).contiguous()
        return out.view(out.size(0), out.size(1), -1)

    def _infer_feature_dim(self, in_chans: int, input_time_length: int) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, 1, in_chans, input_time_length)
            out = self._temporal_forward(dummy)
        return out.size(-1)

    def _local_filter(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.local_filter_weight.unsqueeze(0).expand(x.size(0), -1, -1)
        return F.relu(x * weight - self.local_filter_bias)

    def _adjacency(self, x: torch.Tensor) -> torch.Tensor:
        similarity = torch.bmm(x, x.transpose(1, 2))
        adj = F.relu(similarity * (self.global_adj + self.global_adj.transpose(1, 0)))
        adj = adj + torch.eye(self.num_regions, device=x.device).unsqueeze(0)
        degree = adj.sum(dim=-1)
        inv_sqrt = degree.clamp_min(1e-6).pow(-0.5)
        inv_mat = torch.diag_embed(inv_sqrt)
        return torch.bmm(torch.bmm(inv_mat, adj), inv_mat)

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(1)
        x = self._temporal_forward(x)
        x = self._local_filter(x)
        x = self.aggregate.forward(x)
        adj = self._adjacency(x)
        x = self.bn_g1(x)
        x = self.gcn(x, adj)
        x = self.bn_g2(x)
        embedding = x.reshape(x.size(0), -1)
        logits = self.fc(embedding)
        return logits, {"embedding": embedding, "adj": adj}
