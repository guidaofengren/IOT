import torch


def _safe_zscore(x: torch.Tensor, dim: int) -> torch.Tensor:
    mean = x.mean(dim=dim, keepdim=True)
    std = x.std(dim=dim, keepdim=True, unbiased=False).clamp_min(1e-6)
    return (x - mean) / std


def extract_node_features(x: torch.Tensor, sfreq: float = 250.0) -> torch.Tensor:
    # x: [B, C, T]
    centered = x - x.mean(dim=-1, keepdim=True)
    var = centered.pow(2).mean(dim=-1)

    diff_1 = torch.diff(x, dim=-1)
    diff_2 = torch.diff(diff_1, dim=-1)

    var_diff_1 = diff_1.pow(2).mean(dim=-1).clamp_min(1e-6)
    var_diff_2 = diff_2.pow(2).mean(dim=-1).clamp_min(1e-6)

    hjorth_activity = var.clamp_min(1e-6)
    hjorth_mobility = torch.sqrt(var_diff_1 / hjorth_activity)
    hjorth_complexity = torch.sqrt(var_diff_2 / var_diff_1) / hjorth_mobility.clamp_min(1e-6)

    fft = torch.fft.rfft(centered, dim=-1)
    psd = fft.abs().pow(2)
    freqs = torch.fft.rfftfreq(x.size(-1), d=1.0 / sfreq).to(x.device)

    def bandpower(low: float, high: float) -> torch.Tensor:
        mask = (freqs >= low) & (freqs < high)
        if mask.sum() == 0:
            return torch.zeros_like(var)
        return psd[..., mask].mean(dim=-1)

    mu_power = bandpower(8.0, 13.0).clamp_min(1e-6)
    beta_power = bandpower(13.0, 30.0).clamp_min(1e-6)

    psd_norm = psd / psd.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    spectral_entropy = -(psd_norm * (psd_norm.clamp_min(1e-8).log())).sum(dim=-1)

    features = torch.stack(
        [
            var.clamp_min(1e-6).log(),
            hjorth_activity.log(),
            hjorth_mobility,
            hjorth_complexity,
            mu_power.log(),
            beta_power.log(),
            spectral_entropy,
        ],
        dim=-1,
    )
    return _safe_zscore(features, dim=1)


def build_dynamic_adj(x: torch.Tensor, topk: int = 8) -> torch.Tensor:
    # x: [B, C, T]
    centered = x - x.mean(dim=-1, keepdim=True)
    cov = torch.matmul(centered, centered.transpose(1, 2))
    std = centered.pow(2).sum(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
    corr = cov / torch.matmul(std, std.transpose(1, 2))
    corr = corr.abs()

    batch_size, num_nodes, _ = corr.shape
    eye = torch.eye(num_nodes, device=x.device).unsqueeze(0)
    corr = corr * (1.0 - eye) + eye

    topk = min(topk, num_nodes)
    vals, idx = corr.topk(k=topk, dim=-1)
    sparse = torch.zeros_like(corr)
    sparse.scatter_(-1, idx, vals)
    sparse = torch.maximum(sparse, sparse.transpose(1, 2))
    sparse = sparse + eye

    degree = sparse.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    inv_sqrt = degree.rsqrt()
    return inv_sqrt * sparse * inv_sqrt.transpose(1, 2)
