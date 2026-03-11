"""
eeg_graph_features.py — Enhanced 15-dim node feature extraction + dynamic adjacency.

Features (15 dims):
  0. log(variance)
  1. log(Hjorth activity)
  2. Hjorth mobility
  3. Hjorth complexity
  4. log(mu band power 8-13 Hz)
  5. log(beta band power 13-30 Hz)
  6. spectral entropy
  7. log(theta band power 4-8 Hz)
  8. log(alpha band power 8-13 Hz)   — alias of mu, kept for naming clarity
  9. log(gamma band power 30-45 Hz)
 10. skewness
 11. kurtosis
 12. zero-crossing rate
 13. log(RMS)
 14. log(mu/beta ratio)
"""
import torch


def _safe_zscore(x: torch.Tensor, dim: int) -> torch.Tensor:
    mean = x.mean(dim=dim, keepdim=True)
    std = x.std(dim=dim, keepdim=True, unbiased=False).clamp_min(1e-6)
    return (x - mean) / std


def extract_node_features(x: torch.Tensor, sfreq: float = 250.0) -> torch.Tensor:
    """Extract 15-dim node features from raw EEG signal.

    Args:
        x: [B, C, T] raw EEG signal
        sfreq: sampling frequency (default 250 Hz)

    Returns:
        features: [B, C, 15] z-scored node features
    """
    # --- Time-domain features ---
    centered = x - x.mean(dim=-1, keepdim=True)
    var = centered.pow(2).mean(dim=-1)

    diff_1 = torch.diff(x, dim=-1)
    diff_2 = torch.diff(diff_1, dim=-1)

    var_diff_1 = diff_1.pow(2).mean(dim=-1).clamp_min(1e-6)
    var_diff_2 = diff_2.pow(2).mean(dim=-1).clamp_min(1e-6)

    hjorth_activity = var.clamp_min(1e-6)
    hjorth_mobility = torch.sqrt(var_diff_1 / hjorth_activity)
    hjorth_complexity = torch.sqrt(var_diff_2 / var_diff_1) / hjorth_mobility.clamp_min(1e-6)

    # Skewness & Kurtosis
    std_vals = centered.pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-6).sqrt()
    normed = centered / std_vals
    skewness = normed.pow(3).mean(dim=-1)
    kurtosis = normed.pow(4).mean(dim=-1) - 3.0  # excess kurtosis

    # Zero-crossing rate
    sign_change = (x[..., 1:] * x[..., :-1]) < 0
    zcr = sign_change.float().mean(dim=-1)

    # RMS
    rms = x.pow(2).mean(dim=-1).clamp_min(1e-6).sqrt()

    # --- Frequency-domain features ---
    fft = torch.fft.rfft(centered, dim=-1)
    psd = fft.abs().pow(2)
    freqs = torch.fft.rfftfreq(x.size(-1), d=1.0 / sfreq).to(x.device)

    def bandpower(low: float, high: float) -> torch.Tensor:
        mask = (freqs >= low) & (freqs < high)
        if mask.sum() == 0:
            return torch.zeros_like(var)
        return psd[..., mask].mean(dim=-1)

    theta_power = bandpower(4.0, 8.0).clamp_min(1e-6)
    mu_power = bandpower(8.0, 13.0).clamp_min(1e-6)
    beta_power = bandpower(13.0, 30.0).clamp_min(1e-6)
    gamma_power = bandpower(30.0, 45.0).clamp_min(1e-6)

    # Spectral entropy
    psd_norm = psd / psd.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    spectral_entropy = -(psd_norm * (psd_norm.clamp_min(1e-8).log())).sum(dim=-1)

    # Band power ratio
    mu_beta_ratio = mu_power / beta_power

    features = torch.stack(
        [
            var.clamp_min(1e-6).log(),         # 0
            hjorth_activity.log(),             # 1
            hjorth_mobility,                   # 2
            hjorth_complexity,                 # 3
            mu_power.log(),                    # 4
            beta_power.log(),                  # 5
            spectral_entropy,                  # 6
            theta_power.log(),                 # 7
            mu_power.log(),                    # 8 (alpha = mu)
            gamma_power.log(),                 # 9
            skewness,                          # 10
            kurtosis,                          # 11
            zcr,                               # 12
            rms.log(),                         # 13
            mu_beta_ratio.clamp_min(1e-6).log(),  # 14
        ],
        dim=-1,
    )
    return _safe_zscore(features, dim=1)


def build_dynamic_adj(x: torch.Tensor, topk: int = 8) -> torch.Tensor:
    """Build dynamic adjacency from trial-level channel correlations.

    Args:
        x: [B, C, T] raw EEG
        topk: number of top neighbors to keep per channel

    Returns:
        normalized sparse adjacency: [B, C, C]
    """
    # Pearson correlation
    centered = x - x.mean(dim=-1, keepdim=True)
    cov = torch.matmul(centered, centered.transpose(1, 2))
    std = centered.pow(2).sum(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
    corr = cov / torch.matmul(std, std.transpose(1, 2))
    corr = corr.abs()

    batch_size, num_nodes, _ = corr.shape
    eye = torch.eye(num_nodes, device=x.device).unsqueeze(0)
    # Remove self-corr before top-k
    corr_no_self = corr * (1.0 - eye)

    topk = min(topk, num_nodes - 1)
    vals, idx = corr_no_self.topk(k=topk, dim=-1)
    sparse = torch.zeros_like(corr)
    sparse.scatter_(-1, idx, vals)
    # Symmetrize
    sparse = torch.maximum(sparse, sparse.transpose(1, 2))
    # Add self-loop
    sparse = sparse + eye

    # Symmetric normalization  D^{-1/2} A D^{-1/2}
    degree = sparse.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    inv_sqrt = degree.rsqrt()
    return inv_sqrt * sparse * inv_sqrt.transpose(1, 2)
