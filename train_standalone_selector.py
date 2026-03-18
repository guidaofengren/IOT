import argparse
import json
import os
import random

import torch
import torch.nn.functional as F

from models.IoTChannelSelectionFramework import GraphGuidedChannelSelector
from tools.datasets import EEG_CHANNELS, infer_dataset_name, load_single_subject
from tools.utils import load_adj, set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", default="bciciv2a", choices=["bciciv2a", "bciciv2b"])
    parser.add_argument("-subject_id", type=int, default=1)
    parser.add_argument("-duration", type=float, default=4.0)
    parser.add_argument("-topk", type=int, default=12)
    parser.add_argument("-epochs", type=int, default=120)
    parser.add_argument("-batch_size", type=int, default=32)
    parser.add_argument("-lr", type=float, default=1e-3)
    parser.add_argument("-w_decay", type=float, default=0.0)
    parser.add_argument("-selector_hidden", type=int, default=64)
    parser.add_argument("-selector_layers", type=int, default=2)
    parser.add_argument("-selector_dropout", type=float, default=0.1)
    parser.add_argument("-smooth_lambda", type=float, default=5e-4)
    parser.add_argument("-budget_lambda", type=float, default=1e-3)
    parser.add_argument("-pairwise_lambda", type=float, default=1.0)
    parser.add_argument("-bce_lambda", type=float, default=1.0)
    parser.add_argument("-margin", type=float, default=0.1)
    parser.add_argument("-start_temp", type=float, default=2.5)
    parser.add_argument("-end_temp", type=float, default=0.5)
    parser.add_argument("-device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("-seed", type=int, default=42)
    parser.add_argument("-output", default="standalone_selector_s1.json")
    return parser.parse_args()


def graph_smoothness(scores: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
    degree = adj.sum(dim=-1)
    lap = torch.diag_embed(degree) - adj
    scores_vec = scores.unsqueeze(-1)
    return torch.matmul(scores_vec.transpose(1, 2), torch.matmul(lap, scores_vec)).mean()


def budget_penalty(scores: torch.Tensor, target_budget: float) -> torch.Tensor:
    usage = scores.mean(dim=1)
    return (usage - target_budget).abs().mean()


def pairwise_margin_loss(scores: torch.Tensor, positive_indices, negative_indices, margin: float, device: str) -> torch.Tensor:
    if not positive_indices or not negative_indices:
        return torch.tensor(0.0, device=device)
    pos_index = torch.tensor(positive_indices, dtype=torch.long, device=device)
    neg_index = torch.tensor(negative_indices, dtype=torch.long, device=device)
    pos_scores = scores.index_select(1, pos_index)
    neg_scores = scores.index_select(1, neg_index)
    diffs = pos_scores.unsqueeze(-1) - neg_scores.unsqueeze(1)
    return torch.relu(margin - diffs).mean()


def make_batches(data: torch.Tensor, batch_size: int):
    indices = list(range(len(data)))
    random.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch_ids = indices[start : start + batch_size]
        index_tensor = torch.tensor(batch_ids, dtype=torch.long)
        yield data.index_select(0, index_tensor)


def _normalize_vector(values: torch.Tensor) -> torch.Tensor:
    centered = values - values.mean()
    scaled = centered / centered.std(unbiased=False).clamp_min(1e-6)
    return torch.sigmoid(scaled)


def _normalize_batch_logits(values: torch.Tensor) -> torch.Tensor:
    centered = values - values.mean(dim=1, keepdim=True)
    scaled = centered / values.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
    return torch.sigmoid(scaled)


def build_region_masks(dataset_name: str):
    channel_names = EEG_CHANNELS[dataset_name]
    masks = {
        "motor_core": torch.zeros(len(channel_names), dtype=torch.float32),
        "motor_support": torch.zeros(len(channel_names), dtype=torch.float32),
        "frontal_motor": torch.zeros(len(channel_names), dtype=torch.float32),
        "central_motor": torch.zeros(len(channel_names), dtype=torch.float32),
        "centroparietal_motor": torch.zeros(len(channel_names), dtype=torch.float32),
        "posterior": torch.zeros(len(channel_names), dtype=torch.float32),
    }
    core_prefixes = ("FC", "C", "CP")
    support_prefixes = ("F",)
    posterior_prefixes = ("P", "PO", "O")
    for idx, name in enumerate(channel_names):
        if name.startswith(core_prefixes):
            masks["motor_core"][idx] = 1.0
            if name.startswith("FC"):
                masks["frontal_motor"][idx] = 1.0
            elif name.startswith("CP"):
                masks["centroparietal_motor"][idx] = 1.0
            else:
                masks["central_motor"][idx] = 1.0
        elif name.startswith(support_prefixes):
            masks["motor_support"][idx] = 1.0
        elif name.startswith(posterior_prefixes):
            masks["posterior"][idx] = 1.0
    return masks


def build_region_strengths(dataset_name: str):
    masks = build_region_masks(dataset_name)
    return {
        "frontal_motor": masks["frontal_motor"],
        "central_motor": masks["central_motor"],
        "centroparietal_motor": masks["centroparietal_motor"],
        "motor_core": masks["motor_core"],
        "posterior": masks["posterior"],
    }


def build_motor_anatomy_prior(dataset_name: str) -> torch.Tensor:
    channel_names = EEG_CHANNELS[dataset_name]
    prior = torch.full((len(channel_names),), 0.15, dtype=torch.float32)
    strong = {
        "FC3": 0.88,
        "FC1": 0.92,
        "FCz": 0.95,
        "FC2": 0.92,
        "FC4": 0.88,
        "C3": 0.95,
        "C1": 0.98,
        "Cz": 1.00,
        "C2": 0.98,
        "C4": 0.95,
        "CP3": 0.80,
        "CP1": 0.88,
        "CPz": 0.90,
        "CP2": 0.88,
        "CP4": 0.80,
    }
    medium = {
        "Fz": 0.55,
        "C5": 0.45,
        "C6": 0.45,
    }
    weak = {
        "P1": 0.12,
        "Pz": 0.10,
        "P2": 0.12,
        "POz": 0.05,
    }
    for idx, name in enumerate(channel_names):
        if name in strong:
            prior[idx] = strong[name]
        elif name in medium:
            prior[idx] = medium[name]
        elif name in weak:
            prior[idx] = weak[name]
    return prior


def build_backbone_compatibility_prior(dataset_name: str) -> torch.Tensor:
    channel_names = EEG_CHANNELS[dataset_name]
    prior = torch.full((len(channel_names),), 0.08, dtype=torch.float32)
    strong = {
        "FC3": 0.96,
        "FC1": 0.98,
        "FCz": 1.00,
        "FC2": 0.98,
        "FC4": 0.96,
        "C3": 0.98,
        "C1": 1.00,
        "Cz": 1.00,
        "C2": 1.00,
        "C4": 0.98,
    }
    medium = {
        "CP3": 0.72,
        "CP1": 0.78,
        "CPz": 0.80,
        "CP2": 0.78,
        "CP4": 0.72,
        "Fz": 0.45,
        "C5": 0.36,
        "C6": 0.36,
    }
    weak = {
        "P1": 0.10,
        "Pz": 0.08,
        "P2": 0.10,
        "POz": 0.02,
    }
    for idx, name in enumerate(channel_names):
        if name in strong:
            prior[idx] = strong[name]
        elif name in medium:
            prior[idx] = medium[name]
        elif name in weak:
            prior[idx] = weak[name]
    return prior


def build_structural_rank_bias(dataset_name: str) -> torch.Tensor:
    channel_names = EEG_CHANNELS[dataset_name]
    bias = torch.zeros(len(channel_names), dtype=torch.float32)
    preferred = {
        "FC1": 0.22,
        "FCz": 0.24,
        "FC2": 0.22,
        "C1": 0.20,
        "Cz": 0.22,
        "C2": 0.20,
        "C3": 0.16,
        "C4": 0.16,
        "FC3": 0.14,
        "FC4": 0.14,
    }
    suppressed = {
        "CP3": -0.05,
        "CP1": -0.06,
        "CPz": -0.07,
        "CP2": -0.06,
        "CP4": -0.05,
        "P1": -0.16,
        "Pz": -0.18,
        "P2": -0.16,
        "POz": -0.24,
    }
    for idx, name in enumerate(channel_names):
        if name in preferred:
            bias[idx] = preferred[name]
        elif name in suppressed:
            bias[idx] = suppressed[name]
    return bias


def _fisher_score(features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    classes = torch.unique(labels)
    global_mean = features.mean(dim=0)
    numerator = torch.zeros(features.shape[1], dtype=features.dtype)
    denominator = torch.zeros(features.shape[1], dtype=features.dtype)
    for cls in classes:
        mask = labels == cls
        cls_features = features[mask]
        if cls_features.numel() == 0:
            continue
        cls_mean = cls_features.mean(dim=0)
        cls_var = cls_features.var(dim=0, unbiased=False)
        numerator += cls_features.shape[0] * (cls_mean - global_mean).pow(2)
        denominator += cls_features.shape[0] * cls_var
    return numerator / denominator.clamp_min(1e-6)


def build_task_driven_target(train_X: torch.Tensor, train_y: torch.Tensor, static_adj: torch.Tensor, topk: int):
    data = train_X.detach().float()
    energy = data.pow(2).mean(dim=(0, 2))
    temporal_var = data.var(dim=2, unbiased=False).mean(dim=0)
    diff_energy = (data[:, :, 1:] - data[:, :, :-1]).pow(2).mean(dim=(0, 2))
    mean_abs = data.abs().mean(dim=2)
    log_power = torch.log1p(data.pow(2).mean(dim=2))
    labels = train_y.detach().long().cpu()
    fisher_abs = _fisher_score(mean_abs.cpu(), labels)
    fisher_power = _fisher_score(log_power.cpu(), labels)
    fisher_joint = 0.5 * _normalize_vector(fisher_abs) + 0.5 * _normalize_vector(fisher_power)

    flattened = data.permute(1, 0, 2).reshape(data.shape[1], -1)
    flattened = flattened - flattened.mean(dim=1, keepdim=True)
    covariance = torch.matmul(flattened, flattened.transpose(0, 1)) / max(1, flattened.shape[1] - 1)
    std = covariance.diag().clamp_min(1e-6).sqrt()
    corr = covariance / (std.unsqueeze(1) * std.unsqueeze(0))
    corr = corr.abs()
    corr.fill_diagonal_(0.0)
    corr_centrality = corr.mean(dim=1)

    graph_degree = static_adj.float().sum(dim=1)
    dataset_name = "BNCI2014001" if data.shape[1] == 22 else "BNCI2014004"
    anatomy_prior = build_motor_anatomy_prior(dataset_name)
    region_masks = build_region_masks(dataset_name)
    motor_core = region_masks["motor_core"]
    motor_support = region_masks["motor_support"]
    frontal_motor = region_masks["frontal_motor"]
    central_motor = region_masks["central_motor"]
    centroparietal_motor = region_masks["centroparietal_motor"]
    posterior = region_masks["posterior"]

    # Reward motor-strip channels by default, but only keep posterior channels if
    # they are strongly discriminative for the current subject.
    posterior_gate = torch.sigmoid((fisher_joint - 0.64) * 10.0)
    posterior_penalty = posterior * (1.0 - posterior_gate)
    motor_boost = (
        0.16 * frontal_motor
        + 0.12 * central_motor
        + 0.06 * centroparietal_motor
        + 0.03 * motor_support
    )
    spatial_bias = (anatomy_prior + motor_boost - 0.22 * posterior_penalty).clamp(0.0, 1.0)

    backbone_prior = build_backbone_compatibility_prior(dataset_name)
    frontal_central_focus = _normalize_vector(1.2 * frontal_motor + 1.0 * central_motor + 0.35 * centroparietal_motor)
    prior = (
        0.30 * _normalize_vector(spatial_bias)
        + 0.24 * fisher_joint
        + 0.10 * _normalize_vector(backbone_prior)
        + 0.07 * frontal_central_focus
        + 0.09 * _normalize_vector(temporal_var)
        + 0.07 * _normalize_vector(diff_energy)
        + 0.05 * _normalize_vector(corr_centrality)
        + 0.04 * _normalize_vector(graph_degree)
        + 0.04 * _normalize_vector(energy)
        - 0.04 * posterior_penalty
    )
    target = prior.clamp(0.0, 1.0)
    topk = min(max(1, topk), target.numel())
    positive_indices = target.topk(topk, dim=0).indices.tolist()
    negative_indices = [idx for idx in range(target.numel()) if idx not in positive_indices]
    return target, positive_indices, negative_indices


def compose_backbone_friendly_scores(
    scores: torch.Tensor,
    view_scores: dict,
    dataset_name: str,
    temperature: float,
) -> torch.Tensor:
    stat = _normalize_batch_logits(view_scores["stat"])
    temp = _normalize_batch_logits(view_scores["temp"])
    graph = _normalize_batch_logits(view_scores["graph"])
    compatibility_prior = build_backbone_compatibility_prior(dataset_name).to(scores.device)
    compatibility_prior = compatibility_prior.unsqueeze(0).expand_as(scores)
    region_masks = build_region_masks(dataset_name)
    frontal_motor = region_masks["frontal_motor"].to(scores.device).unsqueeze(0).expand_as(scores)
    central_motor = region_masks["central_motor"].to(scores.device).unsqueeze(0).expand_as(scores)
    centroparietal_motor = region_masks["centroparietal_motor"].to(scores.device).unsqueeze(0).expand_as(scores)
    posterior = region_masks["posterior"].to(scores.device).unsqueeze(0).expand_as(scores)
    frontal_central_focus = (1.1 * frontal_motor + 1.0 * central_motor + 0.3 * centroparietal_motor).clamp(0.0, 1.0)
    structural_bias = build_structural_rank_bias(dataset_name).to(scores.device).unsqueeze(0).expand_as(scores)
    cp_excess = torch.relu(centroparietal_motor - 0.5 * (frontal_motor + central_motor))
    cp_penalty = cp_excess * torch.sigmoid((graph - 0.52) * 8.0)
    proxy = (
        0.36 * stat
        + 0.36 * temp
        + 0.08 * graph
        + 0.08 * compatibility_prior
        + 0.07 * frontal_central_focus
        + 0.05 * structural_bias
        - 0.04 * cp_penalty
        - 0.04 * posterior
    )
    proxy = proxy.clamp(0.0, 1.0)
    # Keep a residual link to the learned fusion head so ranking stays data-driven.
    return (0.52 * proxy + 0.48 * scores).clamp(0.0, 1.0)


def main():
    args = parse_args()
    set_seed(args.seed)
    random.seed(args.seed)

    dataset_name = infer_dataset_name(args.dataset)
    train_X, train_y, _, _, _ = load_single_subject(
        dataset=dataset_name,
        subject_id=args.subject_id,
        duration=args.duration,
        to_tensor=True,
    )
    static_adj, _ = load_adj(dataset_name)
    static_adj = torch.tensor(static_adj, dtype=torch.float32)
    selector = GraphGuidedChannelSelector(
        in_chans=train_X.shape[1],
        static_adj=static_adj,
        hidden_dim=args.selector_hidden,
        num_layers=args.selector_layers,
        dropout=args.selector_dropout,
        topk=args.topk,
        use_dynamic_graph=True,
    ).to(args.device)

    optimizer = torch.optim.Adam(selector.parameters(), lr=args.lr, weight_decay=args.w_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    target_source = "task_driven_gnn"
    target, positive_indices, negative_indices = build_task_driven_target(train_X, train_y, static_adj, args.topk)
    target = target.to(args.device)
    target_budget = float(args.topk) / float(train_X.shape[1])

    for epoch_idx in range(args.epochs):
        selector.train()
        progress = epoch_idx / max(1, args.epochs - 1)
        selector.set_temperature(args.start_temp + (args.end_temp - args.start_temp) * progress)
        for batch in make_batches(train_X, args.batch_size):
            batch = batch.to(args.device)
            optimizer.zero_grad()
            out = selector(batch, use_hard_mask=False)
            scores = out["scores"]
            aligned_scores = compose_backbone_friendly_scores(
                scores,
                out["view_scores"],
                dataset_name,
                selector.temperature,
            )
            target_batch = target.unsqueeze(0).expand(scores.size(0), -1)
            loss = args.bce_lambda * F.binary_cross_entropy(aligned_scores, target_batch)
            loss = loss + args.pairwise_lambda * pairwise_margin_loss(
                aligned_scores,
                positive_indices,
                negative_indices,
                args.margin,
                args.device,
            )
            loss = loss + args.smooth_lambda * graph_smoothness(scores, out["adj"])
            loss = loss + args.budget_lambda * budget_penalty(scores, target_budget)
            loss = loss + 0.10 * F.mse_loss(scores, aligned_scores.detach())
            loss.backward()
            optimizer.step()
        scheduler.step()

    selector.eval()
    with torch.no_grad():
        out = selector(train_X.to(args.device), use_hard_mask=False)
        aligned_scores = compose_backbone_friendly_scores(
            out["scores"],
            out["view_scores"],
            dataset_name,
            selector.temperature,
        )
        channel_scores = aligned_scores.mean(dim=0).detach().cpu().tolist()
        raw_scores = out["scores"].mean(dim=0).detach().cpu().tolist()
        channel_mask = out["mask"].mean(dim=0).detach().cpu().tolist()

    ranking = sorted(
        [
            {
                "index": idx,
                "channel": EEG_CHANNELS[dataset_name][idx],
                "score": channel_scores[idx],
                "raw_score": raw_scores[idx],
                "mask": channel_mask[idx],
            }
            for idx in range(len(channel_scores))
        ],
        key=lambda item: item["score"],
        reverse=True,
    )

    payload = {
        "dataset": dataset_name,
        "subject_id": args.subject_id,
        "selector_type": "standalone_gnn",
        "ranking_mode": "task_driven_gnn_nexus_proxy",
        "topk": args.topk,
        "target_indices": None,
        "target_source": target_source,
        "selected_indices": [int(item["index"]) for item in ranking[: args.topk]],
        "selected_channels": [item["channel"] for item in ranking[: args.topk]],
        "ranking": ranking,
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "w_decay": args.w_decay,
            "selector_hidden": args.selector_hidden,
            "selector_layers": args.selector_layers,
            "selector_dropout": args.selector_dropout,
            "smooth_lambda": args.smooth_lambda,
            "budget_lambda": args.budget_lambda,
            "pairwise_lambda": args.pairwise_lambda,
            "bce_lambda": args.bce_lambda,
            "margin": args.margin,
            "start_temp": args.start_temp,
            "end_temp": args.end_temp,
        },
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(payload)


if __name__ == "__main__":
    main()
