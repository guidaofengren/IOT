import argparse
import json
import os

import torch

from models.IoTChannelSelectionFramework import build_backbone
from official_profiles import get_official_defaults
from tools.complexity import benchmark_forward, count_parameters
from tools.datasets import EEG_CHANNELS, infer_dataset_name, load_single_subject
from tools.utils import load_adj, set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", default="bciciv2a", choices=["bciciv2a", "bciciv2b"])
    parser.add_argument("-subject_id", type=int, default=1)
    parser.add_argument("-duration", type=float, default=4.0)
    parser.add_argument("-backbone", default="nexusnet", choices=["nexusnet", "mshallowconvnet", "lggnet"])
    parser.add_argument("-device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("-seed", type=int, default=42)
    parser.add_argument("-channel_subset_json", required=True)
    parser.add_argument("-subset_topk", type=int, required=True)
    parser.add_argument("-dropout", type=float, default=0.25)
    parser.add_argument("-channel_weight_mode", default="none", choices=["none", "rank_linear", "rank_gate", "rank_gate_graph"])
    parser.add_argument("-channel_weight_floor", type=float, default=0.6)
    parser.add_argument("-checkpoint_paths", nargs="+", required=True)
    parser.add_argument("-checkpoint_metrics_jsons", nargs="+", default=None)
    parser.add_argument("-weighting", default="uniform", choices=["uniform", "val_acc"])
    parser.add_argument("-top_models", type=int, default=None)
    parser.add_argument("-output", default="ensemble_eval.json")
    return parser.parse_args()


def build_rank_weights(num_channels: int, floor: float):
    if num_channels <= 0:
        return None
    if num_channels == 1:
        return [1.0]
    floor = max(0.0, min(float(floor), 1.0))
    weights = torch.linspace(1.0, floor, steps=num_channels, dtype=torch.float32)
    weights = weights / weights.mean().clamp_min(1e-6)
    return weights.tolist()


def resolve_subset_spec(channel_subset_json, subset_topk, channel_weight_mode, channel_weight_floor):
    with open(channel_subset_json, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    ranking = payload.get("ranking")
    if ranking is None:
        raise ValueError("channel_subset_json must contain ranking.")
    selected = ranking[:subset_topk]
    weights = None
    if channel_weight_mode in {"rank_linear", "rank_gate", "rank_gate_graph"}:
        weights = build_rank_weights(len(selected), channel_weight_floor)
    return [int(item["index"]) for item in selected], weights


def apply_channel_subset(train_X, test_X, eu_adj, subset_indices, channel_weights=None, weight_mode="none"):
    subset_pairs = [(int(idx), None if channel_weights is None else float(weight)) for idx, weight in zip(subset_indices, channel_weights or [None] * len(subset_indices))]
    subset_pairs.sort(key=lambda item: item[0])
    subset_indices = [idx for idx, _ in subset_pairs]
    sorted_weights = None if channel_weights is None else [weight for _, weight in subset_pairs]
    index_tensor = torch.tensor(subset_indices, dtype=torch.long)
    train_X = train_X.index_select(1, index_tensor)
    test_X = test_X.index_select(1, index_tensor)
    if sorted_weights is not None and weight_mode == "rank_linear":
        weight_tensor = torch.tensor(sorted_weights, dtype=train_X.dtype).view(1, -1, 1)
        train_X = train_X * weight_tensor
        test_X = test_X * weight_tensor
    if isinstance(eu_adj, torch.Tensor):
        eu_adj = eu_adj.index_select(0, index_tensor).index_select(1, index_tensor)
    else:
        eu_adj = eu_adj[index_tensor.numpy()][:, index_tensor.numpy()]
    return train_X, test_X, eu_adj, index_tensor.tolist(), sorted_weights


def apply_graph_subset(static_adj, centrality, subset_indices):
    index_tensor = torch.tensor(subset_indices, dtype=torch.long)
    static_adj = static_adj.index_select(0, index_tensor).index_select(1, index_tensor)
    centrality = centrality[index_tensor.numpy()]
    return static_adj, centrality


def build_model(args, train_X, train_y, eu_adj, dataset_name, subset_indices, channel_weights=None):
    static_adj, centrality = load_adj(dataset_name)
    static_adj = torch.tensor(static_adj, dtype=torch.float32)
    static_adj, centrality = apply_graph_subset(static_adj, centrality.copy(), subset_indices)
    official = get_official_defaults(args.backbone)
    backbone_kwargs = dict(official["backbone_kwargs"])
    if args.backbone == "nexusnet":
        backbone_kwargs.update(
            {
                "Adj": static_adj,
                "eu_adj": eu_adj,
                "centrality": torch.tensor(centrality, dtype=torch.int64),
                "drop_prob": args.dropout,
                "dataset": dataset_name,
                "channel_indices": subset_indices,
            }
        )
        if args.channel_weight_mode in {"rank_gate", "rank_gate_graph"} and channel_weights is not None:
            backbone_kwargs["channel_gate_init"] = channel_weights
            backbone_kwargs["channel_gate_target"] = "graph" if args.channel_weight_mode == "rank_gate_graph" else "feature"
    elif args.backbone == "mshallowconvnet":
        backbone_kwargs["dropout"] = args.dropout
    elif args.backbone == "lggnet":
        backbone_kwargs["dropout"] = args.dropout
        backbone_kwargs["dataset"] = dataset_name
        backbone_kwargs["channel_indices"] = subset_indices
    return build_backbone(
        args.backbone,
        num_classes=len(torch.unique(train_y)),
        in_chans=train_X.shape[1],
        input_time_length=train_X.shape[2],
        backbone_kwargs=backbone_kwargs,
    )


def main():
    args = parse_args()
    set_seed(args.seed)
    dataset_name = infer_dataset_name(args.dataset)
    subset_indices, channel_weights = resolve_subset_spec(
        args.channel_subset_json,
        args.subset_topk,
        args.channel_weight_mode,
        args.channel_weight_floor,
    )
    train_X, train_y, test_X, test_y, eu_adj = load_single_subject(
        dataset=dataset_name,
        subject_id=args.subject_id,
        duration=args.duration,
        to_tensor=True,
    )
    train_X, test_X, eu_adj, subset_indices, channel_weights = apply_channel_subset(
        train_X,
        test_X,
        eu_adj,
        subset_indices,
        channel_weights,
        weight_mode=args.channel_weight_mode,
    )
    final_test_X = test_X[len(test_X) // 2 :]
    final_test_y = test_y[len(test_y) // 2 :]

    metrics_payloads = []
    if args.checkpoint_metrics_jsons:
        if len(args.checkpoint_metrics_jsons) != len(args.checkpoint_paths):
            raise ValueError("checkpoint_metrics_jsons must align with checkpoint_paths.")
        for metrics_path in args.checkpoint_metrics_jsons:
            with open(metrics_path, "r", encoding="utf-8") as handle:
                metrics_payloads.append(json.load(handle))
    else:
        metrics_payloads = [{"val_acc": 1.0} for _ in args.checkpoint_paths]

    entries = []
    for checkpoint_path, metrics in zip(args.checkpoint_paths, metrics_payloads):
        entries.append(
            {
                "checkpoint_path": checkpoint_path,
                "val_acc": float(metrics["val_acc"]),
            }
        )
    entries.sort(key=lambda item: item["val_acc"], reverse=True)
    if args.top_models is not None:
        entries = entries[: max(1, min(args.top_models, len(entries)))]

    if args.weighting == "val_acc":
        weight_tensor = torch.tensor([item["val_acc"] for item in entries], dtype=torch.float32)
        weights = (weight_tensor / weight_tensor.sum().clamp_min(1e-6)).tolist()
    else:
        weights = [1.0 / len(entries)] * len(entries)

    logits_sum = None
    model_count = 0
    for entry, weight in zip(entries, weights):
        checkpoint_path = entry["checkpoint_path"]
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(checkpoint_path)
        model = build_model(args, train_X, train_y, eu_adj, dataset_name, subset_indices, channel_weights).to(args.device)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_classifier"])
        model.eval()
        with torch.no_grad():
            logits, _ = model(final_test_X.to(args.device))
        weighted_logits = logits * float(weight)
        logits_sum = weighted_logits if logits_sum is None else logits_sum + weighted_logits
        model_count += 1

    ensemble_logits = logits_sum
    test_acc = (ensemble_logits.argmax(dim=1).cpu() == final_test_y).float().mean().item()

    payload = {
        "dataset": dataset_name,
        "subject_id": args.subject_id,
        "backbone": args.backbone,
        "subset_topk": args.subset_topk,
        "selected_indices": subset_indices,
        "selected_channels": [EEG_CHANNELS[dataset_name][idx] for idx in subset_indices],
        "channel_weight_mode": args.channel_weight_mode,
        "channel_weights": channel_weights,
        "checkpoint_paths": [os.path.abspath(path) for path in args.checkpoint_paths],
        "weighting": args.weighting,
        "weights": weights,
        "top_models": args.top_models,
        "used_checkpoint_paths": [os.path.abspath(item["checkpoint_path"]) for item in entries],
        "ensemble_size": model_count,
        "test_acc": test_acc,
        "params_per_model": count_parameters(build_model(args, train_X, train_y, eu_adj, dataset_name, subset_indices, channel_weights)),
        "avg_forward_seconds": benchmark_forward(
            build_model(args, train_X, train_y, eu_adj, dataset_name, subset_indices, channel_weights).to(args.device),
            final_test_X[:1],
            args.device,
        ),
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(payload)


if __name__ == "__main__":
    main()
