import argparse
import json
import os

import torch

from models.IoTChannelSelectionFramework import (
    GraphGuidedChannelSelector,
    ModelAgnosticChannelSelectionWrapper,
    build_backbone,
)
from tools.channel_selection import ranking_to_channels, ranking_to_indices
from tools.complexity import benchmark_forward, count_parameters
from tools.datasets import EEG_CHANNELS, load_single_subject
from tools.utils import load_adj


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", default="bciciv2a", choices=["bciciv2a", "bciciv2b"])
    parser.add_argument("-subject_id", type=int, default=1)
    parser.add_argument("-duration", type=float, default=4.0)
    parser.add_argument("-backbone", default="nexusnet", choices=["nexusnet", "shallowconvnet", "lggnet"])
    parser.add_argument("-checkpoint", required=True)
    parser.add_argument("-selector_hidden", type=int, default=64)
    parser.add_argument("-selector_layers", type=int, default=2)
    parser.add_argument("-selector_dropout", type=float, default=0.1)
    parser.add_argument("-dropout", type=float, default=0.25)
    parser.add_argument("-topk", type=int, default=5)
    parser.add_argument("-device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("-output", default="iot_eval_s1.json")
    return parser.parse_args()


def build_model(args, train_X, train_y, eu_adj, dataset_name):
    static_adj, centrality = load_adj(dataset_name)
    static_adj = torch.tensor(static_adj, dtype=torch.float32)
    num_classes = len(torch.unique(train_y))
    backbone_kwargs = {"drop_prob": args.dropout}
    if args.backbone == "nexusnet":
        backbone_kwargs.update(
            {
                "Adj": static_adj,
                "eu_adj": eu_adj,
                "centrality": torch.tensor(centrality, dtype=torch.int64),
                "pool_mode": "mean",
                "f1": 8,
                "f2": 16,
                "kernel_length": 64,
                "dataset": dataset_name,
            }
        )
    elif args.backbone == "shallowconvnet":
        backbone_kwargs = {
            "temporal_filters": 16,
            "spatial_filters": 32,
            "kernel_length": 25,
            "dropout": args.dropout,
        }
    elif args.backbone == "lggnet":
        backbone_kwargs = {
            "sampling_rate": 250,
            "num_t_filters": 16,
            "out_graph": 16,
            "pool": 16,
            "pool_step_rate": 0.25,
            "dropout": args.dropout,
        }
    else:
        backbone_kwargs = {"hidden_dim": 32, "dropout": args.dropout}

    selector = GraphGuidedChannelSelector(
        in_chans=train_X.shape[1],
        static_adj=static_adj,
        hidden_dim=args.selector_hidden,
        num_layers=args.selector_layers,
        dropout=args.selector_dropout,
        topk=args.topk,
        use_dynamic_graph=True,
    )
    backbone = build_backbone(
        args.backbone,
        num_classes=num_classes,
        in_chans=train_X.shape[1],
        input_time_length=train_X.shape[2],
        backbone_kwargs=backbone_kwargs,
    )
    return ModelAgnosticChannelSelectionWrapper(
        selector=selector,
        backbone=backbone,
        graph_decoder_dim=args.selector_hidden,
        num_classes=num_classes,
    )


def main():
    args = parse_args()
    dataset_name = "BNCI2014001" if args.dataset == "bciciv2a" else "BNCI2014004"
    train_X, train_y, test_X, test_y, eu_adj = load_single_subject(
        dataset=dataset_name,
        subject_id=args.subject_id,
        duration=args.duration,
        to_tensor=True,
    )
    model = build_model(args, train_X, train_y, eu_adj, dataset_name).to(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=torch.device(args.device))
    model.load_state_dict(checkpoint["model_classifier"])
    model.eval()

    final_test_X = test_X[len(test_X) // 2 :]
    final_test_y = test_y[len(test_y) // 2 :]
    with torch.no_grad():
        logits, aux = model(final_test_X.to(args.device), use_hard_mask=True)
        test_acc = (logits.argmax(dim=1).cpu() == final_test_y).float().mean().item()
        channel_scores = aux["channel_scores"].mean(dim=0).cpu().tolist()
        channel_mask = aux["channel_mask"].mean(dim=0).cpu().tolist()

    ranking = sorted(
        [
            {
                "index": idx,
                "channel": EEG_CHANNELS[dataset_name][idx],
                "score": channel_scores[idx],
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
        "backbone": args.backbone,
        "topk": args.topk,
        "checkpoint": os.path.abspath(args.checkpoint),
        "test_acc": test_acc,
        "params": count_parameters(model),
        "avg_forward_seconds": benchmark_forward(model, final_test_X[:1], args.device),
        "selected_channels": ranking_to_channels(ranking, args.topk),
        "selected_indices": ranking_to_indices(ranking, args.topk),
        "ranking": ranking,
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(payload)


if __name__ == "__main__":
    main()
