import argparse
import json
import os
from statistics import mean
from typing import List, Tuple

import torch

from models.NexusNet import NexusNet
from tools.datasets import EEG_CHANNELS, load_single_subject
from tools.utils import accuracy, load_adj


def build_model(dataset: str, subject_id: int, duration: float, device: str):
    train_X, train_y, test_X, test_y, eu_adj = load_single_subject(
        dataset=dataset,
        subject_id=subject_id,
        duration=duration,
        to_tensor=True,
    )
    adj, centrality = load_adj(dataset)
    model = NexusNet(
        flag=[1, 1, 1, 1],
        Adj=torch.tensor(adj, dtype=torch.float32),
        eu_adj=eu_adj,
        centrality=torch.tensor(centrality, dtype=torch.int64),
        in_chans=train_X.shape[1],
        n_classes=len(torch.unique(train_y)),
        input_time_length=train_X.shape[2],
        drop_prob=0.25,
        pool_mode="mean",
        f1=8,
        f2=16,
        kernel_length=64,
        dataset=dataset,
    ).to(device)
    return model, test_X.to(device), test_y.to(device)


def checkpoint_path(root: str, dataset_alias: str, subject_id: int) -> str:
    ckpt_dir = os.path.join(root, f"{dataset_alias}_checkpoint")
    return os.path.join(ckpt_dir, f"{subject_id}_train_0.pth.tar")


@torch.no_grad()
def eval_acc(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    logits, _ = model(x)
    acc, _ = accuracy(logits, y)
    return float(acc[0].item())


@torch.no_grad()
def rank_channels(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    channel_names: List[str],
) -> Tuple[float, List[dict]]:
    baseline_acc = eval_acc(model, x, y)
    ranking = []

    for idx, name in enumerate(channel_names):
        masked = x.clone()
        masked[:, idx] = 0.0
        masked_acc = eval_acc(model, masked, y)
        ranking.append(
            {
                "index": idx,
                "channel": name,
                "masked_acc": masked_acc,
                "importance_drop": baseline_acc - masked_acc,
            }
        )

    ranking.sort(key=lambda item: item["importance_drop"], reverse=True)
    return baseline_acc, ranking


@torch.no_grad()
def topk_mask_eval(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    ranked_indices: List[int],
    k_values: List[int],
) -> List[dict]:
    results = []
    total_channels = x.size(1)

    for k in k_values:
        keep = set(ranked_indices[:k])
        masked = x.clone()
        for idx in range(total_channels):
            if idx not in keep:
                masked[:, idx] = 0.0
        results.append(
            {
                "topk": k,
                "acc": eval_acc(model, masked, y),
            }
        )
    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", default="bciciv2a", choices=["bciciv2a", "bciciv2b"])
    parser.add_argument("-subject_id", type=int, default=0)
    parser.add_argument("-duration", type=float, default=4.0)
    parser.add_argument("-device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("-topk", type=int, nargs="*", default=[3, 5, 8, 10])
    parser.add_argument("-output", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = "BNCI2014001" if args.dataset == "bciciv2a" else "BNCI2014004"
    channel_names = EEG_CHANNELS[dataset]
    subject_ids = [args.subject_id] if args.subject_id > 0 else list(range(1, 10))
    per_subject = []

    for subject_id in subject_ids:
        model, test_X, test_y = build_model(dataset, subject_id, args.duration, args.device)
        ckpt = checkpoint_path(os.path.dirname(__file__), args.dataset, subject_id)
        checkpoint = torch.load(ckpt, map_location=args.device)
        model.load_state_dict(checkpoint["model_classifier"])
        model.eval()

        baseline_acc, ranking = rank_channels(model, test_X, test_y, channel_names)
        ranked_indices = [item["index"] for item in ranking]
        topk_results = topk_mask_eval(model, test_X, test_y, ranked_indices, args.topk)
        per_subject.append(
            {
                "subject_id": subject_id,
                "baseline_acc": baseline_acc,
                "ranking": ranking,
                "topk_results": topk_results,
            }
        )

    summary = {
        "baseline_mean_acc": mean(item["baseline_acc"] for item in per_subject),
    }
    for k in args.topk:
        vals = []
        for item in per_subject:
            for topk_result in item["topk_results"]:
                if topk_result["topk"] == k:
                    vals.append(topk_result["acc"])
                    break
        summary[f"top{k}_mean_acc"] = mean(vals)

    payload = {
        "dataset": dataset,
        "subjects": subject_ids,
        "summary": summary,
        "per_subject": per_subject,
    }

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
