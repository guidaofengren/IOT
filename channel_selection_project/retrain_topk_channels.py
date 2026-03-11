import argparse
import json
import os
from statistics import mean

import torch

from models.NexusNet import NexusNet
from tools.datasets import load_single_subject
from tools.run_tools import evaluate_one_epoch_classifier, train_one_epoch_classifier
from tools.utils import (
    BalancedBatchSizeIterator,
    EarlyStopping,
    accuracy,
    load_adj,
    save,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", default="bciciv2a", choices=["bciciv2a", "bciciv2b"])
    parser.add_argument("-subject_id", type=int, default=0)
    parser.add_argument("-duration", type=float, default=4.0)
    parser.add_argument("-device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("-epochs", type=int, default=300)
    parser.add_argument("-batch_size", type=int, default=32)
    parser.add_argument("-lr", type=float, default=1e-3)
    parser.add_argument("-w_decay", type=float, default=1e-4)
    parser.add_argument("-dropout", type=float, default=0.25)
    parser.add_argument("-seed", type=int, default=42)
    parser.add_argument("-topk", type=int, nargs="*", default=[3, 5])
    parser.add_argument("-ranking_json", default="channel_importance_bciciv2a_all.json")
    parser.add_argument("-output", default="topk_retrain_results.json")
    return parser.parse_args()


def load_rankings(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {item["subject_id"]: item for item in data["per_subject"]}


def keep_topk_channels(x: torch.Tensor, keep_indices):
    masked = x.clone()
    keep = set(keep_indices)
    for idx in range(masked.size(1)):
        if idx not in keep:
            masked[:, idx] = 0.0
    return masked


def build_model(train_X, train_y, eu_adj, dataset_name: str, dropout: float, device: str):
    adj, centrality = load_adj(dataset_name)
    model = NexusNet(
        flag=[1, 1, 1, 1],
        Adj=torch.tensor(adj, dtype=torch.float32),
        eu_adj=eu_adj,
        centrality=torch.tensor(centrality, dtype=torch.int64),
        in_chans=train_X.shape[1],
        n_classes=len(torch.unique(train_y)),
        input_time_length=train_X.shape[2],
        drop_prob=dropout,
        pool_mode="mean",
        f1=8,
        f2=16,
        kernel_length=64,
        dataset=dataset_name,
    ).to(device)
    return model


def eval_model(model, x, y):
    model.eval()
    with torch.no_grad():
        logits, _ = model(x)
        acc, _ = accuracy(logits, y)
    return float(acc[0].item())


def train_for_subject(args, dataset_name: str, subject_id: int, ranking_entry):
    train_X, train_y, test_X, test_y, eu_adj = load_single_subject(
        dataset=dataset_name,
        subject_id=subject_id,
        duration=args.duration,
        to_tensor=True,
    )
    train_X = train_X.to(args.device)
    train_y = train_y.to(args.device)
    test_X = test_X.to(args.device)
    test_y = test_y.to(args.device)

    val_X = test_X[: len(test_X) // 2]
    val_y = test_y[: len(test_y) // 2]
    final_test_X = test_X[len(test_X) // 2 :]
    final_test_y = test_y[len(test_y) // 2 :]
    iterator = BalancedBatchSizeIterator(batch_size=args.batch_size, seed=args.seed)
    criterion = torch.nn.CrossEntropyLoss().to(args.device)

    full_model = build_model(train_X, train_y, eu_adj, dataset_name, args.dropout, args.device)
    optimizer = torch.optim.Adam(full_model.parameters(), lr=args.lr, weight_decay=args.w_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs // 4))
    stopper = EarlyStopping(patience=max(40, args.epochs // 3), max_epochs=args.epochs)

    best_state = None
    best_acc = -1.0
    for _ in range(args.epochs):
        if stopper.early_stop:
            break
        train_one_epoch_classifier(iterator, (train_X, train_y), full_model, args.device, optimizer, criterion)
        scheduler.step()
        val_acc = evaluate_one_epoch_classifier(iterator, (val_X, val_y), full_model, args.device, criterion)
        stopper(val_acc)
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu() for k, v in full_model.state_dict().items()}

    if best_state is not None:
        full_model.load_state_dict(best_state)
    full_test_acc = eval_model(full_model, final_test_X, final_test_y)

    ranked_indices = [item["index"] for item in ranking_entry["ranking"]]
    topk_results = []
    for k in args.topk:
        top_indices = ranked_indices[:k]
        top_train_X = keep_topk_channels(train_X, top_indices)
        top_val_X = keep_topk_channels(val_X, top_indices)
        top_test_X = keep_topk_channels(final_test_X, top_indices)

        model = build_model(top_train_X, train_y, eu_adj, dataset_name, args.dropout, args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.w_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs // 4))
        stopper = EarlyStopping(patience=max(40, args.epochs // 3), max_epochs=args.epochs)

        best_state = None
        best_acc = -1.0
        for _ in range(args.epochs):
            if stopper.early_stop:
                break
            train_one_epoch_classifier(iterator, (top_train_X, train_y), model, args.device, optimizer, criterion)
            scheduler.step()
            val_acc = evaluate_one_epoch_classifier(iterator, (top_val_X, val_y), model, args.device, criterion)
            stopper(val_acc)
            if val_acc > best_acc:
                best_acc = val_acc
                best_state = {name: value.detach().cpu() for name, value in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)
        test_acc = eval_model(model, top_test_X, final_test_y)
        topk_results.append({"topk": k, "test_acc": test_acc, "channels": top_indices})

    return {
        "subject_id": subject_id,
        "full_test_acc": full_test_acc,
        "topk_results": topk_results,
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    dataset_name = "BNCI2014001" if args.dataset == "bciciv2a" else "BNCI2014004"
    rankings = load_rankings(args.ranking_json)
    subject_ids = [args.subject_id] if args.subject_id > 0 else sorted(rankings.keys())

    per_subject = []
    for subject_id in subject_ids:
        result = train_for_subject(args, dataset_name, subject_id, rankings[subject_id])
        per_subject.append(result)
        print(result)

    summary = {
        "full_mean_acc": mean(item["full_test_acc"] for item in per_subject),
    }
    for k in args.topk:
        vals = []
        for item in per_subject:
            vals.append(next(x["test_acc"] for x in item["topk_results"] if x["topk"] == k))
        summary[f"top{k}_mean_acc"] = mean(vals)

    payload = {
        "dataset": dataset_name,
        "subjects": subject_ids,
        "summary": summary,
        "per_subject": per_subject,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(payload["summary"])


if __name__ == "__main__":
    main()
