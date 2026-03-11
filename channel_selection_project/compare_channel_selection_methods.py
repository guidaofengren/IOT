import argparse
import json
import os
import random
from statistics import mean

import torch

from models.ChannelSelectionNexusNet import ChannelSelectionNexusNet
from models.NexusNet import NexusNet
from tools.datasets import EEG_CHANNELS, load_single_subject
from tools.eeg_graph_features import extract_node_features
from tools.run_tools import evaluate_one_epoch_classifier, train_one_epoch_classifier
from tools.utils import BalancedBatchSizeIterator, EarlyStopping, load_adj, set_seed


MANUAL_CHANNELS = {
    "BNCI2014001": {
        3: ["C3", "Cz", "C4"],
        5: ["C3", "Cz", "C4", "CP3", "CP4"],
    },
    "BNCI2014004": {
        3: ["C3", "Cz", "C4"],
        5: ["C3", "Cz", "C4"],
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", default="bciciv2a", choices=["bciciv2a", "bciciv2b"])
    parser.add_argument("-subject_id", type=int, default=1)
    parser.add_argument("-duration", type=float, default=4.0)
    parser.add_argument("-epochs", type=int, default=200)
    parser.add_argument("-batch_size", type=int, default=32)
    parser.add_argument("-lr", type=float, default=5e-3)
    parser.add_argument("-w_decay", type=float, default=0.0)
    parser.add_argument("-dropout", type=float, default=0.25)
    parser.add_argument("-device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("-seed", type=int, default=42)
    parser.add_argument("-topk", type=int, nargs="*", default=[3, 5])
    parser.add_argument("-random_trials", type=int, default=5)
    parser.add_argument("-selector_epochs", type=int, default=120)
    parser.add_argument("-selector_lr", type=float, default=1e-3)
    parser.add_argument("-selector_sparse_lambda", type=float, default=1e-3)
    parser.add_argument("-ranking_json", default="")
    parser.add_argument("-output", default="channel_method_compare.json")
    return parser.parse_args()


def build_backbone(train_X, train_y, eu_adj, dataset_name, dropout, device):
    adj, centrality = load_adj(dataset_name)
    return NexusNet(
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


def train_model(model, train_X, train_y, val_X, val_y, args):
    iterator = BalancedBatchSizeIterator(batch_size=args.batch_size, seed=args.seed)
    criterion = torch.nn.CrossEntropyLoss().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.w_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs // 4))
    stopper = EarlyStopping(patience=max(40, args.epochs // 3), max_epochs=args.epochs)

    best_state = None
    best_acc = -1.0
    for _ in range(args.epochs):
        if stopper.early_stop:
            break
        train_one_epoch_classifier(iterator, (train_X, train_y), model, args.device, optimizer, criterion)
        scheduler.step()
        val_acc = evaluate_one_epoch_classifier(iterator, (val_X, val_y), model, args.device, criterion)
        stopper(val_acc)
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_acc


def eval_model(model, x, y):
    model.eval()
    with torch.no_grad():
        logits, _ = model(x.to(next(model.parameters()).device))
        pred = logits.argmax(dim=1).cpu()
        return float((pred == y.cpu()).float().mean().item())


def keep_indices(x: torch.Tensor, keep_indices):
    masked = x.clone()
    keep = set(keep_indices)
    for idx in range(masked.size(1)):
        if idx not in keep:
            masked[:, idx] = 0.0
    return masked


def fisher_channel_ranking(train_X: torch.Tensor, train_y: torch.Tensor):
    feats = extract_node_features(train_X.cpu()).numpy()  # [N, C, 7]
    labels = train_y.cpu().numpy()
    n_channels = feats.shape[1]
    scores = []
    classes = sorted(set(labels.tolist()))
    for ch in range(n_channels):
        ch_feat = feats[:, ch, :]
        overall_mean = ch_feat.mean(axis=0)
        between = 0.0
        within = 0.0
        for cls in classes:
            cls_feat = ch_feat[labels == cls]
            cls_mean = cls_feat.mean(axis=0)
            between += cls_feat.shape[0] * ((cls_mean - overall_mean) ** 2).mean()
            within += ((cls_feat - cls_mean) ** 2).mean()
        score = float(between / (within + 1e-6))
        scores.append((ch, score))
    scores.sort(key=lambda item: item[1], reverse=True)
    return [idx for idx, _ in scores]


def load_posthoc_ranking(ranking_json: str, subject_id: int):
    if not os.path.exists(ranking_json):
        candidate = os.path.join(os.path.dirname(__file__), "..", ranking_json)
        if os.path.exists(candidate):
            ranking_json = candidate
    with open(ranking_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data["per_subject"]:
        if item["subject_id"] == subject_id:
            return [rank["index"] for rank in item["ranking"]]
    raise ValueError(f"Subject {subject_id} not found in ranking json")


def train_selector_ranking(train_X, train_y, val_X, val_y, eu_adj, dataset_name, args):
    adj, centrality = load_adj(dataset_name)
    model = ChannelSelectionNexusNet(
        flag=[1, 1, 1, 1],
        Adj=torch.tensor(adj, dtype=torch.float32),
        eu_adj=eu_adj,
        centrality=torch.tensor(centrality, dtype=torch.int64),
        in_chans=train_X.shape[1],
        n_classes=len(torch.unique(train_y)),
        input_time_length=train_X.shape[2],
        drop_prob=args.dropout,
        pool_mode="mean",
        f1=8,
        f2=16,
        kernel_length=64,
        dataset=dataset_name,
    ).to(args.device)

    iterator = BalancedBatchSizeIterator(batch_size=args.batch_size, seed=args.seed)
    criterion = torch.nn.CrossEntropyLoss().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.selector_lr, weight_decay=args.w_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.selector_epochs // 4))
    stopper = EarlyStopping(patience=max(30, args.selector_epochs // 3), max_epochs=args.selector_epochs)
    best_state = None
    best_acc = -1.0

    for _ in range(args.selector_epochs):
        if stopper.early_stop:
            break
        model.train()
        for features, labels in iterator.get_batches(train_X, train_y, shuffle=True):
            optimizer.zero_grad()
            logits, aux = model(features)
            loss = criterion(logits, labels) + args.selector_sparse_lambda * aux["channel_weights"].mean()
            loss.backward()
            optimizer.step()
        scheduler.step()
        val_acc = evaluate_one_epoch_classifier(iterator, (val_X, val_y), model, args.device, criterion)
        stopper(val_acc)
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    weights = model.channel_weights().detach().cpu().tolist()
    ranking = sorted(range(len(weights)), key=lambda idx: weights[idx], reverse=True)
    return ranking, best_acc


def method_indices(method_name, k, channel_names, args, train_X, train_y, val_X, val_y, eu_adj, dataset_name):
    if method_name == "manual":
        manual = MANUAL_CHANNELS[dataset_name][3 if k == 3 else 5]
        return [channel_names.index(ch) for ch in manual if ch in channel_names]
    if method_name == "traditional":
        return fisher_channel_ranking(train_X, train_y)[:k]
    if method_name == "posthoc":
        return load_posthoc_ranking(args.ranking_json, args.subject_id)[:k]
    if method_name == "train_time":
        ranking, _ = train_selector_ranking(train_X, train_y, val_X, val_y, eu_adj, dataset_name, args)
        return ranking[:k]
    raise ValueError(method_name)


def retrain_with_indices(indices, train_X, train_y, val_X, val_y, test_X, test_y, eu_adj, dataset_name, args):
    sub_train = keep_indices(train_X, indices)
    sub_val = keep_indices(val_X, indices)
    sub_test = keep_indices(test_X, indices)
    model = build_backbone(sub_train, train_y, eu_adj, dataset_name, args.dropout, args.device)
    model, val_acc = train_model(model, sub_train, train_y, sub_val, val_y, args)
    test_acc = eval_model(model, sub_test, test_y)
    return val_acc, test_acc


def main():
    args = parse_args()
    set_seed(args.seed)
    dataset_name = "BNCI2014001" if args.dataset == "bciciv2a" else "BNCI2014004"
    channel_names = EEG_CHANNELS[dataset_name]
    if args.ranking_json == "":
        args.ranking_json = "channel_importance_bciciv2a_all.json"

    train_X, train_y, test_X, test_y, eu_adj = load_single_subject(
        dataset=dataset_name,
        subject_id=args.subject_id,
        duration=args.duration,
        to_tensor=True,
    )
    val_X = test_X[: len(test_X) // 2]
    val_y = test_y[: len(test_y) // 2]
    final_test_X = test_X[len(test_X) // 2 :]
    final_test_y = test_y[len(test_y) // 2 :]

    full_model = build_backbone(train_X, train_y, eu_adj, dataset_name, args.dropout, args.device)
    full_model, full_val_acc = train_model(full_model, train_X, train_y, val_X, val_y, args)
    full_test_acc = eval_model(full_model, final_test_X, final_test_y)

    results = {
        "dataset": dataset_name,
        "subject_id": args.subject_id,
        "full_channel": {"val_acc": full_val_acc, "test_acc": full_test_acc},
        "methods": {},
    }

    for method in ["manual", "traditional", "posthoc", "train_time"]:
        results["methods"][method] = {}
        for k in args.topk:
            indices = method_indices(method, k, channel_names, args, train_X, train_y, val_X, val_y, eu_adj, dataset_name)
            val_acc, test_acc = retrain_with_indices(
                indices,
                train_X,
                train_y,
                val_X,
                val_y,
                final_test_X,
                final_test_y,
                eu_adj,
                dataset_name,
                args,
            )
            results["methods"][method][str(k)] = {
                "indices": indices,
                "channels": [channel_names[idx] for idx in indices],
                "val_acc": val_acc,
                "test_acc": test_acc,
            }

    random_results = {}
    for k in args.topk:
        vals = []
        channel_sets = []
        for trial in range(args.random_trials):
            rng = random.Random(args.seed + 1000 * k + trial)
            indices = sorted(rng.sample(list(range(len(channel_names))), k=min(k, len(channel_names))))
            _, test_acc = retrain_with_indices(
                indices,
                train_X,
                train_y,
                val_X,
                val_y,
                final_test_X,
                final_test_y,
                eu_adj,
                dataset_name,
                args,
            )
            vals.append(test_acc)
            channel_sets.append([channel_names[idx] for idx in indices])
        random_results[str(k)] = {
            "mean_test_acc": mean(vals),
            "all_test_acc": vals,
            "channel_sets": channel_sets,
        }
    results["methods"]["random"] = random_results

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
