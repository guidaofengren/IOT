import argparse
import json
import os

import torch

from models.GraphChannelSelectionNexusNet import GraphChannelSelectionNexusNet
from tools.datasets import EEG_CHANNELS, load_single_subject
from tools.run_tools import evaluate_one_epoch_classifier
from tools.utils import BalancedBatchSizeIterator, EarlyStopping, load_adj, save, set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", default="bciciv2a", choices=["bciciv2a", "bciciv2b"])
    parser.add_argument("-subject_id", type=int, default=1)
    parser.add_argument("-duration", type=float, default=4.0)
    parser.add_argument("-epochs", type=int, default=300)
    parser.add_argument("-batch_size", type=int, default=32)
    parser.add_argument("-lr", type=float, default=1e-3)
    parser.add_argument("-w_decay", type=float, default=1e-4)
    parser.add_argument("-dropout", type=float, default=0.25)
    parser.add_argument("-selector_hidden", type=int, default=64)
    parser.add_argument("-selector_dropout", type=float, default=0.1)
    parser.add_argument("-sparse_lambda", type=float, default=1e-3)
    parser.add_argument("-device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("-output", default="gnn_channel_selector_result.json")
    parser.add_argument("-seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    dataset_name = "BNCI2014001" if args.dataset == "bciciv2a" else "BNCI2014004"

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

    adj, centrality = load_adj(dataset_name)
    model = GraphChannelSelectionNexusNet(
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
        hidden_dim=args.selector_hidden,
        selector_dropout=args.selector_dropout,
    ).to(args.device)

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
        model.train()
        for features, labels in iterator.get_batches(train_X, train_y, shuffle=True):
            features = features.to(args.device)
            labels = labels.to(args.device)
            optimizer.zero_grad()
            logits, aux = model(features)
            sparse_penalty = aux["channel_weights"].mean()
            loss = criterion(logits, labels) + args.sparse_lambda * sparse_penalty
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
    model.eval()
    with torch.no_grad():
        logits, aux = model(final_test_X.to(args.device))
        test_acc = (logits.argmax(dim=1).cpu() == final_test_y).float().mean().item()
        channel_weights = aux["channel_weights"].mean(dim=0).detach().cpu().tolist()

    ranking = sorted(
        [
            {"index": idx, "channel": EEG_CHANNELS[dataset_name][idx], "weight": weight}
            for idx, weight in enumerate(channel_weights)
        ],
        key=lambda item: item["weight"],
        reverse=True,
    )

    payload = {
        "dataset": dataset_name,
        "subject_id": args.subject_id,
        "val_acc": best_acc,
        "test_acc": test_acc,
        "ranking": ranking,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    ckpt_path = os.path.join(os.path.dirname(__file__), f"gnn_channel_selector_s{args.subject_id}.pth.tar")
    save({"model_classifier": model.state_dict(), "acc": best_acc}, ckpt_path)
    print(payload)


if __name__ == "__main__":
    main()
