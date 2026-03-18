import argparse
import json
import os
from collections import deque

import torch
import torch.nn.functional as F

from models.IoTChannelSelectionFramework import build_backbone, transfer_backbone_weights
from official_profiles import apply_missing_training_defaults, get_official_defaults
from tools.complexity import benchmark_forward, count_parameters
from tools.datasets import EEG_CHANNELS, infer_dataset_name, load_single_subject
from tools.run_tools import evaluate_one_epoch_classifier
from tools.utils import BalancedBatchSizeIterator, EarlyStopping, load_adj, save, set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", default="bciciv2a", choices=["bciciv2a", "bciciv2b"])
    parser.add_argument("-subject_id", type=int, default=1)
    parser.add_argument("-duration", type=float, default=4.0)
    parser.add_argument("-epochs", type=int, default=220)
    parser.add_argument("-patience", type=int, default=35)
    parser.add_argument("-batch_size", type=int, default=64)
    parser.add_argument("-lr", type=float, default=0.0018)
    parser.add_argument("-w_decay", type=float, default=None)
    parser.add_argument("-dropout", type=float, default=0.22)
    parser.add_argument("-backbone", default="nexusnet", choices=["nexusnet", "mshallowconvnet", "lggnet"])
    parser.add_argument("-device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("-output", default="iot_baseline_s1.json")
    parser.add_argument("-seed", type=int, default=42)
    parser.add_argument("-channel_indices", type=int, nargs="+", default=None)
    parser.add_argument("-channel_subset_json", default=None)
    parser.add_argument("-subset_topk", type=int, default=None)
    parser.add_argument("-init_from_full", action="store_true")
    parser.add_argument("-full_checkpoint", default=None)
    parser.add_argument("-distill_alpha", type=float, default=0.0)
    parser.add_argument("-distill_temp", type=float, default=2.0)
    parser.add_argument("-min_epochs", type=int, default=90)
    parser.add_argument("-val_smooth_window", type=int, default=8)
    parser.add_argument("-grad_clip", type=float, default=1.0)
    parser.add_argument("-ema_decay", type=float, default=0.99)
    parser.add_argument("-channel_weight_mode", default="none", choices=["none", "rank_linear", "rank_gate", "rank_gate_graph"])
    parser.add_argument("-channel_weight_floor", type=float, default=0.6)
    parser.add_argument("-subset_channel_dropout", type=float, default=0.0)
    parser.add_argument("-subset_drop_lowrank_first", action="store_true")
    parser.add_argument("-subset_warmup_epochs", type=int, default=0)
    parser.add_argument("-subset_warmup_mode", default="cls", choices=["cls", "cls_gate"])
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


def resolve_subset_spec(args):
    if args.channel_indices:
        return list(args.channel_indices), None
    if not args.channel_subset_json:
        return None, None
    with open(args.channel_subset_json, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if "subsets" in payload:
        topk_key = str(args.subset_topk) if args.subset_topk is not None else None
        if topk_key is None:
            raise ValueError("subset_topk is required when channel_subset_json points to a subsets payload.")
        subset = payload["subsets"][topk_key]
        return list(subset["indices"]), None

    ranking = payload.get("ranking")
    if ranking is None:
        raise ValueError("channel_subset_json must contain either 'subsets' or 'ranking'.")
    topk = args.subset_topk if args.subset_topk is not None else payload.get("topk")
    if topk is None:
        raise ValueError("subset_topk is required when channel_subset_json points to a ranking payload.")
    selected = ranking[:topk]
    weights = None
    if args.channel_weight_mode in {"rank_linear", "rank_gate", "rank_gate_graph"}:
        weights = build_rank_weights(len(selected), args.channel_weight_floor)
    return [int(item["index"]) for item in selected], weights


def apply_channel_subset(train_X, test_X, eu_adj, subset_indices, channel_weights=None, weight_mode="none"):
    if subset_indices is None:
        return train_X, test_X, eu_adj, subset_indices, None
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
        eu_adj = eu_adj[subset_indices][:, subset_indices]
    return train_X, test_X, eu_adj, subset_indices, sorted_weights


def apply_graph_subset(static_adj, centrality, subset_indices):
    if subset_indices is None:
        return static_adj, centrality
    if isinstance(static_adj, torch.Tensor):
        index_tensor = torch.tensor(subset_indices, dtype=torch.long)
        static_adj = static_adj.index_select(0, index_tensor).index_select(1, index_tensor)
    else:
        static_adj = static_adj[subset_indices][:, subset_indices]
    centrality = centrality[subset_indices]
    return static_adj, centrality


def build_baseline(args, train_X, train_y, eu_adj, dataset_name, subset_indices=None, channel_weights=None):
    static_adj, centrality = load_adj(dataset_name)
    static_adj = torch.tensor(static_adj, dtype=torch.float32)
    centrality = centrality.copy()
    static_adj, centrality = apply_graph_subset(static_adj, centrality, subset_indices)
    num_classes = len(torch.unique(train_y))
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
    else:
        backbone_kwargs = {"hidden_dim": 32, "dropout": args.dropout}
    return build_backbone(
        args.backbone,
        num_classes=num_classes,
        in_chans=train_X.shape[1],
        input_time_length=train_X.shape[2],
        backbone_kwargs=backbone_kwargs,
    ), backbone_kwargs


def build_full_teacher(args, full_train_X, train_y, full_eu_adj, dataset_name):
    static_adj, centrality = load_adj(dataset_name)
    static_adj = torch.tensor(static_adj, dtype=torch.float32)
    num_classes = len(torch.unique(train_y))
    official = get_official_defaults(args.backbone)
    backbone_kwargs = dict(official["backbone_kwargs"])
    if args.backbone == "nexusnet":
        backbone_kwargs.update(
            {
                "Adj": static_adj,
                "eu_adj": full_eu_adj,
                "centrality": torch.tensor(centrality, dtype=torch.int64),
                "drop_prob": args.dropout,
                "dataset": dataset_name,
            }
        )
    elif args.backbone == "mshallowconvnet":
        backbone_kwargs["dropout"] = args.dropout
    elif args.backbone == "lggnet":
        backbone_kwargs["dropout"] = args.dropout
        backbone_kwargs["dataset"] = dataset_name
    teacher = build_backbone(
        args.backbone,
        num_classes=num_classes,
        in_chans=full_train_X.shape[1],
        input_time_length=full_train_X.shape[2],
        backbone_kwargs=backbone_kwargs,
    )
    checkpoint_candidates = []
    if args.full_checkpoint is not None:
        checkpoint_candidates.append(args.full_checkpoint)
    checkpoint_candidates.extend(
        [
            os.path.join(
                os.path.dirname(__file__),
                f"iot_baseline_{args.dataset}_{args.backbone}_full_s{args.subject_id}.pth.tar",
            ),
            os.path.join(
                os.path.dirname(__file__),
                f"iot_baseline_{args.dataset}_{args.backbone}_s{args.subject_id}.pth.tar",
            ),
        ]
    )
    checkpoint_path = None
    for candidate in checkpoint_candidates:
        if os.path.exists(candidate):
            checkpoint_path = candidate
            break
    if checkpoint_path is None:
        return None, checkpoint_candidates[0]
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    teacher.load_state_dict(checkpoint["model_classifier"])
    return teacher.to(args.device).eval(), checkpoint_path


def distillation_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float):
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)


def expand_subset_features(subset_features: torch.Tensor, full_in_chans: int, subset_indices) -> torch.Tensor:
    full = subset_features.new_zeros(subset_features.size(0), full_in_chans, subset_features.size(2))
    scatter_index = torch.tensor(subset_indices, device=subset_features.device, dtype=torch.long).view(1, -1, 1)
    scatter_index = scatter_index.expand(subset_features.size(0), -1, subset_features.size(2))
    return full.scatter_(1, scatter_index, subset_features)


def clone_state_dict(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def update_ema_state(ema_state, model, decay):
    model_state = model.state_dict()
    for key, value in model_state.items():
        detached = value.detach().cpu()
        if not detached.is_floating_point():
            ema_state[key] = detached.clone()
            continue
        ema_state[key].mul_(decay).add_(detached, alpha=1.0 - decay)


def apply_subset_channel_dropout(features, dropout_rate, prioritize_low_rank=False):
    if dropout_rate <= 0.0 or features.size(1) <= 1:
        return features
    keep_prob = 1.0 - float(dropout_rate)
    if prioritize_low_rank:
        channel_count = features.size(1)
        rank_bias = torch.linspace(1.0, 0.4, steps=channel_count, device=features.device, dtype=features.dtype)
        keep_probs = (keep_prob * rank_bias).clamp(min=0.2, max=0.995)
        mask = torch.bernoulli(keep_probs.view(1, channel_count, 1).expand(features.size(0), -1, features.size(2)))
    else:
        mask = torch.bernoulli(
            torch.full_like(features, keep_prob)
        )
    min_active = max(1, features.size(1) // 2)
    active_counts = mask[:, :, 0].sum(dim=1)
    if torch.any(active_counts < min_active):
        fallback = torch.ones_like(mask[:, :, 0])
        if prioritize_low_rank:
            fallback[:, min_active:] = 0.0
        else:
            fallback[:, :min_active] = 1.0
        underfilled = active_counts < min_active
        mask[underfilled] = fallback[underfilled].unsqueeze(-1).expand(-1, -1, features.size(2))
    mask = mask / mask.mean(dim=1, keepdim=True).clamp_min(1e-6)
    return features * mask


def configure_subset_warmup(model, active, mode):
    if not active:
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        return

    for name, parameter in model.named_parameters():
        trainable = name.startswith("cls.")
        if mode == "cls_gate" and name == "channel_gate_logits":
            trainable = True
        parameter.requires_grad_(trainable)


def main():
    args = parse_args()
    args = apply_missing_training_defaults(args)
    set_seed(args.seed)
    dataset_name = infer_dataset_name(args.dataset)
    subset_indices, channel_weights = resolve_subset_spec(args)
    full_train_X, train_y, full_test_X, test_y, full_eu_adj = load_single_subject(
        dataset=dataset_name,
        subject_id=args.subject_id,
        duration=args.duration,
        to_tensor=True,
    )
    train_X, test_X, eu_adj, subset_indices, channel_weights = apply_channel_subset(
        full_train_X,
        full_test_X,
        full_eu_adj,
        subset_indices,
        channel_weights,
        weight_mode=args.channel_weight_mode,
    )
    val_X = test_X[: len(test_X) // 2]
    val_y = test_y[: len(test_y) // 2]
    final_test_X = test_X[len(test_X) // 2 :]
    final_test_y = test_y[len(test_y) // 2 :]

    model, backbone_kwargs = build_baseline(
        args,
        train_X,
        train_y,
        eu_adj,
        dataset_name,
        subset_indices=subset_indices,
        channel_weights=channel_weights,
    )
    model = model.to(args.device)
    teacher, teacher_checkpoint = build_full_teacher(args, full_train_X, train_y, full_eu_adj, dataset_name)
    if args.init_from_full and subset_indices is not None and teacher is not None:
        transfer_backbone_weights(args.backbone, teacher, model, subset_indices)
    iterator = BalancedBatchSizeIterator(batch_size=args.batch_size, seed=args.seed)
    criterion = torch.nn.CrossEntropyLoss().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.w_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs // 4))
    stopper = EarlyStopping(patience=args.patience, max_epochs=args.epochs)

    best_state = None
    best_acc = -1.0
    best_epoch = -1
    raw_best_val_acc = -1.0
    val_history = deque(maxlen=max(1, args.val_smooth_window))
    ema_state = clone_state_dict(model) if args.ema_decay and args.ema_decay > 0.0 else None
    for epoch_idx in range(args.epochs):
        if stopper.early_stop:
            break
        warmup_active = (
            subset_indices is not None
            and args.init_from_full
            and args.subset_warmup_epochs > 0
            and epoch_idx < args.subset_warmup_epochs
        )
        configure_subset_warmup(model, warmup_active, args.subset_warmup_mode)
        model.train()
        for features, labels in iterator.get_batches(train_X, train_y, shuffle=True):
            features = features.to(args.device)
            labels = labels.to(args.device)
            if subset_indices is not None and args.subset_channel_dropout > 0.0:
                features = apply_subset_channel_dropout(
                    features,
                    args.subset_channel_dropout,
                    prioritize_low_rank=args.subset_drop_lowrank_first,
                )
            optimizer.zero_grad()
            logits, _ = model(features)
            loss = criterion(logits, labels)
            if teacher is not None and args.distill_alpha > 0.0 and subset_indices is not None:
                with torch.no_grad():
                    teacher_inputs = expand_subset_features(features, full_train_X.shape[1], subset_indices)
                    teacher_logits, _ = teacher(teacher_inputs)
                kd = distillation_loss(logits, teacher_logits, args.distill_temp)
                loss = (1.0 - args.distill_alpha) * loss + args.distill_alpha * kd
            loss.backward()
            if args.grad_clip and args.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            if ema_state is not None:
                update_ema_state(ema_state, model, args.ema_decay)
        scheduler.step()
        eval_model = model
        restore_state = None
        if ema_state is not None:
            restore_state = clone_state_dict(model)
            model.load_state_dict(ema_state)
        val_acc = evaluate_one_epoch_classifier(iterator, (val_X, val_y), eval_model, args.device, criterion)
        if restore_state is not None:
            model.load_state_dict(restore_state)
        raw_best_val_acc = max(raw_best_val_acc, val_acc)
        val_history.append(val_acc)
        smoothed_val_acc = sum(val_history) / len(val_history)
        if epoch_idx + 1 >= args.min_epochs:
            stopper(smoothed_val_acc)
        if smoothed_val_acc > best_acc:
            best_acc = smoothed_val_acc
            best_epoch = epoch_idx + 1
            best_state = clone_state_dict(model) if ema_state is None else {k: v.clone() for k, v in ema_state.items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        logits, _ = model(final_test_X.to(args.device))
        test_acc = (logits.argmax(dim=1).cpu() == final_test_y).float().mean().item()

    payload = {
        "dataset": dataset_name,
        "subject_id": args.subject_id,
        "backbone": args.backbone,
        "setting": "subset" if subset_indices is not None else "full",
        "val_acc": best_acc,
        "raw_best_val_acc": raw_best_val_acc,
        "best_epoch": best_epoch,
        "test_acc": test_acc,
        "params": count_parameters(model),
        "avg_forward_seconds": benchmark_forward(model, final_test_X[:1], args.device),
        "selected_indices": subset_indices,
        "selected_channels": [EEG_CHANNELS[dataset_name][idx] for idx in subset_indices] if subset_indices is not None else None,
        "channel_weight_mode": args.channel_weight_mode,
        "channel_weights": channel_weights,
        "init_from_full": args.init_from_full,
        "teacher_checkpoint": os.path.abspath(teacher_checkpoint) if teacher is not None else None,
        "distill_alpha": args.distill_alpha,
        "distill_temp": args.distill_temp,
        "min_epochs": args.min_epochs,
        "val_smooth_window": args.val_smooth_window,
        "grad_clip": args.grad_clip,
        "ema_decay": args.ema_decay,
        "subset_channel_dropout": args.subset_channel_dropout,
        "subset_drop_lowrank_first": args.subset_drop_lowrank_first,
        "subset_warmup_epochs": args.subset_warmup_epochs,
        "subset_warmup_mode": args.subset_warmup_mode,
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    output_path = os.path.abspath(args.output)
    output_dir = os.path.dirname(output_path) or os.path.dirname(__file__)
    output_stem = os.path.splitext(os.path.basename(output_path))[0]
    ckpt_name = f"{output_stem}.pth.tar"
    save({"model_classifier": model.state_dict(), "acc": best_acc}, os.path.join(output_dir, ckpt_name))
    print(payload)


if __name__ == "__main__":
    main()
