import argparse
import json
import os

import torch
import torch.nn.functional as F

from models.IoTChannelSelectionFramework import (
    GraphGuidedChannelSelector,
    ModelAgnosticChannelSelectionWrapper,
    build_backbone,
)
from official_profiles import apply_missing_training_defaults, get_official_defaults
from tools.channel_selection import ranking_to_channels, ranking_to_indices
from tools.complexity import benchmark_forward, count_parameters
from tools.datasets import EEG_CHANNELS, load_single_subject
from tools.run_tools import evaluate_one_epoch_classifier
from tools.utils import BalancedBatchSizeIterator, EarlyStopping, load_adj, save, set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", default="bciciv2a", choices=["bciciv2a", "bciciv2b"])
    parser.add_argument("-subject_id", type=int, default=1)
    parser.add_argument("-duration", type=float, default=4.0)
    parser.add_argument("-epochs", type=int, default=None)
    parser.add_argument("-patience", type=int, default=None)
    parser.add_argument("-batch_size", type=int, default=None)
    parser.add_argument("-lr", type=float, default=None)
    parser.add_argument("-w_decay", type=float, default=None)
    parser.add_argument("-dropout", type=float, default=None)
    parser.add_argument("-selector_hidden", type=int, default=64)
    parser.add_argument("-selector_layers", type=int, default=2)
    parser.add_argument("-selector_dropout", type=float, default=0.1)
    parser.add_argument("-topk", type=int, default=5)
    parser.add_argument("-warmup_epochs", type=int, default=100)
    parser.add_argument("-sparse_lambda", type=float, default=1e-3)
    parser.add_argument("-smooth_lambda", type=float, default=1e-3)
    parser.add_argument("-separation_lambda", type=float, default=5e-2)
    parser.add_argument("-start_temp", type=float, default=2.5)
    parser.add_argument("-end_temp", type=float, default=0.5)
    parser.add_argument("-backbone", default="nexusnet", choices=["nexusnet", "mshallowconvnet", "lggnet"])
    parser.add_argument("-device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("-output", default="iot_framework_s1.json")
    parser.add_argument("-seed", type=int, default=42)
    parser.add_argument("-teacher_checkpoint", default=None)
    parser.add_argument("-distill_alpha", type=float, default=0.7)
    parser.add_argument("-distill_temp", type=float, default=2.0)
    parser.add_argument("-feature_distill_alpha", type=float, default=0.1)
    parser.add_argument("-stage2_epochs", type=int, default=15)
    return parser.parse_args()


def graph_smoothness(scores: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
    degree = adj.sum(dim=-1)
    lap = torch.diag_embed(degree) - adj
    scores_vec = scores.unsqueeze(-1)
    return torch.matmul(scores_vec.transpose(1, 2), torch.matmul(lap, scores_vec)).mean()


def budget_penalty(mask: torch.Tensor, target_budget: torch.Tensor) -> torch.Tensor:
    usage = mask.mean(dim=1)
    return (usage - target_budget).abs().mean()


def separation_penalty(scores: torch.Tensor, topk: int) -> torch.Tensor:
    if topk <= 0 or topk >= scores.size(1):
        return torch.tensor(0.0, device=scores.device)
    sorted_scores, _ = scores.sort(dim=1, descending=True)
    kth_score = sorted_scores[:, topk - 1]
    next_score = sorted_scores[:, topk]
    margin = 0.15
    return torch.relu(margin - (kth_score - next_score)).mean()


def resolve_backbone_kwargs(args, dataset_name: str, eu_adj, centrality):
    static_adj, _ = load_adj(dataset_name)
    static_adj = torch.tensor(static_adj, dtype=torch.float32)
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
            }
        )
    elif args.backbone == "mshallowconvnet":
        backbone_kwargs["dropout"] = args.dropout
    elif args.backbone == "lggnet":
        backbone_kwargs["dropout"] = args.dropout
    else:
        backbone_kwargs = {"hidden_dim": 32, "dropout": args.dropout}
    return static_adj, backbone_kwargs


def build_teacher(args, train_X, train_y, eu_adj, dataset_name: str, backbone_kwargs):
    num_classes = len(torch.unique(train_y))
    teacher = build_backbone(
        args.backbone,
        num_classes=num_classes,
        in_chans=train_X.shape[1],
        input_time_length=train_X.shape[2],
        backbone_kwargs=backbone_kwargs,
    )
    checkpoint_path = args.teacher_checkpoint
    if checkpoint_path is None:
        checkpoint_path = os.path.join(
            os.path.dirname(__file__),
            f"iot_baseline_{args.backbone}_s{args.subject_id}.pth.tar",
        )
    if not os.path.exists(checkpoint_path):
        return None, checkpoint_path

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    teacher.load_state_dict(checkpoint["model_classifier"])
    return teacher.to(args.device).eval(), checkpoint_path


def distillation_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float):
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)


def main():
    args = parse_args()
    args = apply_missing_training_defaults(args)
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

    static_adj, centrality = load_adj(dataset_name)
    static_adj, backbone_kwargs = resolve_backbone_kwargs(args, dataset_name, eu_adj, centrality)
    num_classes = len(torch.unique(train_y))

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
    model = ModelAgnosticChannelSelectionWrapper(
        selector=selector,
        backbone=backbone,
        graph_decoder_dim=args.selector_hidden,
        num_classes=num_classes,
    ).to(args.device)
    teacher, teacher_checkpoint = build_teacher(args, train_X, train_y, eu_adj, dataset_name, backbone_kwargs)
    if teacher is not None:
        # Warm-start the student backbone with the full-channel baseline.
        model.backbone.load_state_dict(teacher.state_dict(), strict=False)

    iterator = BalancedBatchSizeIterator(batch_size=args.batch_size, seed=args.seed)
    criterion = torch.nn.CrossEntropyLoss().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.w_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs // 4))
    stopper = EarlyStopping(patience=args.patience, max_epochs=args.epochs)

    best_state = None
    best_acc = -1.0
    for epoch_idx in range(args.epochs):
        if stopper.early_stop:
            break
        model.train()
        use_hard_mask = epoch_idx >= min(args.warmup_epochs, max(1, args.epochs - 1))
        if hasattr(model.selector, "set_temperature"):
            progress = epoch_idx / max(1, args.epochs - 1)
            temperature = args.start_temp + (args.end_temp - args.start_temp) * progress
            model.selector.set_temperature(temperature)
        for features, labels in iterator.get_batches(train_X, train_y, shuffle=True):
            features = features.to(args.device)
            labels = labels.to(args.device)
            optimizer.zero_grad()
            logits, aux = model(features, use_hard_mask=use_hard_mask)
            sparse_penalty = budget_penalty(aux["channel_mask"], aux["target_budget"])
            smooth_penalty = graph_smoothness(aux["channel_scores"], aux["selector_adj"])
            gap_penalty = separation_penalty(aux["channel_scores"], args.topk)
            loss = criterion(logits, labels)
            if teacher is not None:
                with torch.no_grad():
                    teacher_logits, teacher_aux = teacher(features)
                kd = distillation_loss(logits, teacher_logits, args.distill_temp)
                feature_kd = torch.tensor(0.0, device=args.device)
                if isinstance(teacher_aux, torch.Tensor) and isinstance(aux["backbone_aux"], torch.Tensor):
                    feature_kd = F.mse_loss(aux["backbone_aux"], teacher_aux)
                loss = (
                    (1.0 - args.distill_alpha) * loss
                    + args.distill_alpha * kd
                    + args.feature_distill_alpha * feature_kd
                )
            loss = (
                loss
                + args.sparse_lambda * sparse_penalty
                + args.smooth_lambda * smooth_penalty
                + args.separation_lambda * gap_penalty
            )
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

    stage2_best_acc = best_acc
    stage2_best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    if args.stage2_epochs > 0:
        for parameter in model.selector.parameters():
            parameter.requires_grad = False
        finetune_params = [p for p in model.parameters() if p.requires_grad]
        finetune_optimizer = torch.optim.Adam(finetune_params, lr=args.lr * 0.5, weight_decay=args.w_decay)
        finetune_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            finetune_optimizer, T_max=max(1, args.stage2_epochs)
        )
        for _ in range(args.stage2_epochs):
            model.train()
            for features, labels in iterator.get_batches(train_X, train_y, shuffle=True):
                features = features.to(args.device)
                labels = labels.to(args.device)
                finetune_optimizer.zero_grad()
                logits, aux = model(features, use_hard_mask=True)
                loss = criterion(logits, labels)
                if teacher is not None:
                    with torch.no_grad():
                        teacher_logits, teacher_aux = teacher(features)
                    kd = distillation_loss(logits, teacher_logits, args.distill_temp)
                    feature_kd = torch.tensor(0.0, device=args.device)
                    if isinstance(teacher_aux, torch.Tensor) and isinstance(aux["backbone_aux"], torch.Tensor):
                        feature_kd = F.mse_loss(aux["backbone_aux"], teacher_aux)
                    loss = (
                        (1.0 - args.distill_alpha) * loss
                        + args.distill_alpha * kd
                        + args.feature_distill_alpha * feature_kd
                    )
                loss.backward()
                finetune_optimizer.step()
            finetune_scheduler.step()
            val_acc = evaluate_one_epoch_classifier(iterator, (val_X, val_y), model, args.device, criterion)
            if val_acc > stage2_best_acc:
                stage2_best_acc = val_acc
                stage2_best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        model.load_state_dict(stage2_best_state)

    model.eval()
    with torch.no_grad():
        logits, aux = model(final_test_X.to(args.device), use_hard_mask=True)
        test_acc = (logits.argmax(dim=1).cpu() == final_test_y).float().mean().item()
        channel_scores = aux["channel_scores"].mean(dim=0).detach().cpu().tolist()
        channel_mask = aux["channel_mask"].mean(dim=0).detach().cpu().tolist()

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
    sample = final_test_X[:1]
    latency = benchmark_forward(model, sample, args.device)
    payload = {
        "dataset": dataset_name,
        "subject_id": args.subject_id,
        "backbone": args.backbone,
        "topk": args.topk,
        "val_acc": best_acc,
        "test_acc": test_acc,
        "params": count_parameters(model),
        "avg_forward_seconds": latency,
        "teacher_checkpoint": os.path.abspath(teacher_checkpoint) if teacher is not None else None,
        "distillation_enabled": teacher is not None,
        "stage2_epochs": args.stage2_epochs,
        "selected_channels": ranking_to_channels(ranking, args.topk),
        "selected_indices": ranking_to_indices(ranking, args.topk),
        "ranking": ranking,
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    ckpt_name = f"iot_selector_{args.backbone}_s{args.subject_id}.pth.tar"
    save({"model_classifier": model.state_dict(), "acc": best_acc}, os.path.join(os.path.dirname(__file__), ckpt_name))
    print(payload)


if __name__ == "__main__":
    main()
