import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", default="bciciv2a", choices=["bciciv2a", "bciciv2b"])
    parser.add_argument("-subject_id", type=int, default=1)
    parser.add_argument("-topk", type=int, default=12)
    parser.add_argument("-workdir", default=None)
    parser.add_argument("-output", default="stable_standalone_pipeline_s1.json")
    parser.add_argument("-selector_seeds", type=int, nargs="+", default=[42, 7, 13])
    parser.add_argument("-classifier_seeds", type=int, nargs="+", default=[42, 7, 13])
    parser.add_argument("-selector_epochs", type=int, default=120)
    parser.add_argument("-selector_batch_size", type=int, default=32)
    parser.add_argument("-selector_lr", type=float, default=1e-3)
    parser.add_argument("-selector_w_decay", type=float, default=0.0)
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
    parser.add_argument("-baseline_epochs", type=int, default=220)
    parser.add_argument("-baseline_patience", type=int, default=35)
    parser.add_argument("-baseline_batch_size", type=int, default=64)
    parser.add_argument("-baseline_lr", type=float, default=0.0018)
    parser.add_argument("-baseline_w_decay", type=float, default=0.0)
    parser.add_argument("-baseline_dropout", type=float, default=0.22)
    parser.add_argument("-baseline_min_epochs", type=int, default=90)
    parser.add_argument("-baseline_val_smooth_window", type=int, default=8)
    parser.add_argument("-baseline_grad_clip", type=float, default=1.0)
    parser.add_argument("-baseline_ema_decay", type=float, default=0.99)
    parser.add_argument("-channel_weight_mode", default="none", choices=["none", "rank_linear", "rank_gate", "rank_gate_graph"])
    parser.add_argument("-channel_weight_floor", type=float, default=0.6)
    parser.add_argument("-baseline_subset_channel_dropout", type=float, default=0.0)
    parser.add_argument("-baseline_subset_drop_lowrank_first", action="store_true")
    parser.add_argument("-baseline_subset_warmup_epochs", type=int, default=0)
    parser.add_argument("-baseline_subset_warmup_mode", default="cls", choices=["cls", "cls_gate"])
    parser.add_argument("-selector_post_rule", default="none", choices=["none", "shared_region_top12", "shared_region_top12_v2", "nexus_graph_top12"])
    parser.add_argument("-selector_pool_size", type=int, default=16)
    parser.add_argument("-ensemble_top_models", type=int, default=2)
    parser.add_argument("-ensemble_weighting", default="val_acc", choices=["uniform", "val_acc"])
    parser.add_argument("-distill_alpha", type=float, default=0.0)
    parser.add_argument("-distill_temp", type=float, default=2.0)
    parser.add_argument("-device", default="cuda")
    return parser.parse_args()


def run_command(command, cwd):
    result = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
    )
    if result.stdout:
        print(result.stdout.strip())
    if result.stderr:
        print(result.stderr.strip(), file=sys.stderr)


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def summarize_metric(values):
    if not values:
        return None
    if len(values) == 1:
        return {
            "mean": values[0],
            "std": 0.0,
            "min": values[0],
            "max": values[0],
        }
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values),
        "min": min(values),
        "max": max(values),
    }


def build_consensus_selector(selector_payloads, topk):
    score_table = {}
    channel_table = {}
    rank_table = {}
    for payload in selector_payloads:
        for rank, item in enumerate(payload["ranking"]):
            idx = int(item["index"])
            score_table.setdefault(idx, []).append(float(item["score"]))
            channel_table[idx] = item["channel"]
            rank_table.setdefault(idx, []).append(rank)
    ranking = []
    for idx, scores in score_table.items():
        ranks = rank_table.get(idx, [])
        avg_score = statistics.mean(scores)
        avg_rank = statistics.mean(ranks) if ranks else 0.0
        rank_score = 1.0 - (avg_rank / max(1, len(score_table) - 1))
        stability_penalty = statistics.stdev(scores) if len(scores) > 1 else 0.0
        ranking.append(
            {
                "index": idx,
                "channel": channel_table[idx],
                "score": 0.55 * avg_score + 0.40 * rank_score - 0.15 * stability_penalty,
                "mean_score": avg_score,
                "mean_rank": avg_rank,
                "score_std": stability_penalty,
            }
        )
    ranking.sort(key=lambda item: item["score"], reverse=True)
    return {
        "topk": topk,
        "selected_indices": [item["index"] for item in ranking[:topk]],
        "selected_channels": [item["channel"] for item in ranking[:topk]],
        "ranking": ranking,
    }


def _channel_region(channel_name: str) -> str:
    if channel_name.startswith("FC"):
        return "FC"
    if channel_name.startswith("CP"):
        return "CP"
    if channel_name.startswith("C"):
        return "C"
    return "OTHER"


def _channel_side(channel_name: str) -> str:
    suffix_digits = "".join(ch for ch in channel_name if ch.isdigit())
    if not suffix_digits:
        return "mid"
    digit = int(suffix_digits[-1])
    if digit == 0:
        return "mid"
    return "left" if digit % 2 == 1 else "right"


def _region_target(region_name: str) -> int:
    if region_name == "FC":
        return 3
    if region_name == "C":
        return 5
    if region_name == "CP":
        return 4
    return 0


def _subset_structural_score(selected_items):
    if not selected_items:
        return -1e6
    region_counts = {"FC": 0, "C": 0, "CP": 0, "OTHER": 0}
    side_counts = {"left": 0, "right": 0, "mid": 0}
    total_base = 0.0
    for item in selected_items:
        name = item["channel"]
        region = _channel_region(name)
        side = _channel_side(name)
        region_counts[region] = region_counts.get(region, 0) + 1
        side_counts[side] += 1
        total_base += float(item["score"])

    score = total_base
    score += 0.06 * min(region_counts["FC"], 3)
    score += 0.08 * min(region_counts["C"], 5)
    score += 0.03 * min(region_counts["CP"], 3)
    score -= 0.05 * max(0, region_counts["CP"] - 4)
    score -= 0.08 * max(0, 3 - region_counts["FC"])
    score -= 0.06 * max(0, 4 - region_counts["C"])
    score += 0.04 * min(side_counts["mid"], 3)
    score -= 0.015 * abs(side_counts["left"] - side_counts["right"])

    names = {item["channel"] for item in selected_items}
    mirror_pairs = [("FC1", "FC2"), ("C1", "C2"), ("CP1", "CP2"), ("C3", "C4"), ("CP3", "CP4"), ("FC3", "FC4")]
    for left_name, right_name in mirror_pairs:
        if left_name in names and right_name in names:
            score += 0.035
        elif left_name in names or right_name in names:
            score -= 0.01
    if "FCz" in names:
        score += 0.04
    if "Cz" in names:
        score += 0.05
    if "CPz" in names:
        score += 0.01
    posterior_count = sum(1 for item in selected_items if _channel_region(item["channel"]) == "OTHER")
    score -= 0.08 * posterior_count
    return score


def _select_nexus_graph_subset(pool, topk):
    selected = []
    remaining = list(pool)
    while remaining and len(selected) < topk:
        best_item = None
        best_score = None
        for item in remaining:
            candidate = selected + [item]
            candidate_score = _subset_structural_score(candidate)
            if best_score is None or candidate_score > best_score:
                best_score = candidate_score
                best_item = item
        selected.append(best_item)
        remaining = [item for item in remaining if item["index"] != best_item["index"]]
    return selected


def apply_selector_post_rule(consensus, topk, rule, pool_size):
    if rule == "none":
        return consensus, None

    ranking = consensus["ranking"]
    pool_size = max(topk, min(len(ranking), pool_size))
    pool = ranking[:pool_size]

    if rule == "nexus_graph_top12":
        selected_items = _select_nexus_graph_subset(pool, topk)
        selected_indices = [item["index"] for item in selected_items]
        selected_lookup = {idx: rank for rank, idx in enumerate(selected_indices)}
        reranked = list(selected_items)
        reranked.extend(item for item in ranking if item["index"] not in selected_lookup)
        updated = {
            "topk": topk,
            "selected_indices": selected_indices,
            "selected_channels": [item["channel"] for item in selected_items],
            "ranking": reranked,
        }
        metadata = {
            "rule": rule,
            "pool_size": pool_size,
            "pool_indices": [item["index"] for item in pool],
            "structural_score": _subset_structural_score(selected_items),
        }
        return updated, metadata

    if rule not in {"shared_region_top12", "shared_region_top12_v2"}:
        raise ValueError(f"Unsupported selector_post_rule: {rule}")

    if rule == "shared_region_top12_v2":
        quotas = {"C": 4, "FC": 4}
        cp_max = 3
        preferred_midline = {"FCz", "Cz"}
    else:
        quotas = {"C": 5, "FC": 3}
        cp_max = 4
        preferred_midline = set()
    selected_indices = []
    selected_set = set()

    for item in pool:
        if item["channel"] not in preferred_midline:
            continue
        if item["index"] in selected_set or len(selected_indices) >= topk:
            continue
        selected_indices.append(item["index"])
        selected_set.add(item["index"])

    def add_from_region(region_name, limit):
        for item in pool:
            if len(selected_indices) >= topk or limit <= 0:
                break
            if item["index"] in selected_set:
                continue
            if _channel_region(item["channel"]) != region_name:
                continue
            selected_indices.append(item["index"])
            selected_set.add(item["index"])
            limit -= 1

    fc_already = sum(1 for item in pool if item["index"] in selected_set and _channel_region(item["channel"]) == "FC")
    c_already = sum(1 for item in pool if item["index"] in selected_set and _channel_region(item["channel"]) == "C")
    add_from_region("C", max(0, quotas["C"] - c_already))
    add_from_region("FC", max(0, quotas["FC"] - fc_already))

    cp_count = 0
    if rule == "shared_region_top12_v2":
        cp_count = sum(1 for item in pool if item["index"] in selected_set and _channel_region(item["channel"]) == "CP")
    for item in pool:
        if len(selected_indices) >= topk:
            break
        idx = item["index"]
        if idx in selected_set:
            continue
        region = _channel_region(item["channel"])
        if region == "CP" and cp_count >= cp_max:
            continue
        selected_indices.append(idx)
        selected_set.add(idx)
        if region == "CP":
            cp_count += 1

    if len(selected_indices) < topk:
        for item in ranking:
            idx = item["index"]
            if idx in selected_set:
                continue
            selected_indices.append(idx)
            selected_set.add(idx)
            if len(selected_indices) >= topk:
                break

    reranked = []
    selected_lookup = {idx: rank for rank, idx in enumerate(selected_indices)}
    selected_items = [item for item in pool if item["index"] in selected_lookup]
    selected_items.sort(key=lambda item: selected_lookup[item["index"]])
    reranked.extend(selected_items)
    reranked.extend(item for item in ranking if item["index"] not in selected_lookup)

    updated = {
        "topk": topk,
        "selected_indices": selected_indices,
        "selected_channels": [item["channel"] for item in reranked[:topk]],
        "ranking": reranked,
    }
    metadata = {
        "rule": rule,
        "pool_size": pool_size,
        "quotas": {"C": quotas["C"], "FC": quotas["FC"], "CP_max": cp_max},
        "preferred_midline": sorted(preferred_midline),
        "pool_indices": [item["index"] for item in pool],
    }
    return updated, metadata


def main():
    args = parse_args()
    repo_dir = Path(__file__).resolve().parent
    workdir = Path(args.workdir) if args.workdir else repo_dir
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = workdir / output_path
    run_dir = output_path.with_suffix("")
    run_dir.mkdir(parents=True, exist_ok=True)

    selector_outputs = []
    for seed in args.selector_seeds:
        selector_output = run_dir / f"selector_seed{seed}.json"
        command = [
            sys.executable,
            str(repo_dir / "train_standalone_selector.py"),
            "-dataset",
            args.dataset,
            "-subject_id",
            str(args.subject_id),
            "-topk",
            str(args.topk),
            "-epochs",
            str(args.selector_epochs),
            "-batch_size",
            str(args.selector_batch_size),
            "-lr",
            str(args.selector_lr),
            "-w_decay",
            str(args.selector_w_decay),
            "-selector_hidden",
            str(args.selector_hidden),
            "-selector_layers",
            str(args.selector_layers),
            "-selector_dropout",
            str(args.selector_dropout),
            "-smooth_lambda",
            str(args.smooth_lambda),
            "-budget_lambda",
            str(args.budget_lambda),
            "-pairwise_lambda",
            str(args.pairwise_lambda),
            "-bce_lambda",
            str(args.bce_lambda),
            "-margin",
            str(args.margin),
            "-start_temp",
            str(args.start_temp),
            "-end_temp",
            str(args.end_temp),
            "-device",
            args.device,
            "-seed",
            str(seed),
            "-output",
            str(selector_output),
        ]
        run_command(command, str(workdir))
        selector_outputs.append(selector_output)

    selector_payloads = [load_json(path) for path in selector_outputs]
    consensus = build_consensus_selector(selector_payloads, args.topk)
    consensus, selector_post_rule = apply_selector_post_rule(
        consensus,
        args.topk,
        args.selector_post_rule,
        args.selector_pool_size,
    )
    consensus_payload = {
        "dataset": selector_payloads[0]["dataset"],
        "subject_id": args.subject_id,
        "selector_type": "standalone_gnn_consensus",
        "ranking_mode": selector_payloads[0].get("ranking_mode", "task_driven_gnn"),
        "topk": args.topk,
        "target_indices": None,
        "selector_seeds": args.selector_seeds,
        "selector_post_rule": selector_post_rule,
        "selected_indices": consensus["selected_indices"],
        "selected_channels": consensus["selected_channels"],
        "ranking": consensus["ranking"],
    }
    consensus_path = run_dir / "selector_consensus.json"
    with open(consensus_path, "w", encoding="utf-8") as handle:
        json.dump(consensus_payload, handle, ensure_ascii=False, indent=2)

    classifier_outputs = []
    for seed in args.classifier_seeds:
        baseline_output = run_dir / f"classifier_seed{seed}.json"
        command = [
            sys.executable,
            str(repo_dir / "train_iot_baseline.py"),
            "-dataset",
            args.dataset,
            "-subject_id",
            str(args.subject_id),
            "-backbone",
            "nexusnet",
            "-channel_subset_json",
            str(consensus_path),
            "-subset_topk",
            str(args.topk),
            "-epochs",
            str(args.baseline_epochs),
            "-patience",
            str(args.baseline_patience),
            "-batch_size",
            str(args.baseline_batch_size),
            "-lr",
            str(args.baseline_lr),
            "-w_decay",
            str(args.baseline_w_decay),
            "-dropout",
            str(args.baseline_dropout),
            "-min_epochs",
            str(args.baseline_min_epochs),
            "-val_smooth_window",
            str(args.baseline_val_smooth_window),
            "-grad_clip",
            str(args.baseline_grad_clip),
            "-ema_decay",
            str(args.baseline_ema_decay),
            "-subset_channel_dropout",
            str(args.baseline_subset_channel_dropout),
            "-subset_warmup_epochs",
            str(args.baseline_subset_warmup_epochs),
            "-subset_warmup_mode",
            args.baseline_subset_warmup_mode,
            "-init_from_full",
            "-distill_alpha",
            str(args.distill_alpha),
            "-distill_temp",
            str(args.distill_temp),
            "-seed",
            str(seed),
            "-channel_weight_mode",
            args.channel_weight_mode,
            "-channel_weight_floor",
            str(args.channel_weight_floor),
            "-output",
            str(baseline_output),
        ]
        if args.baseline_subset_drop_lowrank_first:
            command.append("-subset_drop_lowrank_first")
        run_command(command, str(workdir))
        classifier_outputs.append(baseline_output)

    classifier_payloads = [load_json(path) for path in classifier_outputs]
    test_scores = [float(payload["test_acc"]) for payload in classifier_payloads]
    val_scores = [float(payload["val_acc"]) for payload in classifier_payloads]

    ensemble_output = run_dir / "classifier_ensemble.json"
    checkpoint_paths = [str((path.with_suffix("")).with_suffix(".pth.tar").resolve()) for path in classifier_outputs]
    metrics_jsons = [str(path.resolve()) for path in classifier_outputs]
    ensemble_command = [
        sys.executable,
        str(repo_dir / "evaluate_baseline_ensemble.py"),
        "-dataset",
        args.dataset,
        "-subject_id",
        str(args.subject_id),
        "-backbone",
        "nexusnet",
        "-channel_subset_json",
        str(consensus_path),
        "-subset_topk",
        str(args.topk),
        "-dropout",
        str(args.baseline_dropout),
        "-channel_weight_mode",
        args.channel_weight_mode,
        "-channel_weight_floor",
        str(args.channel_weight_floor),
        "-checkpoint_paths",
    ] + checkpoint_paths + [
        "-checkpoint_metrics_jsons",
    ] + metrics_jsons + [
        "-top_models",
        str(args.ensemble_top_models),
        "-weighting",
        args.ensemble_weighting,
        "-output",
        str(ensemble_output),
    ]
    run_command(ensemble_command, str(workdir))
    ensemble_payload = load_json(ensemble_output)

    payload = {
        "dataset": consensus_payload["dataset"],
        "subject_id": args.subject_id,
        "topk": args.topk,
        "selector_seeds": args.selector_seeds,
        "classifier_seeds": args.classifier_seeds,
        "consensus_selector_json": str(consensus_path.resolve()),
        "selected_indices": consensus_payload["selected_indices"],
        "selected_channels": consensus_payload["selected_channels"],
        "selector_runs": [str(path.resolve()) for path in selector_outputs],
        "classifier_runs": [str(path.resolve()) for path in classifier_outputs],
        "baseline_config": {
            "epochs": args.baseline_epochs,
            "patience": args.baseline_patience,
            "batch_size": args.baseline_batch_size,
            "lr": args.baseline_lr,
            "w_decay": args.baseline_w_decay,
            "dropout": args.baseline_dropout,
            "min_epochs": args.baseline_min_epochs,
            "val_smooth_window": args.baseline_val_smooth_window,
            "grad_clip": args.baseline_grad_clip,
            "ema_decay": args.baseline_ema_decay,
            "channel_weight_mode": args.channel_weight_mode,
            "channel_weight_floor": args.channel_weight_floor,
            "subset_channel_dropout": args.baseline_subset_channel_dropout,
            "subset_drop_lowrank_first": args.baseline_subset_drop_lowrank_first,
            "subset_warmup_epochs": args.baseline_subset_warmup_epochs,
            "subset_warmup_mode": args.baseline_subset_warmup_mode,
            "selector_post_rule": args.selector_post_rule,
            "selector_pool_size": args.selector_pool_size,
        },
        "ensemble_config": {
            "top_models": args.ensemble_top_models,
            "weighting": args.ensemble_weighting,
        },
        "val_acc_summary": summarize_metric(val_scores),
        "test_acc_summary": summarize_metric(test_scores),
        "best_test_acc": max(test_scores),
        "worst_test_acc": min(test_scores),
        "ensemble_result": ensemble_payload,
    }
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(payload)


if __name__ == "__main__":
    main()
