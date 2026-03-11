import argparse
import copy
import ctypes
import json
import os
import subprocess
import sys
import time
from itertools import product
from shutil import which

from experiment_profiles import PAPER_BACKBONES, get_topk_profile
from official_profiles import get_official_defaults


DEFAULT_TOPK_ORDER = [5, 3, 8]
TUNABLE_KEYS = [
    "patience",
    "batch_size",
    "lr",
    "w_decay",
    "dropout",
    "warmup_epochs",
    "stage2_epochs",
    "distill_alpha",
    "feature_distill_alpha",
    "sparse_lambda",
    "smooth_lambda",
    "separation_lambda",
    "start_temp",
    "end_temp",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", default="bciciv2a", choices=["bciciv2a", "bciciv2b"])
    parser.add_argument("-subject_id", type=int, default=1)
    parser.add_argument("-duration", type=float, default=4.0)
    parser.add_argument("-backbones", nargs="+", default=PAPER_BACKBONES)
    parser.add_argument("-topk_order", type=int, nargs="+", default=DEFAULT_TOPK_ORDER)
    parser.add_argument("-device", default=None)
    parser.add_argument("-target_ratio", type=float, default=0.95)
    parser.add_argument("-preferred_ratio", type=float, default=1.0)
    parser.add_argument("-max_trials_per_stage", type=int, default=12)
    parser.add_argument("-parallel_jobs", type=int, default=0)
    parser.add_argument("-reserve_ram_gb", type=float, default=8.0)
    parser.add_argument("-reserve_vram_mb", type=int, default=4096)
    parser.add_argument("-per_process_ram_gb", type=float, default=6.0)
    parser.add_argument("-per_process_vram_mb", type=int, default=4096)
    parser.add_argument("-poll_seconds", type=float, default=20.0)
    parser.add_argument("-cooldown_seconds", type=float, default=8.0)
    parser.add_argument("-resume", action="store_true")
    parser.add_argument("-output", default="paper_topk_tuning_summary_s1.json")
    return parser.parse_args()


def infer_dataset_name(dataset: str) -> str:
    return "BNCI2014001" if dataset == "bciciv2a" else "BNCI2014004"


def build_full_result_path(backbone: str, subject_id: int) -> str:
    return f"compare_{backbone}_full_s{subject_id}.json"


def build_tune_result_path(backbone: str, topk: int, subject_id: int, trial_idx: int) -> str:
    return f"tune_{backbone}_top{topk}_trial{trial_idx:03d}_s{subject_id}.json"


def run_command(command):
    subprocess.run(command, check=True)


def ensure_full_baseline(args, backbone: str):
    output_path = build_full_result_path(backbone, args.subject_id)
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    defaults = get_official_defaults(backbone)

    command = [
        sys.executable,
        "train_iot_baseline.py",
        "-dataset",
        args.dataset,
        "-subject_id",
        str(args.subject_id),
        "-duration",
        str(args.duration),
        "-backbone",
        backbone,
        "-output",
        output_path,
        "-epochs",
        str(defaults["epochs"]),
    ]
    if args.device:
        command.extend(["-device", args.device])
    run_command(command)
    with open(output_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def numeric_grid(base_value, candidates, cast_fn):
    values = []
    seen = set()
    for value in candidates:
        casted = cast_fn(value)
        if casted not in seen:
            seen.add(casted)
            values.append(casted)
    if base_value is not None:
        casted = cast_fn(base_value)
        if casted not in seen:
            values.insert(0, casted)
    return values


def get_search_space(base_profile: dict) -> dict:
    lr = float(base_profile.get("lr", 1e-3))
    w_decay = float(base_profile.get("w_decay", 1e-4))
    dropout = float(base_profile.get("dropout", 0.25))
    patience = int(base_profile.get("patience", 40))
    warmup = int(base_profile.get("warmup_epochs", 100))
    stage2 = int(base_profile.get("stage2_epochs", 30))
    distill = float(base_profile.get("distill_alpha", 0.5))
    feat_distill = float(base_profile.get("feature_distill_alpha", 0.0))
    sparse = float(base_profile.get("sparse_lambda", 1e-3))
    smooth = float(base_profile.get("smooth_lambda", 1e-3))
    separation = float(base_profile.get("separation_lambda", 0.03))
    start_temp = float(base_profile.get("start_temp", 2.5))
    end_temp = float(base_profile.get("end_temp", 0.5))
    batch_size = int(base_profile.get("batch_size", 32))

    return {
        "patience": numeric_grid(patience, [30, 40, 60], int),
        "batch_size": numeric_grid(batch_size, [16, 24, 32, 48], int),
        "lr": numeric_grid(lr, [lr * 0.5, lr, lr * 1.5], float),
        "w_decay": numeric_grid(w_decay, [w_decay * 0.5, w_decay, w_decay * 2.0], float),
        "dropout": numeric_grid(dropout, [max(0.1, dropout - 0.1), dropout, min(0.5, dropout + 0.1)], float),
        "warmup_epochs": numeric_grid(warmup, [max(20, warmup - 20), warmup, warmup + 20], int),
        "stage2_epochs": numeric_grid(stage2, [max(10, stage2 - 10), stage2, stage2 + 10], int),
        "distill_alpha": numeric_grid(
            distill,
            [max(0.0, distill - 0.15), distill, min(0.9, distill + 0.15)],
            float,
        ),
        "feature_distill_alpha": numeric_grid(
            feat_distill,
            [max(0.0, feat_distill - 0.05), feat_distill, min(0.2, feat_distill + 0.05)],
            float,
        ),
        "sparse_lambda": numeric_grid(sparse, [sparse * 0.5, sparse, sparse * 2.0], float),
        "smooth_lambda": numeric_grid(smooth, [smooth * 0.5, smooth, smooth * 2.0], float),
        "separation_lambda": numeric_grid(
            separation,
            [max(0.0, separation * 0.5), separation, separation * 1.5],
            float,
        ),
        "start_temp": numeric_grid(start_temp, [max(1.0, start_temp - 0.5), start_temp, start_temp + 0.5], float),
        "end_temp": numeric_grid(end_temp, [max(0.2, end_temp - 0.1), end_temp, min(1.0, end_temp + 0.1)], float),
    }


def generate_trials(base_profile: dict, max_trials: int):
    search_space = get_search_space(base_profile)
    stages = [
        ["lr", "w_decay", "dropout", "batch_size"],
        ["warmup_epochs", "stage2_epochs", "patience"],
        ["distill_alpha", "feature_distill_alpha"],
        ["sparse_lambda", "smooth_lambda", "separation_lambda"],
        ["start_temp", "end_temp"],
    ]
    trials = [copy.deepcopy(base_profile)]
    seen = {json.dumps(base_profile, sort_keys=True)}

    for stage_keys in stages:
        stage_values = [search_space[key] for key in stage_keys]
        for values in product(*stage_values):
            candidate = copy.deepcopy(base_profile)
            for key, value in zip(stage_keys, values):
                candidate[key] = value
            fingerprint = json.dumps(candidate, sort_keys=True)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            trials.append(candidate)
            if len(trials) >= max_trials:
                return trials
    return trials[:max_trials]


def merge_seed_profile(base_profile: dict, seed_profile: dict | None):
    if not seed_profile:
        return dict(base_profile)
    merged = dict(base_profile)
    for key in TUNABLE_KEYS:
        if key in seed_profile:
            merged[key] = seed_profile[key]
    return merged


def build_train_command(args, backbone: str, topk: int, profile: dict, output_path: str):
    defaults = get_official_defaults(backbone)
    command = [
        sys.executable,
        "train_iot_framework.py",
        "-dataset",
        args.dataset,
        "-subject_id",
        str(args.subject_id),
        "-duration",
        str(args.duration),
        "-backbone",
        backbone,
        "-topk",
        str(topk),
        "-output",
        output_path,
        "-epochs",
        str(defaults["epochs"]),
    ]
    for key in TUNABLE_KEYS:
        if key in profile:
            command.extend([f"-{key}", str(profile[key])])
    if args.device:
        command.extend(["-device", args.device])
    return command


def load_trial_result(output_path: str, profile: dict, full_result: dict, args):
    with open(output_path, "r", encoding="utf-8") as handle:
        result = json.load(handle)

    full_test = float(full_result["test_acc"])
    selected_test = float(result["test_acc"])
    pass_ratio = selected_test / full_test if full_test > 0 else 0.0
    result["target_ratio"] = args.target_ratio
    result["preferred_ratio"] = args.preferred_ratio
    result["pass_ratio_vs_full"] = pass_ratio
    result["meets_target"] = pass_ratio >= args.target_ratio
    result["beats_full"] = pass_ratio >= args.preferred_ratio
    result["trial_profile"] = profile

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    return result


def sort_results(results: list):
    return sorted(
        results,
        key=lambda item: (
            bool(item.get("beats_full")),
            bool(item.get("meets_target")),
            float(item.get("pass_ratio_vs_full", 0.0)),
            float(item.get("val_acc", 0.0)),
            float(item.get("test_acc", 0.0)),
        ),
        reverse=True,
    )


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def get_free_ram_bytes():
    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
        return None
    return int(stat.ullAvailPhys)


def get_free_vram_mb():
    if which("nvidia-smi") is None:
        return None
    command = ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        return None
    values = []
    for line in completed.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            values.append(int(float(line)))
        except ValueError:
            continue
    if not values:
        return None
    return min(values)


def compute_parallel_capacity(args):
    if args.parallel_jobs and args.parallel_jobs > 0:
        return max(1, args.parallel_jobs)

    free_ram_bytes = get_free_ram_bytes()
    free_vram_mb = get_free_vram_mb()

    ram_capacity = 1
    if free_ram_bytes is not None:
        free_ram_gb = free_ram_bytes / (1024 ** 3)
        usable_ram_gb = max(0.0, free_ram_gb - args.reserve_ram_gb)
        ram_capacity = max(1, int(usable_ram_gb // max(args.per_process_ram_gb, 0.5)))

    vram_capacity = 1
    if free_vram_mb is not None:
        usable_vram_mb = max(0, free_vram_mb - args.reserve_vram_mb)
        vram_capacity = max(1, int(usable_vram_mb // max(args.per_process_vram_mb, 512)))

    return max(1, min(ram_capacity, vram_capacity))


def has_capacity_for_new_job(args, active_count: int, target_parallel: int):
    if active_count >= target_parallel:
        return False

    free_ram_bytes = get_free_ram_bytes()
    if free_ram_bytes is not None:
        free_ram_gb = free_ram_bytes / (1024 ** 3)
        if free_ram_gb < args.reserve_ram_gb + args.per_process_ram_gb:
            return False

    free_vram_mb = get_free_vram_mb()
    if free_vram_mb is not None:
        if free_vram_mb < args.reserve_vram_mb + args.per_process_vram_mb:
            return False

    return True


def launch_trial_process(args, job: dict):
    process = subprocess.Popen(job["command"])
    job["process"] = process
    job["pid"] = process.pid
    job["started_at"] = time.time()
    return job


def finalize_completed_jobs(active_jobs: list, completed_results: list, args):
    still_running = []
    for job in active_jobs:
        returncode = job["process"].poll()
        if returncode is None:
            still_running.append(job)
            continue
        if returncode != 0:
            raise subprocess.CalledProcessError(returncode, job["command"])
        result = load_trial_result(job["output_path"], job["profile"], job["full_result"], args)
        completed_results.append(result)
    return still_running


def build_jobs_for_group(args, backbone: str, topk: int, full_result: dict, seed_profile: dict | None = None):
    base_profile = get_topk_profile(backbone, topk)
    merged_base = merge_seed_profile(base_profile, seed_profile)
    trials = generate_trials(merged_base, args.max_trials_per_stage)
    if merged_base != base_profile:
        fallback_trials = generate_trials(base_profile, max(2, min(4, args.max_trials_per_stage)))
        seen = {json.dumps(item, sort_keys=True) for item in trials}
        for candidate in fallback_trials:
            fingerprint = json.dumps(candidate, sort_keys=True)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            trials.append(candidate)
            if len(trials) >= args.max_trials_per_stage:
                break
    jobs = []
    for trial_idx, profile in enumerate(trials, start=1):
        output_path = build_tune_result_path(backbone, topk, args.subject_id, trial_idx)
        if args.resume and os.path.exists(output_path):
            jobs.append(
                {
                    "resume_only": True,
                    "output_path": output_path,
                    "profile": profile,
                    "full_result": full_result,
                }
            )
            continue
        jobs.append(
            {
                "resume_only": False,
                "output_path": output_path,
                "profile": profile,
                "full_result": full_result,
                "command": build_train_command(args, backbone, topk, profile, output_path),
            }
        )
    return jobs


def run_group_jobs(args, backbone: str, topk: int, full_result: dict, seed_profile: dict | None = None):
    jobs = build_jobs_for_group(args, backbone, topk, full_result, seed_profile=seed_profile)
    completed_results = []
    active_jobs = []
    pending_jobs = []

    for job in jobs:
        if job["resume_only"]:
            completed_results.append(load_trial_result(job["output_path"], job["profile"], job["full_result"], args))
        else:
            pending_jobs.append(job)

    target_parallel = compute_parallel_capacity(args)
    print(
        {
            "backbone": backbone,
            "topk": topk,
            "target_parallel": target_parallel,
            "pending_trials": len(pending_jobs),
            "reserve_ram_gb": args.reserve_ram_gb,
            "reserve_vram_mb": args.reserve_vram_mb,
        }
    )

    while pending_jobs or active_jobs:
        active_jobs = finalize_completed_jobs(active_jobs, completed_results, args)
        launched = False
        while pending_jobs and has_capacity_for_new_job(args, len(active_jobs), target_parallel):
            job = pending_jobs.pop(0)
            active_jobs.append(launch_trial_process(args, job))
            launched = True
            if args.cooldown_seconds > 0:
                time.sleep(args.cooldown_seconds)
        if pending_jobs and not launched:
            time.sleep(max(3.0, args.poll_seconds))
        elif active_jobs:
            time.sleep(3.0)

    return sort_results(completed_results)


def main():
    args = parse_args()
    dataset_name = infer_dataset_name(args.dataset)
    summary = {
        "dataset": dataset_name,
        "subject_id": args.subject_id,
        "epochs_policy": "official_per_backbone",
        "target_ratio": args.target_ratio,
        "preferred_ratio": args.preferred_ratio,
        "parallel_jobs": args.parallel_jobs,
        "reserve_ram_gb": args.reserve_ram_gb,
        "reserve_vram_mb": args.reserve_vram_mb,
        "per_process_ram_gb": args.per_process_ram_gb,
        "per_process_vram_mb": args.per_process_vram_mb,
        "order": args.topk_order,
        "results": [],
    }

    full_results = {}
    best_profiles = {}
    for backbone in args.backbones:
        full_results[backbone] = ensure_full_baseline(args, backbone)

    for topk in args.topk_order:
        for backbone in args.backbones:
            seed_profile = None if topk == 5 else best_profiles.get(backbone)
            ranked = run_group_jobs(args, backbone, topk, full_results[backbone], seed_profile=seed_profile)
            if ranked and ranked[0].get("trial_profile"):
                best_profiles[backbone] = ranked[0]["trial_profile"]
            summary["results"].append(
                {
                    "backbone": backbone,
                    "topk": topk,
                    "full_test_acc": full_results[backbone]["test_acc"],
                    "seed_profile_used": seed_profile,
                    "best": ranked[0] if ranked else None,
                    "trials": ranked,
                }
            )
            with open(args.output, "w", encoding="utf-8") as handle:
                json.dump(summary, handle, ensure_ascii=False, indent=2)

    print({"output": args.output, "num_groups": len(summary["results"])})


if __name__ == "__main__":
    main()
