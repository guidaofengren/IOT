import argparse
import json
import subprocess
import sys

from experiment_profiles import PAPER_BACKBONES, PAPER_TOPK, get_topk_profile


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", default="bciciv2a", choices=["bciciv2a", "bciciv2b"])
    parser.add_argument("-subject_id", type=int, default=1)
    parser.add_argument("-epochs", type=int, default=1000)
    parser.add_argument("-patience", type=int, default=40)
    parser.add_argument("-topk", type=int, nargs="+", default=PAPER_TOPK)
    parser.add_argument("-backbones", nargs="+", default=PAPER_BACKBONES)
    parser.add_argument("-batch_size", type=int, default=32)
    parser.add_argument("-device", default=None)
    parser.add_argument("-output", default="iot_compare_s1_1000ep.json")
    return parser.parse_args()


def run_command(command):
    subprocess.run(command, check=True)


def main():
    args = parse_args()
    python_exec = sys.executable
    results = []

    for backbone in args.backbones:
        baseline_path = f"compare_{backbone}_full_s{args.subject_id}.json"
        baseline_cmd = [
            python_exec,
            "train_iot_baseline.py",
            "-dataset",
            args.dataset,
            "-subject_id",
            str(args.subject_id),
            "-backbone",
            backbone,
            "-epochs",
            str(args.epochs),
            "-patience",
            str(args.patience),
            "-batch_size",
            str(args.batch_size),
            "-output",
            baseline_path,
        ]
        if args.device:
            baseline_cmd.extend(["-device", args.device])
        run_command(baseline_cmd)
        with open(baseline_path, "r", encoding="utf-8") as handle:
            results.append(json.load(handle))

        for topk in args.topk:
            profile = get_topk_profile(backbone, topk)
            selected_path = f"compare_{backbone}_top{topk}_s{args.subject_id}.json"
            selected_cmd = [
                python_exec,
                "train_iot_framework.py",
                "-dataset",
                args.dataset,
                "-subject_id",
                str(args.subject_id),
                "-backbone",
                backbone,
                "-epochs",
                str(args.epochs),
                "-batch_size",
                str(args.batch_size),
                "-topk",
                str(topk),
                "-output",
                selected_path,
            ]
            for key, value in profile.items():
                selected_cmd.extend([f"-{key}", str(value)])
            if args.device:
                selected_cmd.extend(["-device", args.device])
            run_command(selected_cmd)
            with open(selected_path, "r", encoding="utf-8") as handle:
                results.append(json.load(handle))

    summary = {
        "dataset": args.dataset,
        "subject_id": args.subject_id,
        "epochs": args.epochs,
        "results": results,
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print({"output": args.output, "num_results": len(results)})


if __name__ == "__main__":
    main()
