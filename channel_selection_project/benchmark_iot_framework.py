import argparse
import json
import subprocess
import sys

from experiment_profiles import PAPER_BACKBONES, PAPER_TOPK, get_topk_profile


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", default="bciciv2a", choices=["bciciv2a", "bciciv2b"])
    parser.add_argument("-subject_id", type=int, default=1)
    parser.add_argument("-duration", type=float, default=4.0)
    parser.add_argument("-epochs", type=int, default=1000)
    parser.add_argument("-batch_size", type=int, default=32)
    parser.add_argument("-topk", type=int, nargs="+", default=PAPER_TOPK)
    parser.add_argument("-backbones", nargs="+", default=PAPER_BACKBONES)
    parser.add_argument("-device", default=None)
    parser.add_argument("-output", default="iot_benchmark_s1.json")
    return parser.parse_args()


def run_command(command):
    subprocess.run(command, check=True)


def main():
    args = parse_args()
    results = []
    python_exec = sys.executable

    for backbone in args.backbones:
        for topk in args.topk:
            profile = get_topk_profile(backbone, topk)
            result_path = f"benchmark_{backbone}_top{topk}_s{args.subject_id}.json"
            train_cmd = [
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
                result_path,
            ]
            for key, value in profile.items():
                train_cmd.extend([f"-{key}", str(value)])
            if args.device:
                train_cmd.extend(["-device", args.device])
            run_command(train_cmd)

            with open(result_path, "r", encoding="utf-8") as handle:
                results.append(json.load(handle))

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "dataset": args.dataset,
                "subject_id": args.subject_id,
                "epochs": args.epochs,
                "results": results,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    print({"output": args.output, "num_runs": len(results)})


if __name__ == "__main__":
    main()
