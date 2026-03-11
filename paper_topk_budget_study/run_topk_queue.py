import argparse
import subprocess
import sys
import time

from experiment_profiles import PAPER_BACKBONES, get_topk_profile


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", default="bciciv2a", choices=["bciciv2a", "bciciv2b"])
    parser.add_argument("-subject_id", type=int, default=1)
    parser.add_argument("-duration", type=float, default=4.0)
    parser.add_argument("-epochs", type=int, default=1000)
    parser.add_argument("-batch_size", type=int, default=32)
    parser.add_argument("-topk", type=int, default=5)
    parser.add_argument("-backbones", nargs="+", default=PAPER_BACKBONES)
    parser.add_argument("-device", default=None)
    parser.add_argument("-cooldown_seconds", type=float, default=10.0)
    return parser.parse_args()


def build_command(args, backbone: str):
    profile = get_topk_profile(backbone, args.topk)
    command = [
        sys.executable,
        "train_iot_framework.py",
        "-dataset",
        args.dataset,
        "-subject_id",
        str(args.subject_id),
        "-duration",
        str(args.duration),
        "-epochs",
        str(args.epochs),
        "-batch_size",
        str(args.batch_size),
        "-backbone",
        backbone,
        "-topk",
        str(args.topk),
        "-output",
        f"queue_{backbone}_top{args.topk}_s{args.subject_id}.json",
    ]
    for key, value in profile.items():
        command.extend([f"-{key}", str(value)])
    if args.device:
        command.extend(["-device", args.device])
    return command


def main():
    args = parse_args()
    for index, backbone in enumerate(args.backbones, start=1):
        command = build_command(args, backbone)
        print({"run": index, "total": len(args.backbones), "backbone": backbone, "command": command})
        subprocess.run(command, check=True)
        if index < len(args.backbones) and args.cooldown_seconds > 0:
            time.sleep(args.cooldown_seconds)


if __name__ == "__main__":
    main()
