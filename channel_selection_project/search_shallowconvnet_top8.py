import argparse
import json
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", default="bciciv2a", choices=["bciciv2a", "bciciv2b"])
    parser.add_argument("-subject_id", type=int, default=1)
    parser.add_argument("-epochs", type=int, nargs="+", default=[50, 100])
    parser.add_argument("-warmup_epochs", type=int, default=20)
    parser.add_argument("-separation_lambda", type=float, nargs="+", default=[0.01, 0.02, 0.05])
    parser.add_argument("-batch_size", type=int, default=8)
    parser.add_argument("-output", default="shallowconvnet_top8_search_s1.json")
    return parser.parse_args()


def main():
    args = parse_args()
    python_exec = sys.executable
    results = []
    for epochs in args.epochs:
        for sep in args.separation_lambda:
            output = f"shallowconvnet_top8_e{epochs}_sep{str(sep).replace('.', 'p')}_s{args.subject_id}.json"
            command = [
                python_exec,
                "train_iot_framework.py",
                "-dataset",
                args.dataset,
                "-subject_id",
                str(args.subject_id),
                "-backbone",
                "shallowconvnet",
                "-epochs",
                str(epochs),
                "-warmup_epochs",
                str(min(args.warmup_epochs, max(1, epochs // 2))),
                "-batch_size",
                str(args.batch_size),
                "-topk",
                "8",
                "-separation_lambda",
                str(sep),
                "-output",
                output,
            ]
            subprocess.run(command, check=True)
            with open(output, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            payload["search_epochs"] = epochs
            payload["search_separation_lambda"] = sep
            results.append(payload)

    results.sort(key=lambda item: item["test_acc"], reverse=True)
    summary = {
        "dataset": args.dataset,
        "subject_id": args.subject_id,
        "results": results,
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(
        {
            "output": args.output,
            "best_test_acc": results[0]["test_acc"] if results else None,
            "best_setting": {
                "epochs": results[0]["search_epochs"],
                "separation_lambda": results[0]["search_separation_lambda"],
            }
            if results
            else None,
        }
    )


if __name__ == "__main__":
    main()
