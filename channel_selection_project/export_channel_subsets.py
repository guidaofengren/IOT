import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-input", required=True, help="Path to a result JSON with ranking.")
    parser.add_argument("-topk", type=int, nargs="+", default=[3, 5, 8])
    parser.add_argument("-output", default="channel_subsets.json")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.input, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    ranking = payload["ranking"]
    subsets = {}
    for topk in args.topk:
        subsets[str(topk)] = {
            "channels": [item["channel"] for item in ranking[:topk]],
            "indices": [item["index"] for item in ranking[:topk]],
        }

    output = {
        "dataset": payload.get("dataset"),
        "subject_id": payload.get("subject_id"),
        "backbone": payload.get("backbone"),
        "subsets": subsets,
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
    print(output)


if __name__ == "__main__":
    main()
