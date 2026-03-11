from typing import List


def ranking_to_channels(ranking, topk: int) -> List[str]:
    return [item["channel"] for item in ranking[:topk]]


def ranking_to_indices(ranking, topk: int):
    return [item["index"] for item in ranking[:topk]]
