import time

import torch


def count_parameters(model) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def benchmark_forward(model, sample: torch.Tensor, device: str, warmup: int = 10, steps: int = 30):
    model = model.to(device)
    sample = sample.to(device)
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(sample)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(steps):
            model(sample)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    return elapsed / steps
