OFFICIAL_BACKBONE_DEFAULTS = {
    "nexusnet": {
        "epochs": 2000,
        "patience": 500,
        "batch_size": 64,
        "lr": 5e-3,
        "w_decay": 0.0,
        "dropout": 0.25,
        "backbone_kwargs": {
            "pool_mode": "mean",
            "f1": 8,
            "f2": 16,
            "kernel_length": 64,
        },
    },
    "lggnet": {
        "epochs": 200,
        "patience": 20,
        "batch_size": 64,
        "lr": 1e-3,
        "w_decay": 0.0,
        "dropout": 0.5,
        "backbone_kwargs": {
            "sampling_rate": 250,
            "num_t_filters": 64,
            "out_graph": 32,
            "pool": 16,
            "pool_step_rate": 0.25,
        },
    },
    "mshallowconvnet": {
        "epochs": 1000,
        "patience": 1000,
        "batch_size": 64,
        "lr": 2e-3,
        "w_decay": 7.5e-2,
        "dropout": 0.5,
        "backbone_kwargs": {
            "sampling_rate": 250,
            "depth": 24,
            "temporal_kernel_seconds": 0.12,
            "temporal_stride": 2,
            "pool_length": 75,
            "pool_stride": 15,
        },
    },
}


def get_official_defaults(backbone: str) -> dict:
    if backbone not in OFFICIAL_BACKBONE_DEFAULTS:
        raise ValueError(f"Unsupported backbone: {backbone}")
    payload = OFFICIAL_BACKBONE_DEFAULTS[backbone]
    result = {k: v for k, v in payload.items() if k != "backbone_kwargs"}
    result["backbone_kwargs"] = dict(payload["backbone_kwargs"])
    return result


def apply_missing_training_defaults(args):
    defaults = get_official_defaults(args.backbone)
    for key in ["epochs", "patience", "batch_size", "lr", "w_decay", "dropout"]:
        if getattr(args, key) is None:
            setattr(args, key, defaults[key])
    return args
