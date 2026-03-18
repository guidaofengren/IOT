# Paper Top-k Budget Study

This folder contains the minimal code needed for the paper-track experiments only.

Scope:
- Backbones: `nexusnet`, `lggnet`, `mshallowconvnet`
- Budgets: `topk=3/5/8`
- Main focus: baseline-vs-selected comparison and unified benchmark runs
- Protocol: training hyperparameters are selected automatically from each backbone's official defaults unless explicitly overridden

Main entrypoints:
- `train_iot_baseline.py`: train the full-channel baseline for one backbone
- `train_iot_framework.py`: train the graph-guided top-k selector for one backbone and one budget
- `compare_iot_baseline_vs_selected.py`: run full vs `topk=3/5/8` for the three paper backbones
- `benchmark_iot_framework.py`: run selector-only benchmark for the three paper backbones
- `experiment_profiles.py`: paper-track hyperparameter profiles for each backbone and budget

Included modules:
- `models/`: only the components needed by NexusNet, LGGNet, M-ShallowConvNet, and the IoT wrapper
- `tools/`: dataset loading, training helpers, complexity helpers, and channel export helpers

Notes:
- Historical exploratory scripts and JSON result files are intentionally excluded.
- Current defaults target the paper-track setup with the three selected backbones.

Current released default:
- Main method: `run_stable_standalone_pipeline.py`
- Global budget: `topk=12`
- Selector defaults: `epochs=120`, `batch_size=32`, `lr=1e-3`, `hidden=64`, `dropout=0.1`
- NexusNet subset defaults: `epochs=220`, `patience=35`, `batch_size=64`, `lr=0.0018`, `dropout=0.22`
- Stability defaults: `min_epochs=90`, `val_smooth_window=8`, `grad_clip=1.0`, `ema_decay=0.99`
- Final pipeline defaults keep `channel_weight_mode=none`, `selector_post_rule=none`, `distill_alpha=0.0`
