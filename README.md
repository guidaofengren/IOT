# Paper Top-k Budget Study

This repository contains the current paper-track code for EEG channel-budget experiments.

## Current status

- The only model with completed parameter tuning is `NexusNet`.
- `LGGNet` and `M-ShallowConvNet` are still kept in the codebase, but they have not been finalized as part of the current experimental results.
- The present reproducible result path is therefore centered on `NexusNet + top-k channel selection`.

## Main entrypoints

- `train_iot_baseline.py`: train the full-channel baseline for one backbone
- `train_iot_framework.py`: train the graph-guided top-k selector for one backbone and one budget
- `compare_iot_baseline_vs_selected.py`: run full vs `topk=3/5/8` for the three paper backbones
- `benchmark_iot_framework.py`: run selector-only benchmark for the three paper backbones
- `experiment_profiles.py`: paper-track hyperparameter profiles for each backbone and budget

## Included modules

- `models/`: only the components needed by NexusNet, LGGNet, M-ShallowConvNet, and the IoT wrapper
- `tools/`: dataset loading, training helpers, complexity helpers, and channel export helpers

## Notes

- Existing multi-backbone code is retained for later extension, but it is not part of the current finalized result set.
- When describing finished results in the paper or README, treat `NexusNet` as the only completed tuned model at this stage.

Current released default:
- Main method: `run_stable_standalone_pipeline.py`
- Global budget: `topk=12`
- Selector defaults: `epochs=120`, `batch_size=32`, `lr=1e-3`, `hidden=64`, `dropout=0.1`
- NexusNet subset defaults: `epochs=220`, `patience=35`, `batch_size=64`, `lr=0.0018`, `dropout=0.22`
- Stability defaults: `min_epochs=90`, `val_smooth_window=8`, `grad_clip=1.0`, `ema_decay=0.99`
- Final pipeline defaults keep `channel_weight_mode=none`, `selector_post_rule=none`, `distill_alpha=0.0`
