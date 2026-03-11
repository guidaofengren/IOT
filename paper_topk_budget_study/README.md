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
