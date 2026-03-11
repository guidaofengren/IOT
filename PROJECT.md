# Channel Selection Project

This folder contains the aligned `NexusNet` baseline and an IoT-oriented sparse EEG decoding extension.

## IoT-oriented direction

- Graph-guided channel selection
- Explicit `top-k` channel budget control
- Model-agnostic wrapping for multiple EEG backbones
- Complexity and latency reporting for edge deployment

## Main entrypoints

- `main.py`: baseline `NexusNet` training and testing
- `channel_importance_analysis.py`: post-hoc masking-based channel ranking
- `retrain_topk_channels.py`: retrain the baseline with only `top-k` channels kept
- `train_channel_selector.py`: train-time scalar channel gate
- `train_gnn_channel_selector.py`: graph-based channel gate
- `train_iot_framework.py`: graph-guided, model-agnostic sparse channel selection framework
- `evaluate_iot_framework.py`: evaluate a saved IoT checkpoint and export selected channels
- `export_channel_subsets.py`: export deployment-ready channel subsets from a ranking JSON
- `benchmark_iot_framework.py`: batch benchmark across backbones and channel budgets

## New framework components

- `models/IoTChannelSelectionFramework.py`: selector, wrapper, and backbone builder
- `models/LGGNetBackbone.py`: LGGNet-style public graph backbone adapted from the official LGG repository
- `models/ShallowConvNetBackbone.py`: ShallowConvNet-style baseline for low-channel comparison
- `tools/complexity.py`: parameter counting and forward latency benchmarking
- `tools/channel_selection.py`: ranking helpers for channel subset export

## Code layout

- `models/`: model definitions
- `tools/`: data loading, graph features, complexity, and training helpers
- `bciciv2a_checkpoint/`: pretrained baseline checkpoints for BCIC-IV-2a
- `bciciv2b_checkpoint/`: pretrained baseline checkpoints for BCIC-IV-2b
