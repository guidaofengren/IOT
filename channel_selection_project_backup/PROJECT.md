# Channel Selection Project

This folder contains the consolidated codebase for the EEG channel-selection study.

## Current focus

- Reproduce the aligned `NexusNet` baseline
- Compare channel-importance methods
- Evaluate reduced-channel retraining with `top-k` channels

## Main entrypoints

- `main.py`: baseline training and testing
- `channel_importance_analysis.py`: post-hoc masking-based channel ranking
- `retrain_topk_channels.py`: retrain the baseline with only `top-k` channels kept

## Code layout

- `models/`: model definitions
- `tools/`: data loading, utilities, and training helpers
- `bciciv2a_checkpoint/`: pretrained baseline checkpoints for BCIC-IV-2a
- `bciciv2b_checkpoint/`: pretrained baseline checkpoints for BCIC-IV-2b
