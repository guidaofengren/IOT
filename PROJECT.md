# Project Scope

This repository currently tracks one completed model line: `NexusNet` under the graph-guided top-k channel-selection framework.

## What is finished

- `NexusNet` full-channel baseline
- `NexusNet` top-k channel-selection training
- `NexusNet`-centered tuning and result comparison workflow

## What is not yet finalized

- `LGGNet` tuning
- `M-ShallowConvNet` tuning
- Any paper claim that depends on those unfinished backbones

## Code layout

- `models/`: model definitions
- `tools/`: data loading, graph features, complexity, and training helpers

## Documentation rule

For now, any project summary, README statement, or paper-facing description should treat `NexusNet` as the only completed tuned model in this repository.
