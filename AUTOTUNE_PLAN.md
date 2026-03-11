# Paper Top-k Autotune Plan

## Goal

This plan describes the current experiment protocol for automatic tuning on `sub1` in the `paper_topk_budget_study` workspace.

The goal is to compare a full-channel classifier against a graph-guided top-k channel selection framework, using three backbone models:

- `nexusnet`
- `lggnet`
- `mshallowconvnet`

The main target is not only to maintain usability after channel reduction, but to push the selected-channel model as close as possible to the full-channel baseline, and ideally exceed it.


## Current Principle

The current tuning principle is:

- first tune `top5`
- then transfer the best `top5` direction to `top3`
- then transfer the best direction again to `top8`

This order is intentional:

- `top5` is the most likely main paper budget point
- once `top5` is stable, `top3` and `top8` can reuse a better starting direction
- this reduces random search waste and shortens total tuning time


## Backbone Scope

The current paper-track scope is limited to three backbones:

1. `nexusnet`
2. `lggnet`
3. `mshallowconvnet`

The old `shallowconvnet` variant has been removed from the active paper pipeline and replaced by `mshallowconvnet`.


## Subject Scope

The current automatic tuning job is for:

- dataset: `bciciv2a`
- subject: `sub1` (`subject_id=1`)

The idea is to first make the protocol work and stabilize on `sub1`, then later expand to more subjects after the tuning logic is validated.


## Hyperparameter Policy

The current hyperparameter policy is no longer "all models forced to 1000 epochs".

Instead, the plan is:

- each backbone uses its own official-style default training setup
- command-line overrides are still allowed when needed
- the tuning script automatically fills missing values from the backbone-specific official profile

Current official-style defaults are:

### NexusNet

- epochs: `2000`
- patience: `500`
- batch size: `64`
- learning rate: `5e-3`
- weight decay: `0`
- dropout: `0.25`

### LGGNet

- epochs: `200`
- patience: `20`
- batch size: `64`
- learning rate: `1e-3`
- weight decay: `0`
- dropout: `0.5`

### M-ShallowConvNet

- epochs: `1000`
- patience: `1000`
- batch size: `64`
- learning rate: `2e-3`
- weight decay: `7.5e-2`
- dropout: `0.5`

These are treated as the base training defaults. The selector-related tuning parameters can still vary by `backbone` and `topk`.


## Search Target

The evaluation target is defined in two levels:

1. minimum acceptable target:
   selected model test accuracy >= `0.95 * full`
2. preferred target:
   selected model test accuracy > `full`

In practice, the tuning script ranks results with the following priority:

1. whether the selected model beats the full baseline
2. whether it reaches at least `95%` of the full baseline
3. ratio of `selected / full`
4. validation accuracy
5. test accuracy


## What Is Tuned

The tuning process keeps the classifier backbone choice fixed and searches around selector/training settings such as:

- `patience`
- `batch_size`
- `lr`
- `w_decay`
- `dropout`
- `warmup_epochs`
- `stage2_epochs`
- `distill_alpha`
- `feature_distill_alpha`
- `sparse_lambda`
- `smooth_lambda`
- `separation_lambda`
- `start_temp`
- `end_temp`

The search is not a blind global brute-force search.

Instead, it is staged:

1. optimizer and regular training scale
2. warmup and stage-2 finetuning schedule
3. distillation strength
4. selector regularization strength
5. temperature schedule

This keeps the search space manageable while still letting the framework adapt by model and budget.


## Baseline Policy

Before tuning any selected-channel model, the pipeline ensures a full-channel baseline exists for that backbone on the same subject.

Those baseline results are stored as files like:

- `compare_nexusnet_full_s1.json`
- `compare_lggnet_full_s1.json`
- `compare_mshallowconvnet_full_s1.json`

These files are the reference points used for:

- pass/fail thresholding
- selected/full ratio computation
- ranking tuned trials


## Automatic Tuning Workflow

The automatic tuning workflow is:

1. build or load the full-channel baseline for each backbone
2. run `top5` trials first
3. sort the completed `top5` results
4. take the best `top5` profile as the seed direction
5. run `top3` using that seed direction plus local variations
6. run `top8` using that seed direction plus local variations
7. continuously write per-trial JSON outputs
8. write the final global summary after all groups complete

This makes the later budgets more sample-efficient than starting from scratch each time.


## Runtime Strategy

The runtime strategy is designed around real hardware constraints rather than only GPU utilization.

The system currently supports:

- hidden background execution
- multiple concurrent training jobs
- reserved RAM and VRAM for normal machine usage

Current runtime choice:

- run up to `3` jobs in parallel
- reserve `12 GB` system RAM
- reserve `2 GB` GPU VRAM

The reason for using resource reservation is:

- GPU utilization alone is not enough to judge safe concurrency
- tuning jobs may not use high compute continuously, but they can still spike memory
- preserving free RAM/VRAM keeps the machine responsive for normal work


## File Outputs

The main output files are:

### Baselines

- `compare_<backbone>_full_s1.json`

### Trial Results

- `tune_<backbone>_top<k>_trialXXX_s1.json`

Examples:

- `tune_nexusnet_top5_trial001_s1.json`
- `tune_lggnet_top3_trial004_s1.json`
- `tune_mshallowconvnet_top8_trial002_s1.json`

### Global Summary

- `paper_autotune_summary_s1.json`

### Logs

- `autotune_stdout.log`
- `autotune_stderr.log`


## Current Status Interpretation

The tuning is considered still running if either of the following is true:

- the `tune_paper_topk.py` main process is still alive
- new `tune_*_s1.json` files are still being created or updated

The tuning is considered complete when:

- the main process exits
- no training child processes remain
- the global summary file has been fully written


## Why This Plan Makes Sense

This plan is reasonable for the current paper stage because:

- it narrows effort to the three most relevant backbones
- it tunes on `sub1` first instead of spreading effort too early
- it gives `top5` priority as the most important paper budget
- it keeps backbone defaults aligned with official model behavior
- it allows selector-specific tuning where the paper method actually differs
- it balances search speed and system stability through controlled concurrency


## Next-Step Expectations

Once the current `sub1` tuning stabilizes, the likely next steps are:

1. freeze the best setting per `(backbone, topk)`
2. rerun those best settings for verification
3. compare selected-channel models directly against full baselines
4. decide which budget points become the final paper tables
5. extend the validated protocol to additional subjects


## One-Sentence Summary

The current plan is to automatically tune `nexusnet`, `lggnet`, and `mshallowconvnet` on `sub1`, using official backbone defaults, prioritizing `top5`, transferring good directions to `top3/8`, and ranking results by whether the selected-channel model can match or exceed the full-channel baseline under a resource-safe parallel search workflow.
