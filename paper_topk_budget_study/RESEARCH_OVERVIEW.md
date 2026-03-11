# Research Overview

## Background

This project studies a practical and research-relevant problem in motor imagery EEG classification:

how to reduce the number of EEG channels while preserving as much classification performance as possible.

This problem matters because full-channel EEG systems are expensive to deploy, inconvenient to wear, and often inefficient in real-world settings. A model that depends on many channels may perform well in the lab, but it is harder to use in lightweight, portable, and cost-sensitive scenarios.

Therefore, the real challenge is not simply to build a strong classifier on all channels. The more meaningful question is:

can we automatically identify a small but highly informative subset of channels, under an explicit channel budget, while keeping performance close to the full-channel baseline, or even exceeding it?


## Core Idea

The core idea of the current plan is to formulate channel reduction as a structured budgeted learning problem rather than a simple static ranking problem.

Instead of assigning each channel an isolated importance score and selecting the top few channels once and for all, this project treats channel selection as a learnable process guided by both:

- classification performance
- structural relationships among EEG channels

The method is designed to operate under explicit top-k budgets, such as:

- `top3`
- `top5`
- `top8`

Under each budget, the framework learns which channels should be retained so that the resulting reduced-channel model remains competitive with the corresponding full-channel classifier.


## Why This Plan

### 1. Why use multiple backbones

The goal of the project is not to claim that one specific EEG classifier is best.

The actual goal is to test whether the proposed channel selection strategy is effective across different model families. For this reason, the current study uses three representative backbones:

- `NexusNet`
- `LGGNet`
- `M-ShallowConvNet`

These models reflect different inductive biases:

- graph-oriented modeling
- local-global representation learning
- compact convolutional decoding

By validating the same channel selection framework across multiple backbones, the study can show that the contribution comes from the channel selection mechanism itself rather than from a single classifier-specific trick.


### 2. Why prioritize top5 first

The current tuning order is:

1. tune `top5`
2. transfer the best direction to `top3`
3. transfer the best direction to `top8`

This choice is deliberate.

`top5` is the most balanced budget point in the current study:

- `top3` is more aggressive and usually harder to stabilize
- `top8` is less constrained and often less persuasive as the main paper result
- `top5` is the most likely budget to best demonstrate the tradeoff between compactness and performance

By stabilizing `top5` first, the study obtains a strong tuning direction that can then be reused to explore nearby budgets more efficiently.


### 3. Why start from sub1

The current stage focuses on `sub1` first.

This is not because one subject is enough for the final conclusion, but because a method pipeline should first be validated in a controlled setting before being expanded to large-scale runs.

Running all subjects, all budgets, and all backbones too early creates unnecessary search cost and debugging noise. Starting from `sub1` allows the project to:

- validate the training and tuning logic
- identify whether the method has real potential
- stabilize the search strategy before scaling up


### 4. Why compare against full-channel baselines explicitly

The project does not define success as "the reduced-channel model still works".

That standard is too weak.

Instead, every tuned result is evaluated relative to its full-channel counterpart. The working targets are:

- minimum acceptable target: selected model >= `95%` of full baseline
- preferred target: selected model > full baseline

This makes the research question much stronger and more meaningful:

not whether channel reduction is possible in principle, but whether a well-designed selection mechanism can preserve or even improve discriminative performance under strict channel constraints.


## Why Simpler Alternatives Are Not Enough

A straightforward approach would be:

1. compute channel importance scores
2. rank all channels
3. keep the top-k channels

This is simple, but it ignores several important aspects of EEG:

- channels are not independent
- useful information often lies in combinations of channels
- spatial relationships matter
- the best subset may depend on how the downstream classifier uses the signals

So the current plan avoids treating channel selection as a one-time preprocessing step. Instead, it learns channel selection jointly with the classification objective under explicit budget control.


## Main Innovation Points

### 1. Channel selection is modeled as a structured learning problem

The project moves beyond static per-channel ranking.

Channel selection is treated as a constrained optimization problem in which the model must decide which subset of channels is most useful under a fixed top-k budget.

This is a stronger formulation because it focuses on the value of a channel subset rather than isolated channel saliency.


### 2. Graph information is used to guide selection

EEG channels have meaningful spatial and functional relationships.

The current framework explicitly uses graph-based information so that channel selection is not purely score-based but structure-aware. This helps the selector favor subsets that are not only individually strong, but also coherent as a signal-supporting subgraph.


### 3. The selection framework is backbone-agnostic

The proposed selection module is designed to wrap around multiple classification backbones rather than being fused into a single fixed architecture.

This is important because it allows the work to claim a more general contribution:

the innovation lies in the channel selection framework, not only in a specially tailored classifier.


### 4. The method studies performance under explicit channel budgets

Rather than reporting one arbitrary reduced setting, the project is organized around discrete budgets such as `top3`, `top5`, and `top8`.

This makes the work more useful and more publishable because it turns the problem into an interpretable performance-budget tradeoff study.


### 5. Full-channel knowledge is transferred into reduced-channel models

The project uses the full-channel classifier as a teacher and distills knowledge into the budgeted model.

This matters because reduced-channel models are not only weaker in raw input information, but also harder to optimize. Distillation helps bridge that gap and increases the chance that the selected-channel model can approach the full-channel reference.


### 6. Tuning is organized around a principal budget and transferred outward

The tuning strategy itself is part of the experimental design.

Instead of independently tuning every budget from scratch, the project first stabilizes the most valuable budget point and then transfers the best direction to neighboring budgets.

This reduces wasted search and creates a more coherent experimental story.


## Expected Scientific Claim

If the plan succeeds, the project will be able to support a stronger claim than:

"fewer channels can still classify EEG."

The stronger claim is:

a graph-guided, budget-aware, backbone-agnostic channel selection framework can learn compact EEG channel subsets that preserve, and in some cases potentially improve, motor imagery classification performance relative to full-channel baselines.


## Why This Matters

This direction is valuable for both theory and application.

From the research side, it pushes EEG channel selection away from heuristic ranking and toward constrained representation learning.

From the practical side, it moves toward EEG systems that are:

- easier to wear
- cheaper to deploy
- faster to compute
- more suitable for low-resource real-world scenarios


## Summary

This plan is worth pursuing because it addresses a real deployment bottleneck in EEG classification, frames channel reduction as a principled learning problem, validates the idea across multiple classifier families, and aims for a strong outcome criterion: reduced-channel models should not merely survive after pruning, but should remain competitive with, and ideally surpass, full-channel baselines under explicit top-k budgets.
