---
title: ClinicalTrialAgent
emoji: 🧬
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
tags:
  - openenv
  - rl-environment
  - clinical-trials
  - healthcare
  - multi-turn
  - grpo
---

# 🧬 ClinicalTrialAgent

**An RL environment for training LLMs to design regulatory-compliant clinical trial protocols.**

> *Bad trial design costs $100M+ and delays life-saving drugs by years. We built the first RL training environment to teach LLMs to get it right.*

[![OpenEnv](https://img.shields.io/badge/OpenEnv-latest-blue)](https://github.com/meta-pytorch/OpenEnv)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace%20Space-yellow)](https://huggingface.co/spaces/hydra007007/clinical-trial-agent)

---

## Problem

Clinical trial protocol design requires 8–15 sequential decisions — choosing trial phase, study design, sample size, statistical power, inclusion/exclusion criteria, safety monitoring — all of which must comply with FDA/ICH regulatory guidelines. Current LLMs fail at this because:

1. They make isolated decisions, not coherent multi-turn protocols
2. They don't learn from regulatory rule violations
3. No RL training environment existed for this domain

**ClinicalTrialAgent is the first RL environment to address this.**

---

## Environment

The agent receives a disease scenario (e.g., *"Type 2 Diabetes: design a Phase 2/3 trial targeting HbA1c reduction"*) and must fill in 9 protocol fields through multi-turn interaction:

| Field | Description |
|-------|-------------|
| `trial_phase` | Phase 1 / Phase 2 / Phase 3 |
| `study_design` | RCT / crossover / open-label / dose-escalation |
| `sample_size` | Number of participants |
| `duration_weeks` | Trial duration |
| `inclusion_criteria` | Who can participate |
| `exclusion_criteria` | Who is excluded |
| `primary_endpoint` | What is measured |
| `statistical_power` | FDA requires ≥ 0.80 |
| `safety_monitoring` | DSMB plan |

### Reward Rubric (Composable — 4 independent signals)

| Rubric | Weight | What it checks |
|--------|--------|----------------|
| Completeness | 30% | All 9 fields filled |
| Regulatory validity | 35% | FDA/ICH rule compliance |
| Scientific coherence | 20% | Cross-field consistency |
| Efficiency bonus | 15% | Steps saved vs maximum |

The rubric is **hard to game** — an agent that fills fields with nonsense scores low on regulatory validity and scientific coherence.

### Difficulty Levels

| Level | Max Steps | Diseases | Hints |
|-------|-----------|----------|-------|
| easy | 10 | 2 | ✓ |
| medium | 14 | 4 | ✗ |
| hard | 8 | 6 | ✗ |

---

## Training Results

> *Plots below from Qwen2.5-3B trained with GRPO on ClinicalTrialAgent (medium difficulty)*

![Reward Curve](plots/reward_curve.png)
*Fig 1: Episode reward over 500 training steps. Trained agent vs untrained baseline.*

![Difficulty Breakdown](plots/difficulty_breakdown.png)
*Fig 2: Score by difficulty level. Trained model shows 2.8× improvement at hard difficulty.*

**Training notebook:** [Open in Colab](https://colab.research.google.com/drive/COLAB_LINK_HERE)

---

## Quick Start

```python
from clinical_trial_agent import ClinicalTrialEnv, TrialAction

# Connect to hosted environment
env = ClinicalTrialEnv.from_hub("hydra007007/clinical-trial-agent", difficulty="medium")

obs = env.reset()
print(f"Disease: {obs.disease}")
print(f"Goal: {obs.target_endpoint_goal}")

# Submit one turn
action = TrialAction(
    trial_phase="Phase 3",
    study_design="RCT",
    sample_size=450,
    statistical_power=0.90,
)
obs, reward, terminated, info = env.step(action)
print(f"Reward so far: {reward:.3f}")
print(f"Errors: {obs.validation_errors}")
```

---

## Resources

- 🤗 **Environment Space**: https://huggingface.co/spaces/hydra007007/clinical-trial-agent
- 📓 **Training Colab**: [Link will be added]
- 📝 **HuggingFace Blog**: [Link will be added]
- 🎥 **Demo Video**: [Link will be added]

---

## Hackathon

Built for **Meta/HuggingFace OpenEnv Hackathon India 2026** — Theme #3.1 World Modeling / Professional Tasks.

Uses OpenEnv latest release with TRL GRPO + Unsloth training on Qwen2.5-3B.
