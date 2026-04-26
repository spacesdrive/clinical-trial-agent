---
name: ClinicalTrialAgent Project
description: OpenEnv hackathon project — RL environment for clinical trial protocol design, deployed on HF Spaces
type: project
---

ClinicalTrialAgent is deployed and live on HuggingFace Spaces at:
- Space: https://huggingface.co/spaces/hydra007007/clinical-trial-agent
- Live API: https://hydra007007-clinical-trial-agent.hf.space

**Why:** Built for Meta/HuggingFace OpenEnv Hackathon India 2026 (Theme 3.1: World Modeling / Professional Tasks). First RL training environment for clinical trial protocol design.

**How to apply:** When continuing work on this project, use the live API base_url above. The local files at `/Users/akashaaprasad/Documents/clinical-trail-agent/` are the source of truth — root = HF Space root (Dockerfile at root, server/ subdir).

HF credentials: username=hydra007007, token stored in project (do not commit .env files).

What still needs doing:
- Colab training notebook (GRPO + TRL + Unsloth on Qwen2.5-3B)
- HF Blog post
- Demo video
- GitHub repo push (user will do manually)
- Register on HF OpenEnv leaderboard
