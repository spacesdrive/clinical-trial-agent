# ClinicalTrialAgent — Project Context

**Hackathon:** Meta/HuggingFace OpenEnv Hackathon India 2026  
**Theme:** #3.1 World Modeling / Professional Tasks  
**Status:** Deployed and live  
**Date created:** 2026-04-26

---

## Credentials & URLs

| Item | Value |
|------|-------|
| HF Username | hydra007007 |
| HF Space URL | https://huggingface.co/spaces/hydra007007/clinical-trial-agent |
| Live API base_url | https://hydra007007-clinical-trial-agent.hf.space |
| Health endpoint | https://hydra007007-clinical-trial-agent.hf.space/health |

**HF Token:** [REDACTED — store in your password manager, never commit tokens]

---

## What Was Built

An RL environment where an LLM agent designs regulatory-compliant clinical trial
protocols for complex diseases. The agent receives a disease scenario and must fill
in 9 protocol fields through multi-turn interaction, learning from FDA/ICH rule violations.

**Why it's novel:** First RL training environment for clinical trial protocol design.
Bad protocol design costs $100M+ and delays life-saving drugs by years.

---

## File Structure (Workspace Root = HF Space Root)

```
/Users/akashaaprasad/Documents/clinical-trail-agent/
├── README.md               ← HF Space metadata + hackathon description
├── Dockerfile              ← HF Space Docker build (python:3.11-slim, port 7860)
├── openenv.yaml            ← OpenEnv manifest (judges check this)
├── pyproject.toml          ← Python package definition
├── models.py               ← TrialAction + TrialObservation Pydantic models
├── client.py               ← HTTP client (ClinicalTrialEnv class)
├── __init__.py             ← Top-level package exports
├── .gitignore              ← Standard Python + secrets gitignore
├── server/
│   ├── __init__.py
│   ├── app.py              ← FastAPI server (9 endpoints + WebSocket)
│   ├── clinical_trial_agent_environment.py  ← Core RL env logic
│   └── requirements.txt
└── clinical_trial_agent/   ← Local pip-installable package (same code)
    ├── __init__.py
    ├── models.py
    ├── client.py
    ├── openenv.yaml
    ├── pyproject.toml
    ├── README.md
    └── server/
        ├── __init__.py
        ├── app.py
        ├── clinical_trial_agent_environment.py
        ├── requirements.txt
        └── Dockerfile
```

---

## Environment Design

### Disease Scenarios (6 total)
1. Type 2 Diabetes (Phase 3, 24 weeks)
2. Treatment-Resistant Hypertension (Phase 2, 12 weeks)
3. Major Depressive Disorder (Phase 2, 8 weeks)
4. Early Alzheimer's Disease (Phase 3, 78 weeks)
5. Moderate-to-Severe Asthma (Phase 3, 52 weeks)
6. Acute Ischemic Stroke (Phase 2, 13 weeks)

### 9 Required Protocol Fields
- trial_phase (Phase 1/2/3)
- study_design (RCT/crossover/open-label/dose-escalation)
- sample_size (integer)
- duration_weeks (integer)
- inclusion_criteria (List[str])
- exclusion_criteria (List[str])
- primary_endpoint (str)
- statistical_power (float, FDA requires ≥0.80)
- safety_monitoring (str)

### Composable Reward Rubric
| Component | Weight | Logic |
|-----------|--------|-------|
| Completeness | 30% | Fields filled / 9 |
| Regulatory validity | 35% | FDA/ICH rule checker (phase, sample, power, design) |
| Scientific coherence | 20% | Cross-field consistency (phase vs sample, crossover vs duration) |
| Efficiency bonus | 15% | Steps saved vs max_steps |

### Difficulty Levels
| Level | max_steps | num_diseases | hints |
|-------|-----------|--------------|-------|
| easy | 10 | 2 | True |
| medium | 14 | 4 | False |
| hard | 8 | 6 | False |

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /health | Health check |
| POST | /create?difficulty=medium | Create new env, returns env_id |
| POST | /reset/{env_id} | Reset to new scenario, returns observation |
| POST | /step/{env_id} | Submit action JSON, returns obs+reward+terminated |
| GET | /state/{env_id} | Get raw env state |
| DELETE | /close/{env_id} | Clean up env |
| WS | /ws/{env_id} | WebSocket for persistent sessions |

---

## Quick Start for Colab Training Notebook

```python
from huggingface_hub import hf_hub_download
import requests

# Connect to live environment
base_url = "https://hydra007007-clinical-trial-agent.hf.space"

# Create and reset
r = requests.post(f"{base_url}/create", params={"difficulty": "medium"})
env_id = r.json()["env_id"]

r = requests.post(f"{base_url}/reset/{env_id}")
obs = r.json()["observation"]
print(f"Disease: {obs['disease']}")

# Submit action
action = {
    "trial_phase": "Phase 3",
    "study_design": "RCT",
    "sample_size": 500,
    "duration_weeks": 52,
    "inclusion_criteria": ["Adults 18-75", "Confirmed diagnosis"],
    "exclusion_criteria": ["Pregnant women", "Severe comorbidities"],
    "primary_endpoint": "Primary endpoint reduction at 52 weeks",
    "statistical_power": 0.90,
    "safety_monitoring": "Independent DSMB meets quarterly with predefined stopping rules",
    "submit_protocol": True
}
r = requests.post(f"{base_url}/step/{env_id}", json=action)
data = r.json()
print(f"Score: {data['reward']}")
print(f"Breakdown: {data['info']['score_breakdown']}")
```

---

## What Still Needs To Be Done (Phase 2)

1. **Training notebook (Colab):** Train Qwen2.5-3B with GRPO on this environment
   - Use TRL + Unsloth for efficient training
   - Log reward curves to W&B
   - Save plots to `plots/` directory
   - Update README with actual training results

2. **HuggingFace Blog post:** Write submission blog post

3. **Demo video:** Record 2-min demo

4. **GitHub repo:** User will push to GitHub for judging URL
   - `cd /Users/akashaaprasad/Documents/clinical-trail-agent`
   - `git init && git add . && git commit -m "feat: ClinicalTrialAgent OpenEnv hackathon submission"`
   - Push to GitHub

5. **Register on HF OpenEnv leaderboard:** Submit entry at hackathon page

---

## Smoke Test Results (local, 2026-04-26)

```
Health: {'status': 'ok', 'env': 'ClinicalTrialAgent', 'version': '1.0.0'}
Created env_id: 1
Disease: Type 2 Diabetes
FINAL SCORE: 0.97
BREAKDOWN: completeness=0.3, regulatory_validity=0.35, 
           scientific_coherence=0.2, efficiency_bonus=0.12
✅ ALL SMOKE TESTS PASSED
```

---

## Notes for Next AI Agent

- This project uses **relative imports** in server/ (`from .clinical_trial_agent_environment import ...`)
- The workspace root IS the HF Space root (flat structure with Dockerfile at root)
- `clinical_trial_agent/` subfolder is the same code as a pip-installable Python package
- The HF Space is deployed at **https://huggingface.co/spaces/hydra007007/clinical-trial-agent**
- The live API URL is **https://hydra007007-clinical-trial-agent.hf.space**
- Server runs with `uvicorn server.app:app --host 0.0.0.0 --port 7860`
- All FDA/ICH regulatory rules are implemented in `_validate_protocol()` method
- The reward rubric is in `_compute_reward()` — composable, 4 independent signals
