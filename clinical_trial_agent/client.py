"""
ClinicalTrialAgent — Client
Connects to the environment server. NEVER imports from server/.
"""
from __future__ import annotations
import requests
from typing import Any, Dict, Optional, Tuple
from .models import TrialAction, TrialObservation


class ClinicalTrialEnv:
    """
    HTTP client for ClinicalTrialAgentEnvironment.
    Use from_hub() to connect to HuggingFace Space.
    Use local() to connect to a locally running server.
    """

    def __init__(self, base_url: str, difficulty: str = "medium"):
        self.base_url = base_url.rstrip("/")
        self.difficulty = difficulty
        self.env_id: Optional[str] = None
        self._create()

    def _create(self):
        resp = requests.post(f"{self.base_url}/create", params={"difficulty": self.difficulty})
        resp.raise_for_status()
        self.env_id = resp.json()["env_id"]

    @classmethod
    def from_hub(cls, repo_id: str, difficulty: str = "medium") -> "ClinicalTrialEnv":
        """Connect to environment hosted on HuggingFace Spaces."""
        base_url = f"https://{repo_id.replace('/', '-')}.hf.space"
        return cls(base_url=base_url, difficulty=difficulty)

    @classmethod
    def local(cls, port: int = 7860, difficulty: str = "medium") -> "ClinicalTrialEnv":
        """Connect to locally running server."""
        return cls(base_url=f"http://localhost:{port}", difficulty=difficulty)

    def reset(self, idx: Optional[int] = None) -> TrialObservation:
        params = {"idx": idx} if idx is not None else {}
        resp = requests.post(f"{self.base_url}/reset/{self.env_id}", params=params)
        resp.raise_for_status()
        return TrialObservation(**resp.json()["observation"])

    def step(self, action: TrialAction) -> Tuple[TrialObservation, float, bool, Dict]:
        resp = requests.post(
            f"{self.base_url}/step/{self.env_id}",
            json=action.model_dump(exclude_none=True),
        )
        resp.raise_for_status()
        data = resp.json()
        obs = TrialObservation(**data["observation"])
        return obs, data["reward"], data["terminated"], data.get("info", {})

    def state(self) -> Dict[str, Any]:
        resp = requests.get(f"{self.base_url}/state/{self.env_id}")
        resp.raise_for_status()
        return resp.json()["state"]

    def close(self):
        if self.env_id:
            requests.delete(f"{self.base_url}/close/{self.env_id}")
            self.env_id = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
