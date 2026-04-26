"""
ClinicalTrialAgent — FastAPI server
OpenEnv-compliant server with WebSocket support.
RULE: Never import from client.py here.
"""
from __future__ import annotations
import json
import threading
from typing import Dict, Any, Optional
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from .clinical_trial_agent_environment import ClinicalTrialAgentEnvironment

app = FastAPI(
    title="ClinicalTrialAgent",
    description="RL environment for drug trial protocol design",
    version="1.0.0",
)

# Thread-safe environment pool
_envs: Dict[str, ClinicalTrialAgentEnvironment] = {}
_lock = threading.Lock()
_id_counter = [0]


def _new_env(difficulty: str = "medium") -> str:
    with _lock:
        _id_counter[0] += 1
        env_id = str(_id_counter[0])
        _envs[env_id] = ClinicalTrialAgentEnvironment(difficulty=difficulty)
    return env_id


@app.get("/health")
def health():
    return {"status": "ok", "env": "ClinicalTrialAgent", "version": "1.0.0"}


@app.post("/create")
def create(difficulty: str = "medium"):
    env_id = _new_env(difficulty)
    return {"env_id": env_id}


@app.post("/reset/{env_id}")
def reset(env_id: str, idx: Optional[int] = None):
    if env_id not in _envs:
        return JSONResponse(status_code=404, content={"error": "env_id not found"})
    obs = _envs[env_id].reset(idx=idx)
    return {"observation": obs, "env_id": env_id}


@app.post("/step/{env_id}")
def step(env_id: str, action: Dict[str, Any]):
    if env_id not in _envs:
        return JSONResponse(status_code=404, content={"error": "env_id not found"})
    obs, reward, terminated, info = _envs[env_id].step(action)
    return {
        "observation": obs,
        "reward": reward,
        "terminated": terminated,
        "info": info,
        "env_id": env_id,
    }


@app.get("/state/{env_id}")
def state(env_id: str):
    if env_id not in _envs:
        return JSONResponse(status_code=404, content={"error": "env_id not found"})
    return {"state": _envs[env_id].state, "env_id": env_id}


@app.delete("/close/{env_id}")
def close(env_id: str):
    with _lock:
        _envs.pop(env_id, None)
    return {"closed": env_id}


@app.websocket("/ws/{env_id}")
async def websocket_endpoint(websocket: WebSocket, env_id: str):
    """WebSocket endpoint for persistent multi-turn sessions."""
    await websocket.accept()
    if env_id not in _envs:
        await websocket.send_json({"error": "env_id not found"})
        await websocket.close()
        return
    try:
        while True:
            data = await websocket.receive_json()
            cmd = data.get("cmd")
            if cmd == "reset":
                obs = _envs[env_id].reset(idx=data.get("idx"))
                await websocket.send_json({"observation": obs})
            elif cmd == "step":
                action = data.get("action", {})
                obs, reward, terminated, info = _envs[env_id].step(action)
                await websocket.send_json({
                    "observation": obs,
                    "reward": reward,
                    "terminated": terminated,
                    "info": info,
                })
                if terminated:
                    break
            elif cmd == "state":
                await websocket.send_json({"state": _envs[env_id].state})
            else:
                await websocket.send_json({"error": f"unknown cmd: {cmd}"})
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)
