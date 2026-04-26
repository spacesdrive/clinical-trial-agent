"""
ClinicalTrialAgent — models.py
Defines Action and Observation Pydantic models.
RULE: client.py imports from here. server/ NEVER imports from client.py.
"""
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional, List


class TrialAction(BaseModel):
    """
    The agent submits protocol fields as key-value pairs.
    Each turn the agent can update one or more fields.
    Field names must match exactly (snake_case).
    """
    trial_phase: Optional[str] = Field(None, description="One of: Phase 1, Phase 2, Phase 3")
    study_design: Optional[str] = Field(None, description="One of: RCT, crossover, open-label, dose-escalation")
    sample_size: Optional[int] = Field(None, description="Number of participants (integer)")
    duration_weeks: Optional[int] = Field(None, description="Trial duration in weeks (integer)")
    inclusion_criteria: Optional[List[str]] = Field(None, description="List of inclusion criteria strings")
    exclusion_criteria: Optional[List[str]] = Field(None, description="List of exclusion criteria strings")
    primary_endpoint: Optional[str] = Field(None, description="Primary endpoint measure string")
    statistical_power: Optional[float] = Field(None, description="Power value 0.0–1.0, FDA requires >= 0.80")
    safety_monitoring: Optional[str] = Field(None, description="Safety monitoring plan description")
    submit_protocol: Optional[bool] = Field(False, description="Set True to finalize and score the protocol")


class TrialObservation(BaseModel):
    """What the agent sees each turn."""
    disease: str
    patient_population: str
    target_endpoint_goal: str
    difficulty: str
    current_protocol: dict
    validation_errors: List[str]
    completeness_pct: float
    steps_used: int
    max_steps: int
    score_breakdown: Optional[dict] = None
    message: str
