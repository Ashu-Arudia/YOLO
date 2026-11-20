"""Pydantic request/response schemas"""
from __future__ import annotations
from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class MatchCreate(BaseModel):
    name: str
    source_type: str = "file"   # "file" | "webcam"


class MatchResponse(BaseModel):
    id: str
    name: str
    video_path: Optional[str]
    source_type: str
    status: str
    created_at: datetime
    started_at: Optional[datetime]
    ended_at: Optional[datetime]

    model_config = {"from_attributes": True}


class AnalyticsSnapshot(BaseModel):
    match_id: str
    frame_idx: int
    fps: float
    player_count: int
    team_a_count: int
    team_b_count: int
    team_a_possession: float
    team_b_possession: float
    ball_x: Optional[float]
    ball_y: Optional[float]
    holder_team: Optional[str]
    calibrated: bool


class PlayerStats(BaseModel):
    track_id: int
    team: str
    distance_px: float
    max_speed_px: float
