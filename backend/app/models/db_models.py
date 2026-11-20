"""
SQLAlchemy ORM Models
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.database.session import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return str(uuid.uuid4())


class Match(Base):
    __tablename__ = "matches"

    id: Mapped[str]       = mapped_column(String, primary_key=True, default=_uuid)
    name: Mapped[str]     = mapped_column(String(200), nullable=False)
    video_path: Mapped[str | None] = mapped_column(String(500))
    source_type: Mapped[str] = mapped_column(String(20), default="file")  # "webcam" | "file"
    status: Mapped[str]   = mapped_column(String(20), default="pending")
    # pending | processing | completed | error | stopped

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    ended_at:   Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Relationships
    analytics: Mapped[list[MatchAnalytics]] = relationship(
        "MatchAnalytics", back_populates="match", cascade="all, delete-orphan"
    )
    players: Mapped[list[PlayerRecord]] = relationship(
        "PlayerRecord", back_populates="match", cascade="all, delete-orphan"
    )


class MatchAnalytics(Base):
    __tablename__ = "match_analytics"

    id: Mapped[int]         = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_id: Mapped[str]   = mapped_column(ForeignKey("matches.id", ondelete="CASCADE"))
    frame_idx: Mapped[int]  = mapped_column(Integer)
    fps: Mapped[float]      = mapped_column(Float, default=0.0)
    player_count: Mapped[int] = mapped_column(Integer, default=0)
    team_a_count: Mapped[int] = mapped_column(Integer, default=0)
    team_b_count: Mapped[int] = mapped_column(Integer, default=0)
    team_a_possession: Mapped[float] = mapped_column(Float, default=50.0)
    team_b_possession: Mapped[float] = mapped_column(Float, default=50.0)
    ball_x: Mapped[float | None] = mapped_column(Float)
    ball_y: Mapped[float | None] = mapped_column(Float)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    match: Mapped[Match] = relationship("Match", back_populates="analytics")


class PlayerRecord(Base):
    __tablename__ = "player_records"

    id: Mapped[int]          = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_id: Mapped[str]    = mapped_column(ForeignKey("matches.id", ondelete="CASCADE"))
    track_id: Mapped[int]    = mapped_column(Integer)
    team: Mapped[str]        = mapped_column(String(20), default="unknown")
    distance_px: Mapped[float] = mapped_column(Float, default=0.0)
    max_speed_px: Mapped[float] = mapped_column(Float, default=0.0)
    first_seen: Mapped[int]  = mapped_column(Integer, default=0)
    last_seen: Mapped[int]   = mapped_column(Integer, default=0)

    match: Mapped[Match] = relationship("Match", back_populates="players")
