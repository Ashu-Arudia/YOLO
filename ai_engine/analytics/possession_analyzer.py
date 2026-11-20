"""
PossessionAnalyzer
──────────────────
Estimates which team currently has ball possession based on
proximity of tracked players to the ball.

Outputs:
  team_a_pct  : float  0-100
  team_b_pct  : float  0-100
  holder_id   : int | None   (track_id of nearest player)
  holder_team : str

All values are *estimates* — not authoritative match events.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from ai_engine.config import POSSESSION_SMOOTH_ALPHA, POSSESSION_MAX_DIST_PX

logger = logging.getLogger(__name__)


class PossessionAnalyzer:
    def __init__(self) -> None:
        self._team_a_accum: float = 0.0
        self._team_b_accum: float = 0.0
        self._total_frames: int   = 0
        # EMA possession
        self._ema_a: float = 50.0
        self._ema_b: float = 50.0

    def update(
        self,
        ball_pos: Optional[Tuple[float, float]],
        player_tracks,              # List[TrackInfo]
    ) -> Dict:
        """
        Update possession for this frame. Returns current stats dict.
        """
        holder_id   = None
        holder_team = "unknown"

        if ball_pos and player_tracks:
            bx, by = ball_pos
            min_dist = float("inf")
            for track in player_tracks:
                if track.centroid is None:
                    continue
                px, py = track.centroid
                d = float(np.hypot(bx - px, by - py))
                if d < min_dist:
                    min_dist = d
                    holder_id  = track.track_id
                    holder_team = track.team

            # Only count if player is actually close enough
            if min_dist > POSSESSION_MAX_DIST_PX:
                holder_id   = None
                holder_team = "unknown"

        if holder_team == "team_a":
            self._team_a_accum += 1
        elif holder_team == "team_b":
            self._team_b_accum += 1
        self._total_frames += 1

        # Compute cumulative percentages
        total = self._team_a_accum + self._team_b_accum
        if total > 0:
            raw_a = 100.0 * self._team_a_accum / total
            raw_b = 100.0 * self._team_b_accum / total
        else:
            raw_a = raw_b = 50.0

        # EMA smoothing for live display
        alpha = POSSESSION_SMOOTH_ALPHA
        self._ema_a = alpha * self._ema_a + (1 - alpha) * raw_a
        self._ema_b = alpha * self._ema_b + (1 - alpha) * raw_b

        return {
            "team_a_pct":   round(self._ema_a, 1),
            "team_b_pct":   round(self._ema_b, 1),
            "holder_id":    holder_id,
            "holder_team":  holder_team,
        }

    def reset(self) -> None:
        self._team_a_accum = 0.0
        self._team_b_accum = 0.0
        self._total_frames = 0
        self._ema_a = 50.0
        self._ema_b = 50.0
