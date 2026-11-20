"""
PitchMapper
───────────
Generates a clean top-down 2D minimap of the football pitch with player
dots and ball position overlaid.

Uses normalised pitch coordinates (0-1) produced by PitchDetector to
position markers on a synthetic bird's-eye-view pitch diagram.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ai_engine.config import (
    PITCH_MINIMAP_W,
    PITCH_MINIMAP_H,
    TEAM_A_COLOR,
    TEAM_B_COLOR,
    UNKNOWN_COLOR,
    BALL_COLOR,
)

logger = logging.getLogger(__name__)

# Type aliases
Point2D = Tuple[float, float]

# ── Pitch drawing constants ──────────────────────────────────────
_PITCH_GREEN  = (34, 139, 34)     # BGR — a natural grass green
_LINE_COLOR   = (255, 255, 255)   # white
_LINE_THICK   = 1
_MARGIN       = 10                # px margin inside minimap edges

# Standard pitch ratios (FIFA: 105 m × 68 m)
_PITCH_RATIO  = 105 / 68          # ≈ 1.544

# Relative dimensions (fraction of pitch length / width)
_PENALTY_W    = 16.5 / 105        # penalty area width (from goal line)
_PENALTY_H    = 40.3 / 68         # penalty area height (centred)
_GOAL_AREA_W  = 5.5  / 105
_GOAL_AREA_H  = 18.3 / 68
_CENTRE_R     = 9.15 / 105        # centre circle radius (relative to width)
_PENALTY_SPOT = 11.0 / 105        # penalty spot distance from goal line

# Team colour map
_TEAM_COLORS = {
    "team_a":  TEAM_A_COLOR,
    "team_b":  TEAM_B_COLOR,
    "unknown": UNKNOWN_COLOR,
    "referee": (30, 210, 30),      # green for referee
}


class PitchMapper:
    """
    Renders a top-down 2D minimap of the pitch with tracked entities.

    Usage
    -----
    mapper = PitchMapper()
    img = mapper.render_minimap(player_tracks, ball_pos)
    """

    def __init__(self) -> None:
        self._base_cache: dict[Tuple[int, int], np.ndarray] = {}

    # ─────────────────────────────────────────────────────────────── public API

    def render_minimap(
        self,
        player_tracks: List,
        ball_pos: Optional[Point2D] = None,
        width: int = PITCH_MINIMAP_W,
        height: int = PITCH_MINIMAP_H,
    ) -> np.ndarray:
        """
        Return a BGR numpy image of size (height, width, 3) showing a
        top-down football pitch with player and ball markers.

        Parameters
        ----------
        player_tracks:
            Iterable of objects with `.centroid` (or a (team, (nx, ny)) tuple)
            and `.team` attributes.  Coordinates must be normalised [0, 1].
        ball_pos:
            Normalised (nx, ny) position of the ball, or None.
        width, height:
            Output image dimensions in pixels.
        """
        img = self._get_base_pitch(width, height).copy()

        # Drawable area (inside margins)
        x0, y0 = _MARGIN, _MARGIN
        pw, ph = width - 2 * _MARGIN, height - 2 * _MARGIN

        # Draw players
        for track in player_tracks:
            pos = self._extract_position(track)
            team = self._extract_team(track)
            if pos is None:
                continue
            nx, ny = pos
            sx = int(x0 + nx * pw)
            sy = int(y0 + ny * ph)
            colour = _TEAM_COLORS.get(team, UNKNOWN_COLOR)
            cv2.circle(img, (sx, sy), 5, colour, -1)
            cv2.circle(img, (sx, sy), 5, _LINE_COLOR, 1)

        # Draw ball
        if ball_pos is not None:
            bx = int(x0 + ball_pos[0] * pw)
            by = int(y0 + ball_pos[1] * ph)
            cv2.circle(img, (bx, by), 4, BALL_COLOR, -1)
            cv2.circle(img, (bx, by), 6, _LINE_COLOR, 1)

        return img

    def reset(self) -> None:
        """Clear the base-pitch image cache."""
        self._base_cache.clear()

    # ─────────────────────────────────────────────────────────────── private

    def _get_base_pitch(self, w: int, h: int) -> np.ndarray:
        """Return (and cache) the blank pitch diagram for the given size."""
        key = (w, h)
        if key not in self._base_cache:
            self._base_cache[key] = self._draw_pitch(w, h)
        return self._base_cache[key]

    @staticmethod
    def _draw_pitch(w: int, h: int) -> np.ndarray:
        """
        Draw a simple top-down football pitch diagram:
        outer boundary, halfway line, centre circle, two penalty areas,
        two goal areas.
        """
        img = np.full((h, w, 3), _PITCH_GREEN, dtype=np.uint8)

        x0, y0 = _MARGIN, _MARGIN
        pw, ph = w - 2 * _MARGIN, h - 2 * _MARGIN

        # Outer boundary
        cv2.rectangle(img, (x0, y0), (x0 + pw, y0 + ph), _LINE_COLOR, _LINE_THICK)

        # Halfway line (vertical centre)
        mid_x = x0 + pw // 2
        cv2.line(img, (mid_x, y0), (mid_x, y0 + ph), _LINE_COLOR, _LINE_THICK)

        # Centre circle
        cr = int(min(_CENTRE_R * pw, ph * 0.25))
        cv2.circle(img, (mid_x, y0 + ph // 2), cr, _LINE_COLOR, _LINE_THICK)

        # Centre spot
        cv2.circle(img, (mid_x, y0 + ph // 2), 2, _LINE_COLOR, -1)

        # ── Left penalty area ────────────────────────────────────
        pa_w = int(_PENALTY_W * pw)
        pa_h = int(_PENALTY_H * ph)
        pa_y = y0 + (ph - pa_h) // 2
        cv2.rectangle(img, (x0, pa_y), (x0 + pa_w, pa_y + pa_h),
                       _LINE_COLOR, _LINE_THICK)

        # Left goal area
        ga_w = int(_GOAL_AREA_W * pw)
        ga_h = int(_GOAL_AREA_H * ph)
        ga_y = y0 + (ph - ga_h) // 2
        cv2.rectangle(img, (x0, ga_y), (x0 + ga_w, ga_y + ga_h),
                       _LINE_COLOR, _LINE_THICK)

        # Left penalty spot
        ps_x = x0 + int(_PENALTY_SPOT * pw)
        cv2.circle(img, (ps_x, y0 + ph // 2), 2, _LINE_COLOR, -1)

        # ── Right penalty area ───────────────────────────────────
        r_pa_x = x0 + pw - pa_w
        cv2.rectangle(img, (r_pa_x, pa_y), (x0 + pw, pa_y + pa_h),
                       _LINE_COLOR, _LINE_THICK)

        # Right goal area
        r_ga_x = x0 + pw - ga_w
        cv2.rectangle(img, (r_ga_x, ga_y), (x0 + pw, ga_y + ga_h),
                       _LINE_COLOR, _LINE_THICK)

        # Right penalty spot
        rps_x = x0 + pw - int(_PENALTY_SPOT * pw)
        cv2.circle(img, (rps_x, y0 + ph // 2), 2, _LINE_COLOR, -1)

        return img

    @staticmethod
    def _extract_position(track) -> Optional[Point2D]:
        """
        Flexibly extract normalised (nx, ny) from a track object.
        Supports TrackInfo objects (with .centroid) and plain tuples.
        """
        # TrackInfo-style object
        if hasattr(track, "pitch_pos"):
            return track.pitch_pos  # type: ignore[return-value]
        if hasattr(track, "centroid") and track.centroid is not None:
            return track.centroid  # type: ignore[return-value]
        # (team, (nx, ny)) tuple
        if isinstance(track, (tuple, list)) and len(track) == 2:
            _, pos = track
            if isinstance(pos, (tuple, list)) and len(pos) == 2:
                return (float(pos[0]), float(pos[1]))
        return None

    @staticmethod
    def _extract_team(track) -> str:
        """Extract team label from a track object or tuple."""
        if hasattr(track, "team"):
            return track.team  # type: ignore[return-value]
        if isinstance(track, (tuple, list)) and len(track) >= 1:
            return str(track[0])
        return "unknown"
