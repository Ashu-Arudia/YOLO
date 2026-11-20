"""
TrackingManager
───────────────
Maintains a per-track-ID registry that survives across frames.
Stores team assignment, last bbox centroid, last-seen frame index,
and estimated speed / distance (pixel-space).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TrackInfo:
    track_id:    int
    team:        str  = "unknown"   # "team_a" | "team_b" | "referee" | "unknown"
    class_name:  str  = "player"    # "player" | "ball"
    centroid:    Optional[Tuple[float, float]] = None
    prev_centroid: Optional[Tuple[float, float]] = None
    last_seen:   int  = 0           # frame index
    distance_px: float = 0.0        # cumulative pixel distance
    speed_px:    float = 0.0        # pixels / frame (last step)
    # History for heatmap / trail
    centroid_history: List[Tuple[float, float]] = field(default_factory=list)

    MAX_HISTORY = 300               # keep last N centroids

    def update_position(self, cx: float, cy: float, frame_idx: int) -> None:
        if self.centroid:
            dx = cx - self.centroid[0]
            dy = cy - self.centroid[1]
            step = float(np.hypot(dx, dy))
            self.distance_px += step
            self.speed_px = step
            self.prev_centroid = self.centroid
        self.centroid = (cx, cy)
        self.last_seen = frame_idx
        self.centroid_history.append((cx, cy))
        if len(self.centroid_history) > self.MAX_HISTORY:
            self.centroid_history.pop(0)


class TrackingManager:
    """
    Call `update()` every frame with the YOLO Results object.
    Query `active_tracks` to get all tracks seen in the last N frames.
    """

    EXPIRE_AFTER_FRAMES = 90    # remove track if unseen for this many frames

    def __init__(self) -> None:
        self._registry: Dict[int, TrackInfo] = {}
        self._frame_idx: int = 0

    # ─────────────────────────────────────────────────────────────── public API

    def update(self, results) -> Dict[int, TrackInfo]:
        """
        Parse Ultralytics Results and update registry.
        Returns the current registry snapshot.
        """
        self._frame_idx += 1
        boxes = results.boxes

        if boxes is None or boxes.id is None:
            self._expire_old()
            return self._registry

        ids      = boxes.id.cpu().numpy().astype(int)
        xyxy     = boxes.xyxy.cpu().numpy()
        cls_ids  = boxes.cls.cpu().numpy().astype(int)

        for tid, box, cls in zip(ids, xyxy, cls_ids):
            cx = float((box[0] + box[2]) / 2)
            cy = float((box[1] + box[3]) / 2)
            cls_name = "ball" if cls == 32 else "player"

            if tid not in self._registry:
                self._registry[tid] = TrackInfo(
                    track_id=tid, class_name=cls_name
                )

            info = self._registry[tid]
            info.class_name = cls_name
            info.update_position(cx, cy, self._frame_idx)

        self._expire_old()
        return self._registry

    def set_team(self, track_id: int, team: str) -> None:
        if track_id in self._registry:
            self._registry[track_id].team = team

    @property
    def active_tracks(self) -> List[TrackInfo]:
        cutoff = self._frame_idx - self.EXPIRE_AFTER_FRAMES
        return [t for t in self._registry.values() if t.last_seen >= cutoff]

    @property
    def player_tracks(self) -> List[TrackInfo]:
        return [t for t in self.active_tracks if t.class_name == "player"]

    @property
    def ball_tracks(self) -> List[TrackInfo]:
        return [t for t in self.active_tracks if t.class_name == "ball"]

    def reset(self) -> None:
        self._registry.clear()
        self._frame_idx = 0

    # ─────────────────────────────────────────────────────────────── private

    def _expire_old(self) -> None:
        cutoff = self._frame_idx - self.EXPIRE_AFTER_FRAMES
        expired = [tid for tid, t in self._registry.items()
                   if t.last_seen < cutoff]
        for tid in expired:
            del self._registry[tid]
