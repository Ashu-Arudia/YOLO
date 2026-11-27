"""
TeamClassifier
──────────────
Assigns players to Team A or Team B by clustering jersey colours.

Algorithm
---------
1. Crop the upper half of each player bounding box (jersey region).
2. Convert to HSV and flat-sample pixels (ignoring green grass).
3. Run K-Means (k=2) globally to establish two jersey colour clusters.
4. Per player: assign to the nearest cluster centroid.
5. Cache assignment per track_id for TEAM_CACHE_TTL frames.

Limitations
-----------
- Works best when jersey colours differ clearly in hue.
- Goal-keeper / referee classification is not automatic in Phase 1;
  they are treated as "player" unless manual labels are provided.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from sklearn.cluster import KMeans

from ai_engine.config import TEAM_CLUSTERS, TEAM_CACHE_TTL

logger = logging.getLogger(__name__)

TeamLabel = str   # "team_a" | "team_b" | "unknown"


class TeamClassifier:
    """
    Usage
    -----
    classifier = TeamClassifier()

    # Feed enough frames first so clustering can calibrate:
    for frame, detections in first_frames:
        classifier.feed(frame, detections)   # detection boxes

    # Then classify every frame:
    team = classifier.classify(track_id, frame, bbox)
    """

    def __init__(self) -> None:
        self._kmeans: Optional[KMeans] = None
        self._cluster_labels: Dict[int, TeamLabel] = {}   # cluster_idx → label
        self._cache: Dict[int, Tuple[int, TeamLabel]] = {}  # track_id → (frame_last, label)
        self._calibrated: bool = False
        self._sample_colors: List[np.ndarray] = []         # accumulated for fitting

    # ─────────────────────────────────────────────────────────────── public API

    def feed(self, frame: np.ndarray, boxes_xyxy: np.ndarray) -> None:
        """
        Collect jersey-colour samples from a batch of bounding boxes.
        Call this on the first ~30 frames before calling classify().
        """
        for box in boxes_xyxy:
            color = self._extract_jersey_color(frame, box)
            if color is not None:
                self._sample_colors.append(color)

        if len(self._sample_colors) >= 60 and not self._calibrated:
            self._fit()

    def classify(
        self,
        track_id: int,
        frame: np.ndarray,
        bbox: np.ndarray,
        frame_idx: int = 0,
    ) -> TeamLabel:
        """Return team label for this track_id / bbox."""
        # Check cache
        if track_id in self._cache:
            last_frame, label = self._cache[track_id]
            if frame_idx - last_frame < TEAM_CACHE_TTL:
                return label

        if not self._calibrated:
            return "unknown"

        color = self._extract_jersey_color(frame, bbox)
        if color is None:
            return "unknown"

        label = self._predict_team(color)
        self._cache[track_id] = (frame_idx, label)
        return label

    def reset(self) -> None:
        self._kmeans = None
        self._cluster_labels = {}
        self._cache = {}
        self._calibrated = False
        self._sample_colors = []

    # ─────────────────────────────────────────────────────────────── private

    def _extract_jersey_color(
        self,
        frame: np.ndarray,
        box: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Crop top ~40% of bounding box, mask out green grass pixels,
        return mean HSV colour of remaining pixels.
        """
        x1, y1, x2, y2 = map(int, box[:4])
        h = y2 - y1
        w = x2 - x1

        # Sanity check
        if h < 20 or w < 10:
            return None

        # Take upper 40% (torso/jersey area)
        y_bot = y1 + int(h * 0.40)
        crop = frame[y1:y_bot, x1:x2]
        if crop.size == 0:
            return None

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        # Mask green grass: H≈35–85, S>40, V>40
        green_lo = np.array([25,  40, 40], dtype=np.uint8)
        green_hi = np.array([90, 255, 255], dtype=np.uint8)
        grass_mask = cv2.inRange(hsv, green_lo, green_hi)
        jersey_mask = cv2.bitwise_not(grass_mask)

        pixels = hsv[jersey_mask > 0]
        if len(pixels) < 10:
            # Fallback: use all pixels
            pixels = hsv.reshape(-1, 3)

        return pixels.mean(axis=0)  # [H, S, V]

    def _fit(self) -> None:
        """Fit K-Means on accumulated colour samples."""
        data = np.array(self._sample_colors, dtype=np.float64)
        n_clusters = min(TEAM_CLUSTERS, len(data))
        self._kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        self._kmeans.fit(data)

        # Assign human-readable labels based on cluster index
        # Cluster 0 → team_a, Cluster 1 → team_b
        # (teams can be relabelled later via UI)
        self._cluster_labels = {i: f"team_{chr(ord('a') + i)}"
                                 for i in range(n_clusters)}
        self._calibrated = True
        logger.info(
            "TeamClassifier calibrated — %d clusters, centroids: %s",
            n_clusters,
            self._kmeans.cluster_centers_.tolist(),
        )

    def _predict_team(self, color: np.ndarray) -> TeamLabel:
        assert self._kmeans is not None
        c = np.asarray(color, dtype=np.float64).reshape(1, -1)
        label_idx = int(self._kmeans.predict(c)[0])
        return self._cluster_labels.get(label_idx, "unknown")
