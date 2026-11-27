"""
BallTracker
───────────
Dedicated ball detection and trajectory tracking.

- Filters YOLO results for class 32 (sports ball).
- Applies a simple Kalman filter for smooth position estimation.
- Stores the last N positions for trail visualisation.
- Handles missing detections with predicted state.
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np

from ai_engine.config import BALL_TRAIL_LEN, CLASS_SPORTS_BALL

logger = logging.getLogger(__name__)

BallPos = Tuple[float, float]   # (cx, cy)


class _BallKalman:
    """
    Simple constant-velocity Kalman filter for 2-D ball position.
    State: [x, y, vx, vy]
    """

    def __init__(self) -> None:
        kf = cv2.KalmanFilter(4, 2)
        kf.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]], dtype=np.float32
        )
        kf.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], dtype=np.float32
        )
        kf.processNoiseCov  = np.eye(4, dtype=np.float32) * 0.03
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0
        kf.errorCovPost     = np.eye(4, dtype=np.float32) * 0.1
        self._kf = kf
        self._initialised = False

    def update(self, cx: float, cy: float) -> BallPos:
        meas = np.array([[np.float32(cx)], [np.float32(cy)]])
        if not self._initialised:
            self._kf.statePre = np.array(
                [[cx], [cy], [0.0], [0.0]], dtype=np.float32
            )
            self._initialised = True
        self._kf.correct(meas)
        pred = self._kf.predict().flatten()
        return float(pred[0]), float(pred[1])

    def predict(self) -> BallPos:
        pred = self._kf.predict().flatten()
        return float(pred[0]), float(pred[1])

    def reset(self) -> None:
        self._initialised = False


class BallTracker:
    """
    Usage
    -----
    bt = BallTracker()
    pos = bt.update(results)     # Ultralytics Results for one frame
    trail = bt.trail             # list of (cx, cy) tuples
    """

    def __init__(self, trail_len: int = BALL_TRAIL_LEN) -> None:
        self._kalman   = _BallKalman()
        self._trail: Deque[BallPos] = deque(maxlen=trail_len)
        self._position: Optional[BallPos] = None
        self._missing_frames: int = 0
        self.MAX_MISSING = 15       # after this many frames predict instead

    # ─────────────────────────────────────────────────────────────── public API

    def update(self, results) -> Optional[BallPos]:
        """
        Extract ball position from YOLO Results and update Kalman filter.
        Returns smoothed (cx, cy) or None if ball has been missing too long.
        """
        detected_pos = self._extract_best_ball(results)

        if detected_pos is not None:
            self._missing_frames = 0
            cx, cy = detected_pos
            smoothed = self._kalman.update(cx, cy)
        else:
            self._missing_frames += 1
            if self._missing_frames <= self.MAX_MISSING:
                smoothed = self._kalman.predict()
            else:
                self._position = None
                return None

        self._position = smoothed
        self._trail.append(smoothed)
        return smoothed

    @property
    def position(self) -> Optional[BallPos]:
        return self._position

    @property
    def trail(self) -> List[BallPos]:
        return list(self._trail)

    def reset(self) -> None:
        self._kalman.reset()
        self._trail.clear()
        self._position = None
        self._missing_frames = 0

    # ─────────────────────────────────────────────────────────────── private

    @staticmethod
    def _extract_best_ball(results) -> Optional[BallPos]:
        """
        Return centroid of the highest-confidence ball detection,
        or None if no ball detected.
        """
        boxes = results.boxes
        if boxes is None:
            return None

        cls_ids = boxes.cls.cpu().numpy().astype(int)
        confs   = boxes.conf.cpu().numpy()
        xyxy    = boxes.xyxy.cpu().numpy()

        best_conf = -1.0
        best_pos  = None

        for cls, conf, box in zip(cls_ids, confs, xyxy):
            if cls == CLASS_SPORTS_BALL and conf > best_conf:
                best_conf = float(conf)
                cx = float((box[0] + box[2]) / 2)
                cy = float((box[1] + box[3]) / 2)
                best_pos = (cx, cy)

        return best_pos
