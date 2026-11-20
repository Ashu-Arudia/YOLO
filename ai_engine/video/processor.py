"""
VideoProcessor
──────────────
Main pipeline orchestrator.

Wires together:
  FrameManager → PlayerDetector → TrackingManager
  → TeamClassifier → BallTracker → PossessionAnalyzer
  → Annotation renderer

Two modes:
  run_local()   — OpenCV window, press Q to quit
  frame_gen()   — generator yielding JPEG bytes (for MJPEG streaming)

Analytics are exposed via `self.latest_analytics` dict updated every frame.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Dict, Generator, Optional, Union

import cv2
import numpy as np

from ai_engine.config import (
    TEAM_A_COLOR, TEAM_B_COLOR, REF_COLOR, BALL_COLOR, UNKNOWN_COLOR,
    MJPEG_QUALITY, STREAM_FPS_MAX,
    CLASS_SPORTS_BALL, CLASS_PERSON,
)
from ai_engine.detection.player_detector import PlayerDetector
from ai_engine.tracking.tracking_manager import TrackingManager
from ai_engine.team_identification.team_classifier import TeamClassifier
from ai_engine.ball_tracking.ball_tracker import BallTracker
from ai_engine.analytics.possession_analyzer import PossessionAnalyzer
from ai_engine.video.frame_manager import FrameManager

logger = logging.getLogger(__name__)

TEAM_COLOR_MAP = {
    "team_a":  TEAM_A_COLOR,
    "team_b":  TEAM_B_COLOR,
    "referee": REF_COLOR,
    "unknown": UNKNOWN_COLOR,
}


class VideoProcessor:
    """
    Parameters
    ----------
    source : int | str
        0 for webcam, or path to video file.
    match_id : str
        Unique identifier attached to analytics events.
    """

    CALIBRATION_FRAMES = 40     # frames to collect before team classification

    def __init__(
        self,
        source: Union[int, str] = 0,
        match_id: str = "default",
        model_path: Optional[str] = None,
    ) -> None:
        self.source   = source
        self.match_id = match_id

        self._fm         = FrameManager(source)
        self._detector   = PlayerDetector(model_path=model_path)
        self._tracker    = TrackingManager()
        self._team_cls   = TeamClassifier()
        self._ball_trk   = BallTracker()
        self._possession = PossessionAnalyzer()

        self._running       = False
        self._frame_count   = 0
        self._fps_measured  = 0.0

        # Shared latest JPEG frame (for MJPEG)
        self._frame_lock    = threading.Lock()
        self._latest_jpeg   : Optional[bytes] = None

        # Analytics snapshot (thread-safe via lock)
        self._analytics_lock = threading.Lock()
        self.latest_analytics: Dict = {}

        # External stop signal
        self._stop_event = threading.Event()

    # ──────────────────────────────────────────────────── public interface

    def run_local(self) -> None:
        """
        Blocking: opens an OpenCV window and processes frames.
        Press Q to quit.
        """
        logger.info("Starting local OpenCV window — press Q to stop")
        self._running = True
        self._fm.open()
        t0 = time.perf_counter()

        try:
            for frame, meta in self._fm.frames():
                if self._stop_event.is_set():
                    break

                annotated = self._process_frame(frame, meta)

                # Overlay FPS
                cv2.putText(
                    annotated,
                    f"FPS: {self._fps_measured:.1f}  Frame: {self._frame_count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 255, 0), 2, cv2.LINE_AA,
                )

                cv2.imshow("SportsVision AI", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("User pressed Q — stopping")
                    break

                # Update FPS
                elapsed = time.perf_counter() - t0
                if elapsed > 0:
                    self._fps_measured = self._frame_count / elapsed

        finally:
            self._running = False
            self._fm.release()
            cv2.destroyAllWindows()
            logger.info("VideoProcessor stopped")

    def frame_gen(self) -> Generator[bytes, None, None]:
        """
        MJPEG generator — yields multipart JPEG byte chunks.
        Start in a background thread, then serve via FastAPI StreamingResponse.
        """
        min_interval = 1.0 / STREAM_FPS_MAX

        while self._running or not self._stop_event.is_set():
            t = time.perf_counter()
            with self._frame_lock:
                jpeg = self._latest_jpeg
            if jpeg:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                )
            # Rate-limit
            elapsed = time.perf_counter() - t
            sleep = min_interval - elapsed
            if sleep > 0:
                time.sleep(sleep)

    def start_background(self) -> None:
        """
        Launch processing in a background thread (for MJPEG / API mode).
        """
        self._stop_event.clear()
        t = threading.Thread(target=self._background_loop, daemon=True)
        t.start()
        logger.info("VideoProcessor background thread started")

    def stop(self) -> None:
        self._stop_event.set()
        self._running = False
        logger.info("VideoProcessor stop requested")

    # ──────────────────────────────────────────────────── private

    def _background_loop(self) -> None:
        self._running = True
        self._fm.open()
        t0 = time.perf_counter()

        try:
            for frame, meta in self._fm.frames():
                if self._stop_event.is_set():
                    break

                annotated = self._process_frame(frame, meta)

                # Overlay FPS
                cv2.putText(
                    annotated,
                    f"FPS: {self._fps_measured:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0), 2, cv2.LINE_AA,
                )

                # Encode to JPEG
                ok, buf = cv2.imencode(
                    ".jpg", annotated,
                    [cv2.IMWRITE_JPEG_QUALITY, MJPEG_QUALITY]
                )
                if ok:
                    with self._frame_lock:
                        self._latest_jpeg = buf.tobytes()

                elapsed = time.perf_counter() - t0
                if elapsed > 0:
                    self._fps_measured = self._frame_count / elapsed

        finally:
            self._running = False
            self._fm.release()
            logger.info("Background processing loop finished")

    def _process_frame(self, frame: np.ndarray, meta: dict) -> np.ndarray:
        """Core per-frame pipeline."""
        self._frame_count += 1
        fi = self._frame_count

        # ── 1. Detect + track ───────────────────────────────────────────
        results = self._detector.track(frame)
        registry = self._tracker.update(results)

        # ── 2. Team classification ──────────────────────────────────────
        boxes = results.boxes
        if boxes is not None and len(boxes) > 0:
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            xyxy    = boxes.xyxy.cpu().numpy()
            ids     = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else []

            # Feed jersey colours during calibration phase
            player_boxes = xyxy[cls_ids == CLASS_PERSON]
            if fi <= self.CALIBRATION_FRAMES:
                self._team_cls.feed(frame, player_boxes)
            else:
                for pid, box, cls in zip(ids, xyxy, cls_ids):
                    if cls == CLASS_PERSON:
                        team = self._team_cls.classify(pid, frame, box, fi)
                        self._tracker.set_team(int(pid), team)

        # ── 3. Ball tracking ────────────────────────────────────────────
        ball_pos = self._ball_trk.update(results)

        # ── 4. Possession ───────────────────────────────────────────────
        poss = self._possession.update(
            ball_pos, self._tracker.player_tracks
        )

        # ── 5. Build analytics snapshot ─────────────────────────────────
        active = self._tracker.active_tracks
        n_team_a = sum(1 for t in active if t.team == "team_a")
        n_team_b = sum(1 for t in active if t.team == "team_b")

        analytics = {
            "frame_idx":  fi,
            "fps":        round(self._fps_measured, 1),
            "match_id":   self.match_id,
            "player_count": len(self._tracker.player_tracks),
            "team_a_count": n_team_a,
            "team_b_count": n_team_b,
            "possession": poss,
            "ball_pos":   ball_pos,
            "calibrated": self._team_cls._calibrated,
        }
        with self._analytics_lock:
            self.latest_analytics = analytics

        # ── 6. Annotate frame ───────────────────────────────────────────
        annotated = self._annotate(frame.copy(), results, ball_pos, poss)
        return annotated

    # ──────────────────────────────────────────────────── annotation

    def _annotate(
        self,
        frame: np.ndarray,
        results,
        ball_pos,
        poss: dict,
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        boxes = results.boxes

        if boxes is None:
            return frame

        ids      = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else []
        cls_ids  = boxes.cls.cpu().numpy().astype(int)
        xyxy     = boxes.xyxy.cpu().numpy()
        confs    = boxes.conf.cpu().numpy()

        for idx, (tid, box, cls, conf) in enumerate(
            zip(ids if len(ids) else range(len(xyxy)), xyxy, cls_ids, confs)
        ):
            x1, y1, x2, y2 = map(int, box)

            if cls == CLASS_SPORTS_BALL:
                color = BALL_COLOR
                label = f"Ball {conf:.2f}"
            else:
                team  = self._tracker._registry.get(int(tid), None)
                team_name = team.team if team else "unknown"
                color = TEAM_COLOR_MAP.get(team_name, UNKNOWN_COLOR)
                label = f"#{tid} {team_name} {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # Label bg
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA,
            )

        # Ball trail
        trail = self._ball_trk.trail
        for i in range(1, len(trail)):
            p1 = (int(trail[i - 1][0]), int(trail[i - 1][1]))
            p2 = (int(trail[i][0]),     int(trail[i][1]))
            alpha = i / len(trail)
            radius = max(1, int(4 * alpha))
            cv2.circle(frame, p2, radius, BALL_COLOR, -1)

        # Possession bar
        self._draw_possession_bar(frame, w, poss)

        return frame

    @staticmethod
    def _draw_possession_bar(
        frame: np.ndarray,
        w: int,
        poss: dict,
    ) -> None:
        bar_h  = 28
        pct_a  = poss.get("team_a_pct", 50)
        pct_b  = poss.get("team_b_pct", 50)
        split  = int(w * pct_a / 100)

        cv2.rectangle(frame, (0, 0), (split, bar_h), TEAM_A_COLOR, -1)
        cv2.rectangle(frame, (split, 0), (w, bar_h), TEAM_B_COLOR, -1)

        label_a = f"Team A {pct_a:.0f}%"
        label_b = f"Team B {pct_b:.0f}%"
        cv2.putText(
            frame, label_a, (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA,
        )
        cv2.putText(
            frame, label_b, (w - 130, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA,
        )
