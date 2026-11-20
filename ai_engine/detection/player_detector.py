"""
PlayerDetector
──────────────
Wraps Ultralytics YOLO with ByteTrack for person + ball detection/tracking.
Designed so the underlying model can be swapped without changing the rest of
the pipeline — just pass a different model_path.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results

from ai_engine.config import (
    YOLO_MODEL,
    DETECT_CONF,
    DETECT_IOU,
    DETECT_IMGSZ,
    DETECT_CLASSES,
    TRACKER_CONFIG,
    MODELS_DIR,
)

logger = logging.getLogger(__name__)


def _resolve_device() -> str:
    """Choose cuda, mps, or cpu automatically."""
    if torch.cuda.is_available():
        logger.info("CUDA available — using GPU: %s", torch.cuda.get_device_name(0))
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("MPS available — using Apple Silicon GPU")
        return "mps"
    logger.info("No GPU detected — using CPU")
    return "cpu"


class PlayerDetector:
    """
    Stateful YOLO + ByteTrack detector.

    Usage
    -----
    detector = PlayerDetector()
    results  = detector.track(frame)          # numpy BGR frame
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        self.device = device or _resolve_device()

        # Check local models dir first, then let ultralytics download
        mp = model_path or YOLO_MODEL
        local = MODELS_DIR / mp
        resolved = str(local) if local.exists() else mp

        logger.info("Loading YOLO model: %s on %s", resolved, self.device)
        self.model = YOLO(resolved)
        self.model.to(self.device)

        self.conf       = DETECT_CONF
        self.iou        = DETECT_IOU
        self.imgsz      = DETECT_IMGSZ
        self.classes    = DETECT_CLASSES
        self.tracker    = TRACKER_CONFIG

        logger.info(
            "PlayerDetector ready — conf=%.2f  iou=%.2f  classes=%s",
            self.conf, self.iou, self.classes,
        )

    # ─────────────────────────────────────────────────────────────── public API

    def track(self, frame) -> Results:
        """
        Run detection + ByteTrack on a single BGR numpy frame.
        Returns the first (and only) Results object.
        persist=True keeps track state between calls.
        """
        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker,
            classes=self.classes,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        return results[0]

    def detect(self, frame) -> Results:
        """
        Run detection only (no tracking) — useful for single-frame inference.
        """
        results = self.model.predict(
            source=frame,
            classes=self.classes,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        return results[0]

    def reset(self) -> None:
        """Reset ByteTrack state — call at the start of a new video."""
        # Re-create model to clear tracker state
        self.__init__(device=self.device)
