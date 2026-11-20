"""
FrameManager
────────────
Handles video capture, optional resizing, and frame-skipping.
Provides a clean generator interface over OpenCV VideoCapture.
"""
from __future__ import annotations

import logging
from typing import Generator, Optional, Tuple, Union

import cv2
import numpy as np

from ai_engine.config import FRAME_SKIP, MAX_FRAME_DIM

logger = logging.getLogger(__name__)

FrameShape = Tuple[int, int, int]   # H, W, C


class FrameManager:
    """
    Usage
    -----
    fm = FrameManager(source=0)         # webcam
    fm = FrameManager(source="path.mp4")

    for frame, meta in fm.frames():
        # frame: BGR numpy array
        # meta:  dict with fps, frame_idx, width, height
        ...

    fm.release()
    """

    def __init__(
        self,
        source: Union[int, str] = 0,
        frame_skip: int = FRAME_SKIP,
        max_dim: int = MAX_FRAME_DIM,
    ) -> None:
        self.source     = source
        self.frame_skip = frame_skip
        self.max_dim    = max_dim
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"FrameManager: cannot open source '{self.source}'"
            )
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        w   = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(
            "Opened source '%s' — %dx%d @ %.1f fps", self.source, w, h, fps
        )

    def frames(self) -> Generator[Tuple[np.ndarray, dict], None, None]:
        if self._cap is None:
            self.open()
        assert self._cap is not None

        frame_idx = 0
        fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        w   = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        while True:
            ret, frame = self._cap.read()
            if not ret:
                logger.info("End of stream at frame %d", frame_idx)
                break

            frame_idx += 1

            # Frame-skip
            if self.frame_skip > 0 and (frame_idx % (self.frame_skip + 1) != 0):
                continue

            # Resize if too large
            frame = self._maybe_resize(frame)

            meta = {
                "frame_idx": frame_idx,
                "source_fps": fps,
                "source_w":  w,
                "source_h":  h,
            }
            yield frame, meta

    def release(self) -> None:
        if self._cap and self._cap.isOpened():
            self._cap.release()
            logger.info("FrameManager: capture released")

    def __del__(self) -> None:
        self.release()

    # ─────────────────────────────────────────────────────────────── private

    def _maybe_resize(self, frame: np.ndarray) -> np.ndarray:
        if self.max_dim <= 0:
            return frame
        h, w = frame.shape[:2]
        if max(h, w) <= self.max_dim:
            return frame
        scale = self.max_dim / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
