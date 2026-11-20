"""
PitchDetector
─────────────
Detects the football pitch boundary and key lines using computer vision.

- Segments the green playing field via HSV thresholding.
- Finds pitch lines with probabilistic Hough transform.
- Estimates a basic homography from detected line intersections
  to a normalised [0, 1] × [0, 1] pitch coordinate system.
- Falls back to an identity-style transform when no reliable
  geometry can be recovered.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ai_engine.config import (
    PITCH_GREEN_LO,
    PITCH_GREEN_HI,
    HOUGH_THRESHOLD,
    HOUGH_MIN_LINE_LEN,
    HOUGH_MAX_LINE_GAP,
)

logger = logging.getLogger(__name__)

Point2D = Tuple[float, float]


class PitchDetector:
    """
    Stateful pitch-boundary and line detector.

    Usage
    -----
    pd = PitchDetector()
    pd.update(frame)                          # BGR numpy frame
    nx, ny = pd.transform_point(px, py)       # pixel → normalised
    on     = pd.is_on_pitch(px, py)           # inside field mask?
    """

    def __init__(self) -> None:
        self._mask: Optional[np.ndarray] = None          # binary field mask
        self._homography: Optional[np.ndarray] = None    # 3×3 pixel→norm
        self._frame_h: int = 0
        self._frame_w: int = 0
        self._field_contour: Optional[np.ndarray] = None
        self._lines: List[Tuple[int, int, int, int]] = []

    # ─────────────────────────────────────────────────────────────── public API

    def update(self, frame: np.ndarray) -> None:
        """
        Analyse *frame* to detect the pitch area and estimate a homography.
        Call once per frame (or less often — the results are cached).
        """
        self._frame_h, self._frame_w = frame.shape[:2]

        # 1. Green-area segmentation
        self._mask = self._segment_field(frame)

        # 2. Largest contour = playing field boundary
        self._field_contour = self._largest_contour(self._mask)

        # 3. Hough lines on the masked edge image
        self._lines = self._detect_lines(frame, self._mask)

        # 4. Try to estimate homography from field corners
        self._homography = self._estimate_homography()

    def transform_point(self, px: float, py: float) -> Point2D:
        """
        Map a pixel coordinate to normalised pitch coordinates in [0, 1].
        Returns (nx, ny).  Falls back to a simple bounding-box normalisation
        when no homography could be computed.
        """
        if self._homography is not None:
            src = np.array([[[px, py]]], dtype=np.float64)
            dst = cv2.perspectiveTransform(src, self._homography)
            nx = float(np.clip(dst[0, 0, 0], 0.0, 1.0))
            ny = float(np.clip(dst[0, 0, 1], 0.0, 1.0))
            return nx, ny

        # Fallback: normalise using detected field bounding rect
        return self._fallback_transform(px, py)

    def is_on_pitch(self, px: float, py: float) -> bool:
        """Return True if the pixel coordinate falls inside the detected field."""
        if self._mask is None:
            return True  # assume on pitch when no analysis has run

        ix, iy = int(round(px)), int(round(py))
        if 0 <= iy < self._mask.shape[0] and 0 <= ix < self._mask.shape[1]:
            return bool(self._mask[iy, ix] > 0)
        return False

    @property
    def field_mask(self) -> Optional[np.ndarray]:
        """Current binary field mask (255 = pitch, 0 = off-pitch)."""
        return self._mask

    @property
    def detected_lines(self) -> List[Tuple[int, int, int, int]]:
        """List of (x1, y1, x2, y2) line segments found on pitch."""
        return self._lines

    def reset(self) -> None:
        """Clear all cached state."""
        self._mask = None
        self._homography = None
        self._field_contour = None
        self._lines = []

    # ─────────────────────────────────────────────────────────────── private

    @staticmethod
    def _segment_field(frame: np.ndarray) -> np.ndarray:
        """HSV threshold → morphology-cleaned binary mask of the green field."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lo = np.array(PITCH_GREEN_LO, dtype=np.uint8)
        hi = np.array(PITCH_GREEN_HI, dtype=np.uint8)
        mask = cv2.inRange(hsv, lo, hi)

        # Clean up with morphological ops
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    @staticmethod
    def _largest_contour(mask: np.ndarray) -> Optional[np.ndarray]:
        """Return the largest contour in *mask*, or None."""
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            return None
        return max(contours, key=cv2.contourArea)

    @staticmethod
    def _detect_lines(
        frame: np.ndarray, mask: np.ndarray,
    ) -> List[Tuple[int, int, int, int]]:
        """Hough-P line detection restricted to the pitch area."""
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        masked_grey = cv2.bitwise_and(grey, grey, mask=mask)
        edges = cv2.Canny(masked_grey, 50, 150, apertureSize=3)

        raw = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=HOUGH_THRESHOLD,
            minLineLength=HOUGH_MIN_LINE_LEN,
            maxLineGap=HOUGH_MAX_LINE_GAP,
        )
        if raw is None:
            return []
        return [(int(l[0]), int(l[1]), int(l[2]), int(l[3])) for l in raw[:, 0]]

    def _estimate_homography(self) -> Optional[np.ndarray]:
        """
        Estimate a perspective transform from the four corners of the
        detected field contour to a [0, 1] × [0, 1] rectangle.

        Falls back to None if fewer than 4 reasonable corner points
        can be derived from the contour.
        """
        if self._field_contour is None:
            return None

        # Approximate the contour to a polygon
        peri = cv2.arcLength(self._field_contour, closed=True)
        approx = cv2.approxPolyDP(self._field_contour, 0.02 * peri, closed=True)

        if len(approx) < 4:
            logger.debug(
                "Contour approximation yielded %d points — need ≥ 4; "
                "falling back to bounding-box transform.",
                len(approx),
            )
            return None

        # Use the bounding rect corners as the 4 source points
        # This is more robust than the polygon vertices themselves
        corners = self._order_corners(approx)
        if corners is None:
            return None

        src_pts = corners.astype(np.float32)
        dst_pts = np.array(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            dtype=np.float32,
        )

        H, status = cv2.findHomography(src_pts, dst_pts)
        if H is None:
            logger.debug("findHomography returned None — using fallback.")
            return None

        return H

    @staticmethod
    def _order_corners(poly: np.ndarray) -> Optional[np.ndarray]:
        """
        Given an Nx1x2 polygon from approxPolyDP, pick the 4 extreme
        corners and order them [top-left, top-right, bottom-right, bottom-left].
        """
        pts = poly.reshape(-1, 2).astype(np.float32)
        if len(pts) < 4:
            return None

        # Sum and diff method for corner ordering
        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1).flatten()

        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmin(d)]
        bl = pts[np.argmax(d)]

        return np.array([tl, tr, br, bl], dtype=np.float32)

    def _fallback_transform(self, px: float, py: float) -> Point2D:
        """
        Simple bounding-box normalisation when no homography is available.
        Uses the field contour bbox, or the whole frame as a last resort.
        """
        if self._field_contour is not None:
            x, y, w, h = cv2.boundingRect(self._field_contour)
            if w > 0 and h > 0:
                nx = float(np.clip((px - x) / w, 0.0, 1.0))
                ny = float(np.clip((py - y) / h, 0.0, 1.0))
                return nx, ny

        # Last resort: use full frame dimensions
        if self._frame_w > 0 and self._frame_h > 0:
            return (
                float(np.clip(px / self._frame_w, 0.0, 1.0)),
                float(np.clip(py / self._frame_h, 0.0, 1.0)),
            )
        return 0.0, 0.0
