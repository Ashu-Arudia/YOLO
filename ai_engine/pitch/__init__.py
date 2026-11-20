"""
pitch — Pitch detection, homography, and minimap rendering.

Public API
----------
PitchDetector : detects field boundary, lines, and provides pixel → pitch transforms.
PitchMapper   : renders a top-down 2D minimap with player / ball markers.
"""
from ai_engine.pitch.pitch_detector import PitchDetector
from ai_engine.pitch.pitch_mapper import PitchMapper

__all__ = ["PitchDetector", "PitchMapper"]
