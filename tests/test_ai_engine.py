"""
Unit tests for AI Engine components:
- PitchDetector
- PitchMapper
- TeamClassifier
- PossessionAnalyzer
- TrackingManager
- BallTracker
"""
import numpy as np
import pytest

from ai_engine.pitch.pitch_detector import PitchDetector
from ai_engine.pitch.pitch_mapper import PitchMapper
from ai_engine.team_identification.team_classifier import TeamClassifier
from ai_engine.analytics.possession_analyzer import PossessionAnalyzer
from ai_engine.tracking.tracking_manager import TrackingManager, TrackInfo
from ai_engine.ball_tracking.ball_tracker import BallTracker


def test_pitch_detector():
    pd = PitchDetector()
    # Create synthetic frame: green rectangle in center
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Green in HSV corresponds roughly to BGR (0, 180, 0)
    frame[50:430, 50:590] = [30, 180, 30]

    pd.update(frame)
    assert pd.field_mask is not None
    # Check point transformation
    nx, ny = pd.transform_point(320, 240)
    assert 0.0 <= nx <= 1.0
    assert 0.0 <= ny <= 1.0

    # Inside green region
    assert pd.is_on_pitch(320, 240) is True
    # Reset
    pd.reset()
    assert pd.field_mask is None


def test_pitch_mapper():
    pm = PitchMapper()
    tracks = [
        TrackInfo(track_id=1, team="team_a", centroid=(0.3, 0.4)),
        TrackInfo(track_id=2, team="team_b", centroid=(0.7, 0.6)),
    ]
    minimap = pm.render_minimap(player_tracks=tracks, ball_pos=(0.5, 0.5))
    assert minimap is not None
    assert minimap.shape == (200, 300, 3)
    pm.reset()


def test_team_classifier():
    tc = TeamClassifier()
    assert tc._calibrated is False

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = [100, 150, 100]  # background
    # Player 1: Red jersey
    frame[50:150, 100:150] = [20, 20, 200]
    # Player 2: Blue jersey
    frame[50:150, 200:250] = [200, 50, 20]

    boxes = np.array([
        [100, 50, 150, 150],
        [200, 50, 250, 150],
    ])

    for _ in range(35):
        tc.feed(frame, boxes)

    assert tc._calibrated is True
    team1 = tc.classify(track_id=1, frame=frame, bbox=boxes[0], frame_idx=1)
    team2 = tc.classify(track_id=2, frame=frame, bbox=boxes[1], frame_idx=1)
    assert team1 in ["team_a", "team_b"]
    assert team2 in ["team_a", "team_b"]
    assert team1 != team2


def test_possession_analyzer():
    pa = PossessionAnalyzer()
    track_a = TrackInfo(track_id=1, team="team_a", centroid=(100.0, 100.0))
    track_b = TrackInfo(track_id=2, team="team_b", centroid=(300.0, 300.0))

    # Ball near player A
    stats = pa.update(ball_pos=(105.0, 102.0), player_tracks=[track_a, track_b])
    assert stats["holder_id"] == 1
    assert stats["holder_team"] == "team_a"

    # Repeated updates increase team A's possession
    for _ in range(10):
        stats = pa.update(ball_pos=(105.0, 102.0), player_tracks=[track_a, track_b])
    assert stats["team_a_pct"] > 50.0

    pa.reset()
    assert pa._total_frames == 0


def test_tracking_manager():
    tm = TrackingManager()
    info = TrackInfo(track_id=10)
    info.update_position(100.0, 100.0, 1)
    assert info.centroid == (100.0, 100.0)
    info.update_position(103.0, 104.0, 2)
    assert info.speed_px == 5.0
    assert info.distance_px == 5.0

    tm._registry[10] = info
    tm._frame_idx = 2
    assert len(tm.active_tracks) == 1
    tm.set_team(10, "team_a")
    assert tm._registry[10].team == "team_a"
    tm.reset()
    assert len(tm.active_tracks) == 0


def test_ball_tracker_predict():
    bt = BallTracker()
    assert bt.position is None
    # Feed kalman update directly
    pos = bt._kalman.update(100.0, 100.0)
    assert pos is not None
    pred = bt._kalman.predict()
    assert pred is not None
    bt.reset()
    assert bt.position is None
