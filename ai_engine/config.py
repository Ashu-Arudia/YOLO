"""
SportsVision AI — Central Configuration
All tunable parameters live here; override via .env
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "ai_engine" / "models"
STORAGE_DIR = Path(os.getenv("STORAGE_DIR", str(BASE_DIR / "storage")))
VIDEOS_DIR  = Path(os.getenv("VIDEOS_DIR",  str(STORAGE_DIR / "videos")))
PROCESSED_DIR = Path(os.getenv("PROCESSED_DIR", str(STORAGE_DIR / "processed")))
FRAMES_DIR  = Path(os.getenv("FRAMES_DIR",  str(STORAGE_DIR / "frames")))

# ── Detection ─────────────────────────────────────────
YOLO_MODEL    : str   = os.getenv("YOLO_MODEL", "yolov8n.pt")
DETECT_CONF   : float = float(os.getenv("DETECTION_CONF", "0.20"))
DETECT_IOU    : float = float(os.getenv("DETECTION_IOU",  "0.45"))
DETECT_IMGSZ  : int   = int(os.getenv("DETECTION_IMGSZ", "640"))

# COCO class IDs we care about
CLASS_PERSON      = 0
CLASS_SPORTS_BALL = 32
DETECT_CLASSES    = [CLASS_PERSON, CLASS_SPORTS_BALL]

# ── Tracking ──────────────────────────────────────────
TRACKER_CONFIG = "bytetrack.yaml"   # built-in to ultralytics

# ── Team identification ───────────────────────────────
TEAM_CLUSTERS   : int   = 2          # K-Means k
TEAM_CACHE_TTL  : int   = 30        # frames before re-classifying

# ── Ball tracking ─────────────────────────────────────
BALL_TRAIL_LEN  : int   = 25        # number of past positions to show

# ── Video processing ──────────────────────────────────
FRAME_SKIP      : int   = int(os.getenv("FRAME_SKIP", "0"))
MAX_FRAME_DIM   : int   = 1280      # resize if larger than this (0 = no resize)

# ── Visualisation ─────────────────────────────────────
TEAM_A_COLOR    = (220, 60,  60)    # BGR red
TEAM_B_COLOR    = (60,  100, 220)   # BGR blue
REF_COLOR       = (30,  210, 30)    # BGR green
BALL_COLOR      = (255, 255, 255)   # BGR white
UNKNOWN_COLOR   = (160, 160, 160)   # BGR grey

# ── MJPEG streaming ───────────────────────────────────
MJPEG_QUALITY   : int   = int(os.getenv("MJPEG_QUALITY", "80"))
STREAM_FPS_MAX  : int   = int(os.getenv("STREAM_FPS_LIMIT", "25"))

# ── Possession ────────────────────────────────────────
POSSESSION_SMOOTH_ALPHA : float = 0.85   # EMA smoothing
POSSESSION_MAX_DIST_PX  : int   = 120    # pixels — ball "owner" threshold

# ── Pitch detection ──────────────────────────────────
PITCH_GREEN_LO          = (25, 40, 40)    # HSV lower bound for grass
PITCH_GREEN_HI          = (90, 255, 255)  # HSV upper bound for grass
PITCH_MINIMAP_W : int   = 300
PITCH_MINIMAP_H : int   = 200
HOUGH_THRESHOLD : int   = 80
HOUGH_MIN_LINE_LEN : int = 100
HOUGH_MAX_LINE_GAP : int = 30
