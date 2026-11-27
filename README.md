# SportsVision AI ⚽🎯

**Real-Time Football Intelligence & Performance Analytics Platform**

SportsVision AI combines computer vision and deep learning (YOLOv8 + ByteTrack) with real-time video analytics to track players, classify teams, follow ball trajectories, analyze possession, and map positions to a 2D pitch minimap.

---

## 🌟 Key Features

1. **Player & Ball Detection**: YOLOv8 neural network optimized for detecting football players and the ball.
2. **Multi-Object Tracking**: ByteTrack algorithm for persistent ID tracking across video frames.
3. **Team Classification**: Unsupervised K-Means clustering on HSV jersey color distributions.
4. **Ball Trajectory & Kalman Filter**: 2D constant-velocity Kalman filter for smooth ball tracking and trail visualization.
5. **Pitch Detection & 2D Minimap**: Pitch homography estimation and top-down radar minimap rendering.
6. **Possession Analytics**: Real-time distance-based possession estimation with exponential moving average (EMA) smoothing.
7. **FastAPI Backend**:
   - REST endpoints for match management, video uploads, player stats, and analytics.
   - Multipart MJPEG streaming for live video playback with annotations.
   - WebSocket endpoint (`/ws/matches/{id}`) streaming live telemetry.
   - Async SQLite persistence for matches, analytics snapshots, and player history.
8. **Next.js 16 Web Dashboard**:
   - Live video stream display with match controls (Start / Stop).
   - Real-time possession bars and timeline charts.
   - Player metrics table (speed, distance covered, team assignment).
   - Video upload modal and live webcam launch.

---

## 🏗️ Project Architecture

```
YOLO/
├── ai_engine/                    # Core Computer Vision & AI Pipeline
│   ├── analytics/                # Possession & performance analytics
│   ├── ball_tracking/            # Kalman filter & ball trajectory tracking
│   ├── detection/                # YOLOv8 detector
│   ├── pitch/                    # Pitch boundary detector & 2D minimap
│   ├── team_identification/      # Jersey color K-Means classifier
│   ├── tracking/                 # Multi-object tracking manager
│   ├── video/                    # Frame capture & processing orchestrator
│   └── config.py                 # Central AI tunable parameters
│
├── backend/                      # FastAPI Backend
│   └── app/
│       ├── api/v1/               # Health, matches, videos endpoints
│       ├── api/websocket.py      # Real-time analytics WebSocket
│       ├── core/config.py        # Backend settings via pydantic-settings
│       ├── database/session.py   # SQLAlchemy async engine & session
│       ├── models/               # ORM & Pydantic schemas
│       ├── services/             # MatchService process manager
│       └── main.py               # FastAPI entry point & lifespan
│
├── frontend/                     # Next.js 16 Frontend
│   ├── src/app/                  # App Router pages (Dashboard, Matches, Match Detail)
│   ├── src/components/           # UI components (Video stream, charts, tables)
│   ├── src/hooks/                # Custom React hooks (useMatches, useAnalytics)
│   └── src/lib/api.ts            # API client & types
│
├── app/                          # Android Client Application
│   └── build.gradle              # Android module configuration
│
├── tests/                        # Pytest Test Suite
│   ├── test_ai_engine.py         # AI engine unit tests
│   └── test_api.py               # Backend API integration tests
│
├── run.py                        # Standalone CLI entry point (OpenCV local display)
└── requirements.txt              # Python dependencies
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+ (tested with Python 3.14)
- Node.js 18+ & npm
- A webcam (optional, for live webcam testing)

### 1. Environment Setup

Copy `.env.example` to `.env` (already configured):
```bash
cp .env.example .env
```

### 2. Run Backend (FastAPI)

```bash
.\venv\Scripts\uvicorn.exe backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```
- API Docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/api/v1/health`

### 3. Run Frontend (Next.js)

```bash
cd frontend
npm run dev
```
- Dashboard: `http://localhost:3000`

### 4. Standalone AI Engine CLI (OpenCV Window)

To test the AI engine directly on a webcam or MP4 file with an OpenCV display window:
```bash
# Webcam:
.\venv\Scripts\python.exe run.py --source 0

# Video file:
.\venv\Scripts\python.exe run.py --source path/to/match.mp4
```

### 5. Run Test Suite

```bash
.\venv\Scripts\pytest.exe -v
```
All unit tests and API integration tests should pass.
