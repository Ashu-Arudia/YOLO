"""
SportsVision AI — CLI Entry Point
──────────────────────────────────
Usage:
  python run.py                          # webcam (device 0)
  python run.py --source 0               # webcam explicit
  python run.py --source path/video.mp4  # video file
  python run.py --source 0 --model yolov8s.pt
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on the path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

from ai_engine.video.processor import VideoProcessor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SportsVision AI — Phase 1 standalone test"
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Video source: 0=webcam, or path to an MP4 file",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="YOLO model file (e.g. yolov8n.pt, yolov8s.pt). "
             "Defaults to value in .env / config.py",
    )
    parser.add_argument(
        "--match-id",
        default="test-match",
        help="Logical match ID for analytics tracking",
    )
    args = parser.parse_args()

    # Convert '0' → int so OpenCV opens the webcam
    source: int | str
    try:
        source = int(args.source)
    except ValueError:
        source = args.source
        if not Path(source).exists():
            logging.error("File not found: %s", source)
            sys.exit(1)

    print(
        "\n╔══════════════════════════════════════════╗"
        "\n║     SportsVision AI — Phase 1 Test       ║"
        "\n╠══════════════════════════════════════════╣"
        f"\n║  Source  : {str(source):<30s}║"
        f"\n║  Model   : {str(args.model or 'default (config.py)'):<30s}║"
        f"\n║  Match ID: {args.match_id:<30s}║"
        "\n╠══════════════════════════════════════════╣"
        "\n║  Press Q in the video window to quit     ║"
        "\n╚══════════════════════════════════════════╝\n"
    )

    vp = VideoProcessor(
        source=source,
        match_id=args.match_id,
        model_path=args.model,
    )
    vp.run_local()


if __name__ == "__main__":
    main()
