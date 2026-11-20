"""
MatchService
────────────
Manages VideoProcessor instances per match.
Persists analytics snapshots to SQLite periodically.
"""
from __future__ import annotations

import asyncio
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend.app.core.config import settings
from backend.app.database.session import AsyncSessionLocal
from backend.app.models.db_models import Match, MatchAnalytics, PlayerRecord
from ai_engine.video.processor import VideoProcessor

logger = logging.getLogger(__name__)


class MatchService:
    """
    Singleton-ish service.
    One VideoProcessor per active match_id.
    """

    def __init__(self) -> None:
        self._processors: Dict[str, VideoProcessor] = {}
        self._save_interval = 30   # save analytics every N frames

    # ─────────────────────────────────────────────────── match management

    async def create_match(
        self,
        name: str,
        source_type: str,
        video_path: Optional[str] = None,
    ) -> Match:
        async with AsyncSessionLocal() as db:
            match = Match(
                name=name,
                source_type=source_type,
                video_path=video_path,
                status="pending",
            )
            db.add(match)
            await db.commit()
            await db.refresh(match)
            logger.info("Created match %s — %s", match.id, name)
            return match

    async def get_match(self, match_id: str) -> Optional[Match]:
        async with AsyncSessionLocal() as db:
            return await db.get(Match, match_id)

    async def list_matches(self) -> list[Match]:
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(Match).order_by(Match.created_at.desc())
            )
            return list(result.scalars().all())

    async def save_video(self, match_id: str, src_path: Path) -> str:
        """Copy uploaded video to storage/videos/<match_id>/."""
        dest_dir = settings.videos_path / match_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src_path.name
        shutil.copy2(src_path, dest)
        dest_str = str(dest)

        async with AsyncSessionLocal() as db:
            match = await db.get(Match, match_id)
            if match:
                match.video_path = dest_str
                await db.commit()

        return dest_str

    # ─────────────────────────────────────────────────── processing

    async def start_processing(self, match_id: str) -> bool:
        match = await self.get_match(match_id)
        if not match:
            return False
        if match_id in self._processors:
            logger.warning("Match %s already processing", match_id)
            return True

        source: int | str
        if match.source_type == "webcam":
            source = 0
        elif match.video_path:
            source = match.video_path
        else:
            logger.error("Match %s has no video path", match_id)
            return False

        vp = VideoProcessor(source=source, match_id=match_id)
        self._processors[match_id] = vp
        vp.start_background()

        # Update DB status
        async with AsyncSessionLocal() as db:
            m = await db.get(Match, match_id)
            if m:
                m.status = "processing"
                m.started_at = datetime.now(timezone.utc)
                await db.commit()

        # Start analytics saver in background
        asyncio.create_task(self._analytics_saver(match_id, vp))
        logger.info("Started processing match %s", match_id)
        return True

    async def stop_processing(self, match_id: str) -> bool:
        vp = self._processors.pop(match_id, None)
        if not vp:
            return False
        vp.stop()

        async with AsyncSessionLocal() as db:
            m = await db.get(Match, match_id)
            if m:
                m.status = "stopped"
                m.ended_at = datetime.now(timezone.utc)
                await db.commit()

        logger.info("Stopped match %s", match_id)
        return True

    def get_processor(self, match_id: str) -> Optional[VideoProcessor]:
        return self._processors.get(match_id)

    def get_analytics(self, match_id: str) -> dict:
        vp = self._processors.get(match_id)
        if not vp:
            return {}
        return vp.latest_analytics

    def get_frame_gen(self, match_id: str):
        vp = self._processors.get(match_id)
        if not vp:
            return None
        return vp.frame_gen()

    # ─────────────────────────────────────────────────── private

    async def _analytics_saver(
        self, match_id: str, vp: VideoProcessor
    ) -> None:
        """Periodically persists analytics snapshot to DB."""
        last_frame = -1
        while match_id in self._processors:
            await asyncio.sleep(2)
            a = vp.latest_analytics
            if not a or a.get("frame_idx", -1) == last_frame:
                continue
            last_frame = a["frame_idx"]
            try:
                poss = a.get("possession", {})
                ball = a.get("ball_pos")
                async with AsyncSessionLocal() as db:
                    snap = MatchAnalytics(
                        match_id=match_id,
                        frame_idx=a["frame_idx"],
                        fps=a.get("fps", 0),
                        player_count=a.get("player_count", 0),
                        team_a_count=a.get("team_a_count", 0),
                        team_b_count=a.get("team_b_count", 0),
                        team_a_possession=poss.get("team_a_pct", 50),
                        team_b_possession=poss.get("team_b_pct", 50),
                        ball_x=ball[0] if ball else None,
                        ball_y=ball[1] if ball else None,
                    )
                    db.add(snap)
                    await db.commit()
            except Exception as exc:
                logger.error("Analytics save error: %s", exc)

        # On exit: save per-player stats
        await self._save_player_stats(match_id, vp)
        async with AsyncSessionLocal() as db:
            m = await db.get(Match, match_id)
            if m and m.status == "processing":
                m.status = "completed"
                m.ended_at = datetime.now(timezone.utc)
                await db.commit()

    async def _save_player_stats(
        self, match_id: str, vp: VideoProcessor
    ) -> None:
        tracks = vp._tracker._registry
        if not tracks:
            return
        async with AsyncSessionLocal() as db:
            for tid, info in tracks.items():
                rec = PlayerRecord(
                    match_id=match_id,
                    track_id=tid,
                    team=info.team,
                    distance_px=info.distance_px,
                    max_speed_px=info.speed_px,
                    first_seen=0,
                    last_seen=info.last_seen,
                )
                db.add(rec)
            await db.commit()


# Global singleton
match_service = MatchService()
