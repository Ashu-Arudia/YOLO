"""
Match management + MJPEG streaming endpoints
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from backend.app.models.schemas import MatchCreate, MatchResponse, AnalyticsSnapshot
from backend.app.services.match_service import match_service

router = APIRouter(prefix="/matches", tags=["matches"])


@router.post("", response_model=MatchResponse, status_code=201)
async def create_match(body: MatchCreate):
    match = await match_service.create_match(
        name=body.name,
        source_type=body.source_type,
    )
    return match


@router.get("", response_model=list[MatchResponse])
async def list_matches():
    return await match_service.list_matches()


@router.get("/{match_id}", response_model=MatchResponse)
async def get_match(match_id: str):
    match = await match_service.get_match(match_id)
    if not match:
        raise HTTPException(404, "Match not found")
    return match


@router.post("/{match_id}/start")
async def start_match(match_id: str):
    ok = await match_service.start_processing(match_id)
    if not ok:
        raise HTTPException(400, "Cannot start processing — check match and video path")
    return {"match_id": match_id, "status": "processing"}


@router.post("/{match_id}/stop")
async def stop_match(match_id: str):
    ok = await match_service.stop_processing(match_id)
    if not ok:
        raise HTTPException(404, "Match not found or not currently processing")
    return {"match_id": match_id, "status": "stopped"}


@router.get("/{match_id}/analytics")
async def get_analytics(match_id: str):
    data = match_service.get_analytics(match_id)
    if not data:
        raise HTTPException(404, "No analytics available — is the match processing?")
    return data


@router.get("/{match_id}/stream")
async def mjpeg_stream(match_id: str):
    """MJPEG video stream for a processing match."""
    gen = match_service.get_frame_gen(match_id)
    if gen is None:
        raise HTTPException(404, "Match not found or not processing")
    return StreamingResponse(
        gen,
        media_type="multipart/x-mixed-replace;boundary=frame",
        headers={"Cache-Control": "no-cache"},
    )


@router.get("/{match_id}/players")
async def get_players(match_id: str):
    vp = match_service.get_processor(match_id)
    if not vp:
        raise HTTPException(404, "Match not found or not processing")
    tracks = vp._tracker.active_tracks
    return [
        {
            "track_id":    t.track_id,
            "team":        t.team,
            "centroid":    t.centroid,
            "distance_px": round(t.distance_px, 1),
            "speed_px":    round(t.speed_px, 1),
            "last_seen":   t.last_seen,
        }
        for t in tracks
        if t.class_name == "player"
    ]
