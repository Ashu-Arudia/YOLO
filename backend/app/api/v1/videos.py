"""
POST /api/v1/videos/upload
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
import aiofiles

from backend.app.services.match_service import match_service

router = APIRouter(prefix="/videos", tags=["videos"])

ALLOWED_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MAX_SIZE_MB = 2048   # 2 GB limit


@router.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    match_name: str  = Form(...),
):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Allowed: {ALLOWED_EXTS}")

    # Create the match record first so we have a match_id
    match = await match_service.create_match(
        name=match_name,
        source_type="file",
    )

    # Stream file to temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp_path = Path(tmp.name)
        total = 0
        async for chunk in file:
            total += len(chunk)
            if total > MAX_SIZE_MB * 1024 * 1024:
                tmp_path.unlink(missing_ok=True)
                raise HTTPException(413, "File too large")
            tmp.write(chunk)

    # Move to permanent storage
    video_path = await match_service.save_video(match.id, tmp_path)
    tmp_path.unlink(missing_ok=True)

    return {
        "match_id":   match.id,
        "match_name": match.name,
        "video_path": video_path,
        "status":     match.status,
    }
