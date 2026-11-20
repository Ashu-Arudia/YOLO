"""
WebSocket endpoint — streams live analytics JSON to connected clients.

WS /ws/matches/{match_id}

Sends a JSON message every ~500 ms with the latest analytics snapshot.
"""
from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.app.services.match_service import match_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])


@router.websocket("/ws/matches/{match_id}")
async def analytics_ws(websocket: WebSocket, match_id: str):
    await websocket.accept()
    logger.info("WS client connected for match %s", match_id)

    try:
        while True:
            data = match_service.get_analytics(match_id)
            if data:
                # Make JSON-serializable
                payload = _serialise(data)
                await websocket.send_text(json.dumps(payload))
            else:
                await websocket.send_text(
                    json.dumps({"match_id": match_id, "status": "waiting"})
                )
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        logger.info("WS client disconnected for match %s", match_id)
    except Exception as exc:
        logger.error("WS error for match %s: %s", match_id, exc)
        await websocket.close()


def _serialise(data: dict) -> dict:
    """Convert non-JSON-serializable types."""
    out = {}
    for k, v in data.items():
        if isinstance(v, dict):
            out[k] = _serialise(v)
        elif isinstance(v, (int, float, str, bool, type(None))):
            out[k] = v
        elif isinstance(v, tuple):
            out[k] = list(v)
        elif isinstance(v, list):
            out[k] = [
                list(i) if isinstance(i, tuple) else i for i in v
            ]
        else:
            out[k] = str(v)
    return out
