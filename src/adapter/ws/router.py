from typing import Annotated

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from src.adapter.ws.session import VoiceSession
from src.config.dependency_container import get_voice_client_factory
from core.ports.live_voice_provider_port import AbstractVoiceClient, VoiceClientFactory

router = APIRouter(prefix="/ws", tags=["voice"])


@router.websocket("/voice")
async def voice_socket(
    websocket: WebSocket,
    client_factory: Annotated[VoiceClientFactory, Depends(get_voice_client_factory)],
) -> None:
    if not _origin_allowed(websocket):
        await websocket.close(code=1008)
        return

    await websocket.accept()
    try:
        await VoiceSession(websocket, client_factory()).run()
    except WebSocketDisconnect:
        pass
    except Exception:
        #logger.exception("Voice session failed")
        await websocket.close(code=1011)