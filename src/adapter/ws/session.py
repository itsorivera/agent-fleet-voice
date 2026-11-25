import asyncio
import logging
from fastapi import WebSocket, WebSocketDisconnect
from src.adapter.ws import protocol
from core.ports.live_voice_provider_port import AbstractVoiceClient

logger = logging.getLogger(__name__)


class VoiceSession:
    def __init__(self, websocket: WebSocket, client: AbstractVoiceClient) -> None:
        self._ws = websocket
        self._client = client

    async def run(self) -> None:
        await self._client.connect()

        uplink_task = asyncio.create_task(self._uplink(), name="ws-uplink")
        downlink_task = asyncio.create_task(self._downlink(), name="ws-downlink")

        # Monitoreo simétrico: Si cualquiera falla o termina, se procede al teardown
        done, pending = await asyncio.wait(
            {uplink_task, downlink_task},
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

        # Evaluar excepciones no capturadas en la tarea que completó primero
        for task in done:
            if not task.cancelled() and (exc := task.exception()):
                if not isinstance(exc, WebSocketDisconnect):
                    logger.exception("Excepción crítica en la sesión de voz: %r", exc)
                    raise exc

        await self._client.close()

    async def _uplink(self) -> None:
        while True:
            msg = await self._ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            
            # Binary frame = Audio crudo PCM
            if (audio_bytes := msg.get("bytes")) is not None:
                await self._client.send_audio(audio_bytes)
            
            # Text frame = Comandos de control (e.g. mute, start, stop)
            elif (text_data := msg.get("text")) is not None:
                # Opcional: procesar json de control si se requiere
                pass

    async def _downlink(self) -> None:
        async for event in self._client.listen():
            if event.type == "audio" and event.data:
                await self._ws.send_bytes(event.data)
            elif event.type == "transcript":
                role = event.metadata.get("role", "assistant") if event.metadata else "assistant"
                await self._ws.send_text(protocol.transcript(role, event.text or ""))
            elif event.type == "interrupted":
                await self._ws.send_text(protocol.interrupted())
            elif event.type == "error":
                await self._ws.send_text(protocol.error(event.text or "Error del proveedor"))
