import asyncio
import contextlib
import logging

from src.core.ports.live_voice_port import AbstractVoiceClient
from src.adapter.ws import protocol

logger = logging.getLogger(__name__)


class VoiceSession:
    def __init__(self, websocket, client: AbstractVoiceClient) -> None:
        self._ws = websocket
        self._client = client

    async def run(self) -> None:
        await self._client.connect()
        downlink = asyncio.create_task(self._downlink(), name="voice-downlink")
        try:
            await self._uplink()
        finally:
            downlink.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await downlink
            await self._client.close()

    async def _uplink(self) -> None:
        while True:
            message = await self._ws.receive()
            if message["type"] == "websocket.disconnect":
                return
            if (audio := message.get("bytes")) is not None:
                await self._client.send_audio(audio)

    async def _downlink(self) -> None:
        async for event in self._client.listen():
            if event.type == "audio":
                await self._ws.send_bytes(event.data)
            elif event.type == "transcript":
                await self._ws.send_text(
                    protocol.transcript(event.metadata["role"], event.text)
                )
            elif event.type == "interrupted":
                await self._ws.send_text(protocol.interrupted())
            elif event.type == "error":
                await self._ws.send_text(protocol.error(event.text or "unknown"))