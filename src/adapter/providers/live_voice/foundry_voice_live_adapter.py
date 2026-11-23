"""Azure AI Foundry Voice Agents realtime adapter.
Targets Foundry agents with ``kind="voice"`` through their own realtime
endpoint (``project.voice_agents.realtime.connect``). The agent owns voice,
VAD, instructions and tools, so this adapter does not send ``session.update``.
"""

from __future__ import annotations

import base64
import binascii
import logging
import os
from collections.abc import AsyncGenerator, Mapping
from typing import TYPE_CHECKING, Any

from azure.ai.projects.aio import AIProjectClient
from azure.ai.projects.models import RealtimeServerEventType
from azure.identity.aio import AzureCliCredential

from src.core.ports.live_voice_port import AbstractVoiceClient, VoiceEvent

if TYPE_CHECKING:
    from azure.ai.projects.aio.operations import (
        AsyncRealtimeConnection,
        AsyncRealtimeConnectionManager,
    )
    from azure.core.credentials_async import AsyncTokenCredential


logger = logging.getLogger(__name__)

_AUDIO_DELTA = RealtimeServerEventType.RESPONSE_OUTPUT_AUDIO_DELTA.value
_ASSISTANT_TRANSCRIPT_DELTA = (
    RealtimeServerEventType.RESPONSE_OUTPUT_AUDIO_TRANSCRIPT_DELTA.value
)
_ASSISTANT_TRANSCRIPT_DONE = (
    RealtimeServerEventType.RESPONSE_OUTPUT_AUDIO_TRANSCRIPT_DONE.value
)
_USER_TRANSCRIPT_DELTA = (
    RealtimeServerEventType.CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_DELTA.value
)
_USER_TRANSCRIPT_DONE = (
    RealtimeServerEventType.CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_COMPLETED.value
)
_SPEECH_STARTED = RealtimeServerEventType.INPUT_AUDIO_BUFFER_SPEECH_STARTED.value
_TOOL_CALL_DONE = (
    RealtimeServerEventType.RESPONSE_FUNCTION_CALL_ARGUMENTS_DONE.value
)
_ERROR = RealtimeServerEventType.ERROR.value


class FoundryVoiceLiveAdapter(AbstractVoiceClient):
    """Async adapter for a Foundry voice agent (``kind="voice"``).

    The adapter owns the project client and closes it in :meth:`close`.
    """

    SAMPLE_RATE = 24_000
    CHANNELS = 1
    SAMPLE_WIDTH_BYTES = 2

    def __init__(
        self,
        *,
        agent_name: str | None = None,
        endpoint: str | None = None,
        credential: AsyncTokenCredential | None = None,
        project_client: AIProjectClient | None = None,
        allow_preview: bool = False,
    ) -> None:
        self.agent_name = agent_name or os.getenv("AZURE_VOICELIVE_AGENT_ID")
        self.endpoint = (
            endpoint
            or os.getenv("PROJECT_ENDPOINT")
            or os.getenv("FOUNDRY_PROJECT_ENDPOINT")
        )
        self.credential = credential
        self.project_client = project_client

        if self.project_client is None:
            self.credential = self.credential or AzureCliCredential()
            self.project_client = AIProjectClient(
                endpoint=self.endpoint,
                credential=self.credential,
                allow_preview=allow_preview,
            )

        self._manager: AsyncRealtimeConnectionManager | None = None
        self._connection: AsyncRealtimeConnection | None = None
        self._closed = False

    async def __aenter__(self) -> "FoundryVoiceLiveAdapter":
        await self.connect()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    async def connect(self) -> None:
        """Open the realtime WebSocket against the voice agent."""
        if self._closed:
            raise RuntimeError("This voice adapter is closed.")

        if self._connection is not None:
            return

        self._manager = self.project_client.voice_agents.realtime.connect(
            agent_name=self.agent_name
        )
        self._connection = await self._manager.__aenter__()
        logger.info("Connected to Foundry voice agent %r.", self.agent_name)

    async def send_audio(self, pcm_chunk: bytes) -> None:
        """Send raw PCM16 mono audio to the agent.

        The SDK base64-encodes the payload before writing it to the WebSocket.
        """
        if not pcm_chunk:
            return

        if len(pcm_chunk) % self.SAMPLE_WIDTH_BYTES:
            raise ValueError("PCM16 chunks must contain whole 16-bit samples.")

        await self._require_connection().input_audio_buffer.append(audio=pcm_chunk)

    async def listen(self) -> AsyncGenerator[VoiceEvent, None]:
        """Yield normalized events from the agent."""
        async for event in self._require_connection():
            normalized = self._normalize_event(event)
            if normalized is not None:
                yield normalized

    async def close(self) -> None:
        """Close the realtime connection and the project client."""
        if self._closed:
            return

        if self._manager is not None:
            await self._manager.__aexit__(None, None, None)
            self._manager = None
            self._connection = None

        await self.project_client.close()
        self._closed = True

    def _require_connection(self) -> AsyncRealtimeConnection:
        if self._connection is None or self._connection.closed:
            raise RuntimeError("The voice connection is not open. Call connect() first.")
        return self._connection

    def _normalize_event(self, event: Any) -> VoiceEvent | None:
        event_type = self._event_type(event)
        metadata = self._metadata(event)

        if event_type == _AUDIO_DELTA:
            audio = self._decode_audio(self._get(event, "delta"))
            if not audio:
                return None
            return VoiceEvent(type="audio", data=audio, metadata=metadata)

        if event_type in {_ASSISTANT_TRANSCRIPT_DELTA, _ASSISTANT_TRANSCRIPT_DONE}:
            key = "delta" if event_type == _ASSISTANT_TRANSCRIPT_DELTA else "transcript"
            text = self._get(event, key)
            if text is None:
                return None
            return VoiceEvent(
                type="transcript",
                text=str(text),
                metadata={
                    **metadata,
                    "role": "assistant",
                    "final": event_type == _ASSISTANT_TRANSCRIPT_DONE,
                },
            )

        if event_type in {_USER_TRANSCRIPT_DELTA, _USER_TRANSCRIPT_DONE}:
            key = "delta" if event_type == _USER_TRANSCRIPT_DELTA else "transcript"
            text = self._get(event, key)
            if text is None:
                return None
            return VoiceEvent(
                type="transcript",
                text=str(text),
                metadata={
                    **metadata,
                    "role": "user",
                    "final": event_type == _USER_TRANSCRIPT_DONE,
                },
            )

        if event_type == _SPEECH_STARTED:
            return VoiceEvent(type="interrupted", metadata=metadata)

        if event_type == _TOOL_CALL_DONE:
            return VoiceEvent(
                type="tool_call",
                text=self._get(event, "arguments"),
                metadata={**metadata, "name": self._get(event, "name")},
            )

        if event_type == _ERROR:
            error = self._get(event, "error")
            return VoiceEvent(
                type="error",
                text=self._get(error, "message"),
                metadata={**metadata, "code": self._get(error, "code")},
            )

        return None

    @classmethod
    def _metadata(cls, event: Any) -> dict[str, Any]:
        return {
            key: value
            for key in ("event_id", "response_id", "item_id", "call_id")
            if (value := cls._get(event, key)) is not None
        }

    @staticmethod
    def _get(source: Any, key: str, default: Any = None) -> Any:
        if source is None:
            return default
        if isinstance(source, Mapping):
            return source.get(key, default)
        return getattr(source, key, default)

    @classmethod
    def _event_type(cls, event: Any) -> str:
        raw_type = cls._get(event, "type")
        value = getattr(raw_type, "value", raw_type)
        return str(value) if value is not None else ""

    @staticmethod
    def _decode_audio(value: Any) -> bytes | None:
        if isinstance(value, (bytes, bytearray, memoryview)):
            return bytes(value)
        if isinstance(value, str):
            try:
                return base64.b64decode(value)
            except (binascii.Error, ValueError):
                return None
        return None
