"""Azure AI Foundry Voice Agents realtime adapter.
Production-ready: Reutiliza Project Client inyectado y maneja cancelación limpia.
"""

from __future__ import annotations

import base64
import binascii
import logging
from collections.abc import AsyncGenerator, Mapping
from typing import TYPE_CHECKING, Any

from azure.ai.projects.models import RealtimeServerEventType
from core.ports.live_voice_provider_port import AbstractVoiceClient, VoiceEvent

if TYPE_CHECKING:
    from azure.ai.projects.aio import AIProjectClient
    from azure.ai.projects.aio.operations import (
        AsyncRealtimeConnection,
        AsyncRealtimeConnectionManager,
    )

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

    SAMPLE_RATE = 24_000
    CHANNELS = 1
    SAMPLE_WIDTH_BYTES = 2

    def __init__(
        self,
        *,
        project_client: AIProjectClient,
        agent_name: str,
    ) -> None:
        self.project_client = project_client
        self.agent_name = agent_name

        self._manager: AsyncRealtimeConnectionManager | None = None
        self._connection: AsyncRealtimeConnection | None = None
        self._closed = False
        self._pending_audio_buffer = bytearray()

    async def connect(self) -> None:
        if self._closed:
            raise RuntimeError("El adaptador ya ha sido cerrado.")
        if self._connection is not None:
            return

        self._manager = self.project_client.voice_agents.realtime.connect(
            agent_name=self.agent_name
        )
        self._connection = await self._manager.__aenter__()
        logger.info("Sesión iniciada con Azure Foundry Agent: %s", self.agent_name)

    async def send_audio(self, pcm_chunk: bytes) -> None:
        if not pcm_chunk or self._closed:
            return

        conn = self._require_connection()

        # Manejo tolerante a fragmentación impar de PCM16
        if self._pending_audio_buffer:
            pcm_chunk = bytes(self._pending_audio_buffer) + pcm_chunk
            self._pending_audio_buffer.clear()

        remainder = len(pcm_chunk) % self.SAMPLE_WIDTH_BYTES
        if remainder:
            # Almacena el byte desalineado para el próximo paquete
            self._pending_audio_buffer.extend(pcm_chunk[-remainder:])
            pcm_chunk = pcm_chunk[:-remainder]

        if pcm_chunk:
            await conn.input_audio_buffer.append(audio=pcm_chunk)

    async def listen(self) -> AsyncGenerator[VoiceEvent, None]:
        conn = self._require_connection()
        try:
            async for event in conn:
                normalized = self._normalize_event(event)
                if normalized is not None:
                    yield normalized
        except Exception as exc:
            # Si el socket cierra o se cancela la tarea, no relanzamos si estamos cerrando
            if not self._closed:
                logger.error("Error en streaming downstream de Foundry: %r", exc)
                raise

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        if self._manager is not None:
            try:
                # Cierre ordenado del socket WebSocket del manager
                await self._manager.__aexit__(None, None, None)
            except Exception as e:
                logger.warning("Error cerrando la conexión de Azure Realtime: %r", e)
            finally:
                self._manager = None
                self._connection = None

        # NO cerrar self.project_client aquí. Es un recurso compartido.

    def _require_connection(self) -> AsyncRealtimeConnection:
        if self._connection is None or getattr(self._connection, "closed", False):
            raise RuntimeError("Conexión realtime inactiva. Llama a connect() primero.")
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
