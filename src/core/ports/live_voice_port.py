from abc import ABC, abstractmethod
from typing import AsyncGenerator
from dataclasses import dataclass

@dataclass
class VoiceEvent:
    type: str  # "audio", "transcript", "interrupted", "tool_call"
    data: bytes | None = None
    text: str | None = None
    metadata: dict | None = None

class AbstractVoiceClient(ABC):
    @abstractmethod
    async def connect(self):
        """Inicia el handshake WS/gRPC contra el proveedor."""
        pass

    @abstractmethod
    async def send_audio(self, pcm_chunk: bytes):
        """Traduce PCM crudo al payload esperado (Blob, Base64 JSON, etc.)."""
        pass

    @abstractmethod
    async def listen(self) -> AsyncGenerator[VoiceEvent, None]:
        """Normaliza los eventos propietarios a VoiceEvent estándar."""
        yield

    @abstractmethod
    async def close(self):
        """Cierre ordenado de la conexión."""
        pass
