from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VoiceEvent:
    type: str  # "audio" | "transcript" | "interrupted" | "error"
    data: bytes | None = None
    text: str | None = None
    metadata: dict[str, Any] | None = None


class AbstractVoiceClient(ABC):
    @abstractmethod
    async def connect(self) -> None:
        """Establece conexión con el proveedor (handshake, WS, auth)."""
        pass

    @abstractmethod
    async def send_audio(self, audio_chunk: bytes) -> None:
        """Envía audio PCM crudo del usuario hacia el proveedor."""
        pass

    @abstractmethod
    async def listen(self) -> AsyncGenerator[VoiceEvent, None]:
        """Produce eventos de voz normalizados desde el proveedor."""
        yield  # type: ignore

    @abstractmethod
    async def close(self) -> None:
        """Cierra la sesión con el proveedor liberando sockets y tareas."""
        pass


VoiceClientFactory = Callable[[], AbstractVoiceClient]
