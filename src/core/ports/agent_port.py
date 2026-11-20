from abc import ABC, abstractmethod
from typing import Any


class AgentPort(ABC):
    @abstractmethod
    async def create_agent(self) -> Any:
        pass
    
    @abstractmethod
    async def process_message(self, request: Any) -> Any:
        pass