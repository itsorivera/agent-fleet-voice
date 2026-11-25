from functools import lru_cache
from typing import Optional
import os
from azure.ai.projects.aio import AIProjectClient
from azure.identity.aio import DefaultAzureCredential
from src.core.ports.llm_provider_port import LLMProviderPort
from adapter.providers.llm.aws_bedrock_provder_adapter import AWSBedrockProviderAdapter
from src.core.ports.stm_port import STMProviderPort
from src.adapter.memory.postgres_stm_adapter import PostgresSTMAdapter
from src.core.ports.agent_port import AgentPort
from src.adapter.agent.langchain_agent_adapter import LangChainAgentAdapter
from core.ports.live_voice_provider_port import AbstractVoiceClient, VoiceClientFactory
from src.adapter.providers.live_voice.foundry_voice_live_adapter import FoundryVoiceLiveAdapter
from src.core.tools.local_tools import FINANCIAL_ANALYST_TOOLS

class DependenciesContainer:
    def __init__(self):
        self.llm_provider: Optional[LLMProviderPort] = None
        self.stm: Optional[STMProviderPort] = None
        self.agent_1: Optional[AgentPort] = None
        self._foundry_project_client: Optional[AIProjectClient] = None

    def get_llm_provider(self) -> LLMProviderPort:
        if self.llm_provider is None:
            self.llm_provider = AWSBedrockProviderAdapter()
        return self.llm_provider
    
    def get_stm(self) -> STMProviderPort:
        if self.stm is None:
            self.stm = PostgresSTMAdapter()
        return self.stm
    
    async def get_agent_1(self) -> AgentPort:
        llm_provider = self.get_llm_provider()
        stm = self.get_stm()

        if self.agent_1 is None:
            self.agent_1 = LangChainAgentAdapter(
                agent_name="agent-financial-analyst",
                llm_provider=llm_provider,
                system_prompt=
                """Eres un agente que que ayudas con la evaluación de indicadores clave, ratios financieros y tendencias de desempeño.
                Los analistas realizan consultas complejas sobre múltiples periodos financieros y esperan obtener resultados rápidos, precisos y con costos controlados para apoyar la toma de decisiones de inversión.""",
                llm_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                stm_provider=stm,
                tools=FINANCIAL_ANALYST_TOOLS
            )
            await self.agent_1.create_agent()
        return self.agent_1

    def _get_foundry_project_client(self) -> AIProjectClient:
        """Singleton del cliente SDK de Azure Foundry."""
        if self._foundry_project_client is None:
            endpoint = os.getenv("FOUNDRY_PROJECT_ENDPOINT") or os.getenv("PROJECT_ENDPOINT")
            if not endpoint:
                raise ValueError("Variable de entorno FOUNDRY_PROJECT_ENDPOINT no configurada.")
            
            self._foundry_project_client = AIProjectClient(
                endpoint=endpoint,
                credential=DefaultAzureCredential(),
            )
        return self._foundry_project_client

    def get_live_voice_factory(self) -> VoiceClientFactory:
        """
        Retorna un Callable (Factory) para crear una sesión de voz fresca por cada WebSocket.
        Inyecta el AIProjectClient singleton compartido.
        """
        project_client = self._get_foundry_project_client()
        agent_name = os.getenv("AZURE_VOICELIVE_AGENT_ID", "default-voice-agent")

        # Retornamos una función lambda que instancia un nuevo adaptador efímero
        return lambda: FoundryVoiceLiveAdapter(
            project_client=project_client,
            agent_name=agent_name
        )
    

@lru_cache()
def get_dependencies_container() -> DependenciesContainer:
    return DependenciesContainer()

async def get_agent_1() -> AgentPort:
    dependecies = get_dependencies_container()
    return await dependecies.get_agent_1()

def get_voice_client_factory() -> VoiceClientFactory:
    return get_dependencies_container().get_live_voice_factory()