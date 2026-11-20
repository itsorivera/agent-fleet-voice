from fastapi import APIRouter, Depends, HTTPException
from typing import Annotated
from src.adapter.rest.schemas import QueryRequest
from src.core.ports.agent_port import AgentPort
from src.config.dependency_container import get_agent_1

router = APIRouter(prefix="/api/v1/analysis",
                   tags=["analysis"])

@router.post(
    path="/query",
    description="Endpoint para consultar al asistente de voz.")
async def query_analysis(
    request: QueryRequest,
    agent: Annotated[AgentPort, Depends(get_agent_1)]
    ):
    try:
        return await agent.process_message(
            message=request.message,
            user_id=request.user_id,
            thread_id=request.thread_id,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=str(e)) from e
