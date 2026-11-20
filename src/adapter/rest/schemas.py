from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        description="Mensaje del usuario para el agente.",
        examples=["¿Cuál es la tendencia de ingresos de los últimos 3 meses?"],
    )
    user_id: str = Field(
        ...,
        min_length=1,
        description="Identificador del usuario.",
        examples=["user-001"],
    )
    thread_id: str = Field(
        ...,
        min_length=1,
        description="Identificador de la conversación (hilo) a recuperar.",
        examples=["thread-001"],
    )
