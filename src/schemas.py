# src/schemas.py

from pydantic import BaseModel, Field
from typing import Optional


class QuestionRequest(BaseModel):
    """
    User ka request — API ko yeh milega.
    """
    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="User ka question"
    )
    session_id: Optional[str] = Field(
        default="default",
        description="User ka unique session ID"
    )


class AnswerResponse(BaseModel):
    """
    API ka response — yeh wapas jayega.
    """
    answer: str = Field(description="Bot ka jawab")
    sources: list[str] = Field(
        default=[],
        description="Jinse answer aaya"
    )
    session_id: str = Field(description="Session ID")


class HealthResponse(BaseModel):
    """
    Server health check response.
    """
    status: str
    message: str


class ClearHistoryRequest(BaseModel):
    """
    History clear karne ka request.
    """
    session_id: Optional[str] = Field(
        default="default",
        description="Kis session ki history clear karni hai"
    )


class ClearHistoryResponse(BaseModel):
    """
    History clear hone ka confirmation.
    """
    message: str
    session_id: str