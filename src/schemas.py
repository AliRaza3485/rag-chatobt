# src/schemas.py

from pydantic import BaseModel, Field
from typing import Optional

class QuestionRequest(BaseModel):
    """
    User's request means api will get this.
    
    """
    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="User's question"
    )
    session_id: Optional[str] = Field(
        default="default",
        description="User's unique session ID"
    )

class AnswerResponse(BaseModel):
    """
    The response of api call - will be sent back to the user.
    
    """
    answer: str = Field(description="Bot's answer")
    sources: list[str] = Field(
        default=[],
        description="Sources from which the answer was derived"
    )
    session_id: str = Field(description="Session ID")

class HealthResponse(BaseModel):
    """
    Server health check response.
    
    """
    status: str = Field(description="Health status, e.g., 'healthy'")
    message: str = Field(description="Additional info about the health status")

class ClearHistoryRequest(BaseModel):
    """
    Request to clear chat history for a session.
    
    """
    session_id: Optional[str] = Field(
        default="default",
        description="The session ID for which the chat history should be cleared"
    )

# class ClearHistoryResponse(BaseModel):
#     """
#     History clear hone ka confirmation.
#     """
#     message: str
#     session_id: str
class ClearHistoryResponse(BaseModel):
    """
    The confirmation response after clearing chat history.
    
    """
    message: str = Field(description="Confirmation message about history clearance")
    session_id: str = Field(description="The session ID for which the history was cleared")