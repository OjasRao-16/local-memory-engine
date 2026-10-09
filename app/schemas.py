from typing import Any
from pydantic import BaseModel, Field

class MemoryRequest(BaseModel):
    message: str = Field(min_length=1, max_length=10000)

class RecallRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=50)
    threshold: float | None = Field(default=None, ge=-1.0, le=1.0)

class MemoryOut(BaseModel):
    id: int
    text: str
    type: str
    timestamp: str
    source_message: str
    status: str
    replaced_by: int | None = None
    score: float | None = None

class MemoryResponse(BaseModel):
    stored: list[dict[str, Any]]
    ignored: bool = False
    message: str
