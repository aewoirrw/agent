from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field, ConfigDict, AliasChoices


class ChatRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    id: Optional[str] = Field(default=None, alias='Id', validation_alias=AliasChoices('Id', 'id', 'ID'))
    question: str = Field(default='', alias='Question', validation_alias=AliasChoices('Question', 'question', 'QUESTION'))


class ClearRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    id: Optional[str] = Field(default=None, alias='Id', validation_alias=AliasChoices('Id', 'id', 'ID'))


class ChatResponse(BaseModel):
    success: bool
    answer: Optional[str] = None
    errorMessage: Optional[str] = None


class ApiResponse(BaseModel):
    code: int = 200
    message: str = 'success'
    data: Any = None


class SseMessage(BaseModel):
    type: str
    data: Optional[str] = None


class SessionInfoResponse(BaseModel):
    sessionId: str
    messagePairCount: int
    createTime: int


class FileUploadRes(BaseModel):
    fileName: str
    filePath: str
    fileSize: int
    taskId: Optional[str] = None
    taskStatus: Optional[str] = None


class UploadTaskStatusResponse(BaseModel):
    taskId: str
    fileName: str
    filePath: str
    fileSize: int
    status: str
    progress: int
    stage: str
    message: str
    totalChunks: int = 0
    completedChunks: int = 0
    startedAt: int
    finishedAt: Optional[int] = None
    durationMs: Optional[int] = None
    error: Optional[str] = None
    extra: dict[str, Any] = Field(default_factory=dict)
