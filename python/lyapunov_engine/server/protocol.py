import time
from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "lyapunov-model"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = -1
    max_tokens: Optional[int] = 128
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    include_stability_diagnostics: Optional[bool] = False


class CompletionRequest(BaseModel):
    model: str = "lyapunov-model"
    prompt: Union[str, List[int]]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = -1
    max_tokens: Optional[int] = 128
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    include_stability_diagnostics: Optional[bool] = False


class StabilityDiagnostics(BaseModel):
    lyapunov_exponent: float = 0.0
    semantic_entropy: float = 0.0
    confidence_rating: str = "high"
    diagnostics_status: str = "executed" # "executed" | "lightweight" | "disabled_context_limit"
    cluster_count: int = 1


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo
    stability_diagnostics: Optional[StabilityDiagnostics] = None


class ChatCompletionChunkDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionChunkChoice(BaseModel):
    index: int
    delta: ChatCompletionChunkDelta
    finish_reason: Optional[str] = None


class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChunkChoice]
    stability_diagnostics: Optional[StabilityDiagnostics] = None


class CompletionResponseChoice(BaseModel):
    index: int
    text: str
    finish_reason: Optional[str] = "stop"


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseChoice]
    usage: UsageInfo
    stability_diagnostics: Optional[StabilityDiagnostics] = None


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "lyapunov-engine"


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard]
