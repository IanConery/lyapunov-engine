import time

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "lyapunov-model"
    messages: list[ChatMessage]
    temperature: float | None = 0.7
    top_p: float | None = 1.0
    top_k: int | None = -1
    max_tokens: int | None = 128
    stream: bool | None = False
    stop: str | list[str] | None = None
    include_stability_diagnostics: bool | None = False


class CompletionRequest(BaseModel):
    model: str = "lyapunov-model"
    prompt: str | list[int]
    temperature: float | None = 0.7
    top_p: float | None = 1.0
    top_k: int | None = -1
    max_tokens: int | None = 128
    stream: bool | None = False
    stop: str | list[str] | None = None
    include_stability_diagnostics: bool | None = False


class StabilityDiagnostics(BaseModel):
    lyapunov_exponent: float = 0.0
    semantic_entropy: float = 0.0
    confidence_rating: str = "high"
    diagnostics_status: str = (
        "executed"  # "executed" | "lightweight" | "disabled_context_limit"
    )
    cluster_count: int = 1


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str | None = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionResponseChoice]
    usage: UsageInfo
    stability_diagnostics: StabilityDiagnostics | None = None


class ChatCompletionChunkDelta(BaseModel):
    role: str | None = None
    content: str | None = None


class ChatCompletionChunkChoice(BaseModel):
    index: int
    delta: ChatCompletionChunkDelta
    finish_reason: str | None = None


class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChunkChoice]
    stability_diagnostics: StabilityDiagnostics | None = None


class CompletionResponseChoice(BaseModel):
    index: int
    text: str
    finish_reason: str | None = "stop"


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionResponseChoice]
    usage: UsageInfo
    stability_diagnostics: StabilityDiagnostics | None = None


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "lyapunov-engine"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelCard]
