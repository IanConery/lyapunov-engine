from lyapunov_engine.server.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    UsageInfo,
)


def test_chat_protocol_serialization():
    req = ChatCompletionRequest(
        model="meta-llama/Llama-3.2-1B-Instruct",
        messages=[
            ChatMessage(role="system", content="You are a helpful assistant."),
            ChatMessage(role="user", content="Explain PagedAttention."),
        ],
        temperature=0.7,
        max_tokens=64,
        stream=True,
    )
    req_dict = req.model_dump()
    assert req_dict["model"] == "meta-llama/Llama-3.2-1B-Instruct"
    assert len(req_dict["messages"]) == 2
    assert req_dict["stream"] is True


def test_completion_response_structure():
    resp = ChatCompletionResponse(
        id="chatcmpl-12345",
        model="lyapunov-model",
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(
                    role="assistant", content="PagedAttention maps virtual KV blocks."
                ),
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(prompt_tokens=10, completion_tokens=8, total_tokens=18),
    )
    assert resp.choices[0].message.content == "PagedAttention maps virtual KV blocks."
    assert resp.usage.total_tokens == 18
