import pytest
import torch
from lyapunov_engine.engine.entropy import compute_semantic_entropy, simple_string_similarity
from lyapunov_engine.engine.lyapunov import compute_lyapunov_divergence
from lyapunov_engine.models.llama import LlamaConfig, LlamaForCausalLM
from lyapunov_engine.server.protocol import ChatCompletionRequest, StabilityDiagnostics, ChatCompletionResponse, ChatMessage, UsageInfo


def test_semantic_entropy_identical():
    candidates = ["The capital of France is Paris."] * 4
    entropy, conf, clusters = compute_semantic_entropy(candidates)
    assert entropy == 0.0
    assert conf == "high"
    assert len(set(clusters)) == 1


def test_semantic_entropy_divergent():
    candidates = [
        "The capital of France is Paris.",
        "Quantum mechanics governs subatomic physics.",
        "A recipe for chocolate chip cookies with butter.",
        "Deep learning models optimize loss functions with Adam."
    ]
    entropy, conf, clusters = compute_semantic_entropy(candidates, similarity_threshold=0.5)
    assert entropy > 1.0 # Significant entropy
    assert conf in ("moderate", "low")
    assert len(set(clusters)) == 4


def test_lyapunov_divergence():
    config = LlamaConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16
    )
    model = LlamaForCausalLM(config)
    prompt_tokens = [12, 45, 78, 90, 33]

    lambda_exp, history = compute_lyapunov_divergence(
        model=model,
        prompt_tokens=prompt_tokens,
        num_steps=8,
        device="cpu"
    )

    assert isinstance(lambda_exp, float)
    assert len(history) == 8
    assert not math_is_nan(lambda_exp)


def math_is_nan(val):
    import math
    return math.isnan(val)


def test_stability_protocol_serialization():
    req = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="Hello")],
        include_stability_diagnostics=True
    )
    assert req.include_stability_diagnostics is True

    diag = StabilityDiagnostics(
        lyapunov_exponent=0.34,
        semantic_entropy=0.05,
        confidence_rating="high",
        diagnostics_status="executed",
        cluster_count=1
    )

    resp = ChatCompletionResponse(
        id="test-123",
        model="lyapunov-model",
        choices=[],
        usage=UsageInfo(),
        stability_diagnostics=diag
    )

    json_str = resp.model_dump_json()
    assert "stability_diagnostics" in json_str
    assert "lyapunov_exponent" in json_str
    assert "confidence_rating" in json_str
