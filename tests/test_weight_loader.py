import torch
from lyapunov_engine.models.llama import LlamaConfig, LlamaForCausalLM
from lyapunov_engine.models.weight_loader import load_hf_weights_into_model


def test_weight_loader_synthetic_state_dict():
    config = LlamaConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=512,
    )

    model = LlamaForCausalLM(config)

    # Synthetic HF state dict
    weights = {
        "model.embed_tokens.weight": torch.randn(256, 64),
        "model.norm.weight": torch.randn(64),
        "lm_head.weight": torch.randn(256, 64),
        "model.layers.0.input_layernorm.weight": torch.randn(64),
        "model.layers.0.post_attention_layernorm.weight": torch.randn(64),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(64, 64),
        "model.layers.0.self_attn.k_proj.weight": torch.randn(32, 64),
        "model.layers.0.self_attn.v_proj.weight": torch.randn(32, 64),
        "model.layers.0.self_attn.o_proj.weight": torch.randn(64, 64),
        "model.layers.0.mlp.gate_proj.weight": torch.randn(128, 64),
        "model.layers.0.mlp.up_proj.weight": torch.randn(128, 64),
        "model.layers.0.mlp.down_proj.weight": torch.randn(64, 128),
        "model.layers.1.input_layernorm.weight": torch.randn(64),
        "model.layers.1.post_attention_layernorm.weight": torch.randn(64),
        "model.layers.1.self_attn.q_proj.weight": torch.randn(64, 64),
        "model.layers.1.self_attn.k_proj.weight": torch.randn(32, 64),
        "model.layers.1.self_attn.v_proj.weight": torch.randn(32, 64),
        "model.layers.1.self_attn.o_proj.weight": torch.randn(64, 64),
        "model.layers.1.mlp.gate_proj.weight": torch.randn(128, 64),
        "model.layers.1.mlp.up_proj.weight": torch.randn(128, 64),
        "model.layers.1.mlp.down_proj.weight": torch.randn(64, 128),
    }

    load_hf_weights_into_model(model, weights, dtype=torch.float32, device="cpu")

    # Verify gate_up_proj was fused properly: shape [2 * 128, 64] = [256, 64]
    assert model.layers[0].mlp.gate_up_proj.weight.shape == (256, 64)
    expected_fused_0 = torch.cat(
        [
            weights["model.layers.0.mlp.gate_proj.weight"],
            weights["model.layers.0.mlp.up_proj.weight"],
        ],
        dim=0,
    )
    assert torch.allclose(model.layers[0].mlp.gate_up_proj.weight, expected_fused_0)
