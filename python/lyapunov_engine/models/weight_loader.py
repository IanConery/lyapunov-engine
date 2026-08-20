import os
import json
from typing import Dict, Optional, Union
import torch
import torch.nn as nn
from safetensors.torch import load_file
from lyapunov_engine.models.llama import LlamaForCausalLM, LlamaConfig


def load_llama_config_from_hf(model_dir_or_repo: str) -> LlamaConfig:
    """Load and parse model config.json into LlamaConfig."""
    config_path = os.path.join(model_dir_or_repo, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg_dict = json.load(f)
    else:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(repo_id=model_dir_or_repo, filename="config.json")
        with open(downloaded, "r", encoding="utf-8") as f:
            cfg_dict = json.load(f)

    return LlamaConfig(
        vocab_size=cfg_dict.get("vocab_size", 128256),
        hidden_size=cfg_dict.get("hidden_size", 2048),
        intermediate_size=cfg_dict.get("intermediate_size", 8192),
        num_hidden_layers=cfg_dict.get("num_hidden_layers", 16),
        num_attention_heads=cfg_dict.get("num_attention_heads", 32),
        num_key_value_heads=cfg_dict.get("num_key_value_heads", 8),
        head_dim=cfg_dict.get("head_dim", cfg_dict.get("hidden_size", 2048) // cfg_dict.get("num_attention_heads", 32)),
        max_position_embeddings=cfg_dict.get("max_position_embeddings", 8192),
        rms_norm_eps=cfg_dict.get("rms_norm_eps", 1e-5),
        rope_theta=cfg_dict.get("rope_theta", 500000.0)
    )


def load_safetensors_weights(model_dir_or_repo: str, device: str = "cpu") -> Dict[str, torch.Tensor]:
    """Load all safetensors shards from local directory or HuggingFace Hub."""
    state_dict = {}

    if os.path.isdir(model_dir_or_repo):
        # Local directory
        files = [f for f in os.listdir(model_dir_or_repo) if f.endswith(".safetensors")]
        if not files:
            raise FileNotFoundError(f"No .safetensors files found in {model_dir_or_repo}")
        for file in sorted(files):
            file_path = os.path.join(model_dir_or_repo, file)
            shard = load_file(file_path, device=device)
            state_dict.update(shard)
    else:
        # Download from HuggingFace Hub
        from huggingface_hub import snapshot_download
        download_dir = snapshot_download(repo_id=model_dir_or_repo, allow_patterns=["*.safetensors", "*.json"])
        files = [f for f in os.listdir(download_dir) if f.endswith(".safetensors")]
        for file in sorted(files):
            file_path = os.path.join(download_dir, file)
            shard = load_file(file_path, device=device)
            state_dict.update(shard)

    return state_dict


def load_hf_weights_into_model(
    model: LlamaForCausalLM,
    weights: Dict[str, torch.Tensor],
    dtype: torch.dtype = torch.float16,
    device: str = "cpu"
) -> LlamaForCausalLM:
    """Map standard HuggingFace parameter names into lyapunov-engine Llama model parameters.
    
    Handles:
    - Embedding and LM head mapping
    - Fusing separate gate_proj and up_proj weights into fused gate_up_proj for custom SwiGLU kernel
    - Normalization weights and Q/K/V projections
    """
    model_state = model.state_dict()
    num_layers = model.config.num_hidden_layers

    # 1. Token Embeddings & Final Norm
    if "model.embed_tokens.weight" in weights:
        model.embed_tokens.weight.data.copy_(weights["model.embed_tokens.weight"].to(dtype=dtype, device=device))

    if "model.norm.weight" in weights:
        model.norm_weight.data.copy_(weights["model.norm.weight"].to(dtype=dtype, device=device))

    # 2. LM Head (handle tied weights if lm_head is omitted)
    if "lm_head.weight" in weights:
        model.lm_head.weight.data.copy_(weights["lm_head.weight"].to(dtype=dtype, device=device))
    elif "model.embed_tokens.weight" in weights:
        model.lm_head.weight.data.copy_(weights["model.embed_tokens.weight"].to(dtype=dtype, device=device))

    # 3. Transformer Decoder Layers
    for i in range(num_layers):
        layer = model.layers[i]
        prefix = f"model.layers.{i}."

        # Input Layernorm & Post Attention Layernorm
        in_norm_key = f"{prefix}input_layernorm.weight"
        if in_norm_key in weights:
            layer.input_layernorm_weight.data.copy_(weights[in_norm_key].to(dtype=dtype, device=device))

        post_norm_key = f"{prefix}post_attention_layernorm.weight"
        if post_norm_key in weights:
            layer.post_attention_layernorm_weight.data.copy_(weights[post_norm_key].to(dtype=dtype, device=device))

        # Attention Projections: Q, K, V, O
        q_key = f"{prefix}self_attn.q_proj.weight"
        if q_key in weights:
            layer.self_attn.q_proj.weight.data.copy_(weights[q_key].to(dtype=dtype, device=device))

        k_key = f"{prefix}self_attn.k_proj.weight"
        if k_key in weights:
            layer.self_attn.k_proj.weight.data.copy_(weights[k_key].to(dtype=dtype, device=device))

        v_key = f"{prefix}self_attn.v_proj.weight"
        if v_key in weights:
            layer.self_attn.v_proj.weight.data.copy_(weights[v_key].to(dtype=dtype, device=device))

        o_key = f"{prefix}self_attn.o_proj.weight"
        if o_key in weights:
            layer.self_attn.o_proj.weight.data.copy_(weights[o_key].to(dtype=dtype, device=device))

        # MLP Projections: Fuse gate_proj and up_proj into single gate_up_proj
        gate_key = f"{prefix}mlp.gate_proj.weight"
        up_key = f"{prefix}mlp.up_proj.weight"
        if gate_key in weights and up_key in weights:
            gate_w = weights[gate_key].to(dtype=dtype, device=device)
            up_w = weights[up_key].to(dtype=dtype, device=device)
            fused_gate_up = torch.cat([gate_w, up_w], dim=0) # [2 * intermediate_size, hidden_size]
            layer.mlp.gate_up_proj.weight.data.copy_(fused_gate_up)

        down_key = f"{prefix}mlp.down_proj.weight"
        if down_key in weights:
            layer.mlp.down_proj.weight.data.copy_(weights[down_key].to(dtype=dtype, device=device))

    return model


def from_pretrained_llama(
    model_dir_or_repo: str,
    dtype: torch.dtype = torch.float16,
    device: str = "cpu"
) -> LlamaForCausalLM:
    """Instantiate and load a pretrained Llama model from local files or HuggingFace."""
    config = load_llama_config_from_hf(model_dir_or_repo)
    model = LlamaForCausalLM(config).to(dtype=dtype, device=device)
    weights = load_safetensors_weights(model_dir_or_repo, device=device)
    load_hf_weights_into_model(model, weights, dtype=dtype, device=device)
    model.eval()
    return model
