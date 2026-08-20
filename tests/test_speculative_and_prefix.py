import pytest
import torch
from lyapunov_engine.engine.prefix_cache import RadixPrefixCache
from lyapunov_engine.engine.speculative import SpeculativeDecoder
from lyapunov_engine.models.llama import LlamaConfig, LlamaForCausalLM


def test_radix_prefix_cache_matching():
    cache = RadixPrefixCache(block_size=4)

    # Insert a prompt prefix with allocated physical block IDs [10, 11]
    prompt_a = [1, 2, 3, 4, 5, 6, 7, 8] # 2 blocks
    cache.insert_prefix(prompt_a, [10, 11])

    # Query identical prefix
    matched = cache.match_prefix(prompt_a)
    assert matched == [10, 11]

    # Query longer prompt sharing first 2 blocks
    prompt_b = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    matched_b = cache.match_prefix(prompt_b)
    assert matched_b == [10, 11]

    # Query prompt with mismatched 2nd block
    prompt_c = [1, 2, 3, 4, 99, 100, 101, 102]
    matched_c = cache.match_prefix(prompt_c)
    assert matched_c == [10]


def test_speculative_decoder_verification():
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16
    )
    target_model = LlamaForCausalLM(config)

    decoder = SpeculativeDecoder(
        target_model=target_model,
        num_draft_tokens=4,
        temperature=0.7
    )

    prefix_tokens = [10, 20, 30]
    draft_tokens = [40, 50, 60, 70]

    accepted, bonus_tok = decoder.verify_draft_tokens(
        prefix_tokens=prefix_tokens,
        draft_tokens=draft_tokens,
        device="cpu"
    )

    assert isinstance(accepted, list)
    assert len(accepted) <= len(draft_tokens)
    assert decoder.total_drafted == 4
    assert 0.0 <= decoder.acceptance_rate <= 1.0
    if bonus_tok is not None:
        assert isinstance(bonus_tok, int)
