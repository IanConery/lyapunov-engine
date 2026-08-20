import torch
from lyapunov_engine.distributed.tensor_parallel import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
from torch import nn


def test_column_parallel_linear_parity():
    torch.manual_seed(42)
    in_features = 64
    out_features = 128
    world_size = 2

    # Reference single unpartitioned linear layer
    ref_linear = nn.Linear(in_features, out_features, bias=False, dtype=torch.float32)

    # Shard 0 & Shard 1
    shard_0 = ColumnParallelLinear(
        in_features,
        out_features,
        bias=False,
        rank=0,
        world_size=world_size,
        dtype=torch.float32,
    )
    shard_1 = ColumnParallelLinear(
        in_features,
        out_features,
        bias=False,
        rank=1,
        world_size=world_size,
        dtype=torch.float32,
    )

    shard_0.load_unpartitioned_weight(ref_linear.weight.data)
    shard_1.load_unpartitioned_weight(ref_linear.weight.data)

    x = torch.randn((4, in_features), dtype=torch.float32)

    out_0 = shard_0(x)  # [4, 64]
    out_1 = shard_1(x)  # [4, 64]

    # Concatenate column shards
    combined_out = torch.cat([out_0, out_1], dim=-1)
    ref_out = ref_linear(x)

    assert torch.allclose(combined_out, ref_out, atol=1e-5, rtol=1e-5)


def test_row_parallel_linear_parity():
    torch.manual_seed(42)
    in_features = 64
    out_features = 128
    world_size = 2

    ref_linear = nn.Linear(in_features, out_features, bias=False, dtype=torch.float32)

    shard_0 = RowParallelLinear(
        in_features,
        out_features,
        bias=False,
        rank=0,
        world_size=world_size,
        dtype=torch.float32,
    )
    shard_1 = RowParallelLinear(
        in_features,
        out_features,
        bias=False,
        rank=1,
        world_size=world_size,
        dtype=torch.float32,
    )

    shard_0.load_unpartitioned_weight(ref_linear.weight.data)
    shard_1.load_unpartitioned_weight(ref_linear.weight.data)

    x = torch.randn((4, in_features), dtype=torch.float32)
    x_0 = x[:, : in_features // world_size]
    x_1 = x[:, in_features // world_size :]

    out_0 = shard_0(x_0)
    out_1 = shard_1(x_1)

    # Simulated all-reduce sum across ranks
    combined_out = out_0 + out_1
    ref_out = ref_linear(x)

    assert torch.allclose(combined_out, ref_out, atol=1e-5, rtol=1e-5)


def test_parallel_embedding():
    torch.manual_seed(42)
    num_embeddings = 100
    embedding_dim = 32
    world_size = 2

    embed_0 = ParallelEmbedding(
        num_embeddings,
        embedding_dim,
        rank=0,
        world_size=world_size,
        dtype=torch.float32,
    )
    embed_1 = ParallelEmbedding(
        num_embeddings,
        embedding_dim,
        rank=1,
        world_size=world_size,
        dtype=torch.float32,
    )

    input_ids = torch.tensor([5, 60, 25, 80], dtype=torch.long)

    out_0 = embed_0(input_ids)
    out_1 = embed_1(input_ids)

    combined = out_0 + out_1
    assert combined.shape == (4, embedding_dim)
    assert not torch.isnan(combined).any()
