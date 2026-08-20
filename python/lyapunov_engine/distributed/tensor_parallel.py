from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.distributed as dist


_TP_RANK = 0
_TP_WORLD_SIZE = 1


def initialize_tensor_parallel(rank: int = 0, world_size: int = 1):
    """Set global tensor parallel rank and world size."""
    global _TP_RANK, _TP_WORLD_SIZE
    _TP_RANK = rank
    _TP_WORLD_SIZE = world_size


def get_tensor_parallel_rank() -> int:
    return _TP_RANK


def get_tensor_parallel_world_size() -> int:
    return _TP_WORLD_SIZE


class ColumnParallelLinear(nn.Module):
    """Linear layer with column parallelism (sharded across out_features).
    
    Used for Q, K, V projections and gate_up projections in Megatron-LM architecture.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float16
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank if rank is not None else get_tensor_parallel_rank()
        self.world_size = world_size if world_size is not None else get_tensor_parallel_world_size()

        assert out_features % self.world_size == 0, "out_features must be divisible by world_size"
        self.out_features_per_partition = out_features // self.world_size

        self.weight = nn.Parameter(
            torch.empty((self.out_features_per_partition, in_features), dtype=dtype, device=device)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(self.out_features_per_partition, dtype=dtype, device=device)
            )
        else:
            self.register_parameter("bias", None)

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def load_unpartitioned_weight(self, full_weight: torch.Tensor):
        """Slice and load an unpartitioned weight tensor for this rank."""
        start_idx = self.rank * self.out_features_per_partition
        end_idx = start_idx + self.out_features_per_partition
        self.weight.data.copy_(full_weight[start_idx:end_idx, :])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.matmul(x, self.weight.t())
        if self.bias is not None:
            out += self.bias
        return out


class RowParallelLinear(nn.Module):
    """Linear layer with row parallelism (sharded across in_features).
    
    Used for Attention output projections (O) and MLP down projections.
    Performs an all-reduce sum collective operation across tensor parallel ranks.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float16
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank if rank is not None else get_tensor_parallel_rank()
        self.world_size = world_size if world_size is not None else get_tensor_parallel_world_size()

        assert in_features % self.world_size == 0, "in_features must be divisible by world_size"
        self.in_features_per_partition = in_features // self.world_size

        self.weight = nn.Parameter(
            torch.empty((out_features, self.in_features_per_partition), dtype=dtype, device=device)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, dtype=dtype, device=device)
            )
        else:
            self.register_parameter("bias", None)

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def load_unpartitioned_weight(self, full_weight: torch.Tensor):
        """Slice and load an unpartitioned weight tensor for this rank."""
        start_idx = self.rank * self.in_features_per_partition
        end_idx = start_idx + self.in_features_per_partition
        self.weight.data.copy_(full_weight[:, start_idx:end_idx])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.matmul(x, self.weight.t())

        # All-reduce sum across tensor parallel group if distributed is initialized
        if self.world_size > 1 and dist.is_initialized():
            dist.all_reduce(out, op=dist.ReduceOp.SUM)

        if self.bias is not None:
            out += self.bias
        return out


class ParallelEmbedding(nn.Module):
    """Embedding layer partitioned across vocabulary dimension."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float16
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.rank = rank if rank is not None else get_tensor_parallel_rank()
        self.world_size = world_size if world_size is not None else get_tensor_parallel_world_size()

        self.vocab_per_partition = (num_embeddings + self.world_size - 1) // self.world_size
        self.vocab_start_idx = self.rank * self.vocab_per_partition
        self.vocab_end_idx = min(self.vocab_start_idx + self.vocab_per_partition, num_embeddings)

        self.weight = nn.Parameter(
            torch.empty((self.vocab_end_idx - self.vocab_start_idx, embedding_dim), dtype=dtype, device=device)
        )
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Mask tokens that belong to this rank
        mask = (input_ids >= self.vocab_start_idx) & (input_ids < self.vocab_end_idx)
        local_ids = (input_ids - self.vocab_start_idx) * mask.long()

        local_embed = torch.embedding(self.weight, local_ids)
        local_embed = local_embed * mask.unsqueeze(-1).to(local_embed.dtype)

        if self.world_size > 1 and dist.is_initialized():
            dist.all_reduce(local_embed, op=dist.ReduceOp.SUM)

        return local_embed
