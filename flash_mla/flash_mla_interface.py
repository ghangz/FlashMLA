# Adapted from deepseek-ai/FlashMLA(https://github.com/deepseek-ai/FlashMLA)
from typing import Optional, Tuple

import torch

# import flash_mla_cuda
import flash_mla_cuda as flash_mla


def _check_int32_tensor(name: str, tensor: torch.Tensor) -> None:
    if tensor.dtype != torch.int32:
        raise TypeError(f"{name} must use torch.int32, got {tensor.dtype}")


def _validate_flash_mla_inputs(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int,
    tile_scheduler_metadata: torch.Tensor,
    num_splits: torch.Tensor,
) -> None:
    for name, tensor in (
        ("q", q),
        ("k_cache", k_cache),
        ("block_table", block_table),
        ("cache_seqlens", cache_seqlens),
        ("tile_scheduler_metadata", tile_scheduler_metadata),
        ("num_splits", num_splits),
    ):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor)}")
        if tensor.device != q.device:
            raise ValueError(
                f"All tensors must be on the same device, but {name} is on {tensor.device} "
                f"while q is on {q.device}"
            )
    if q.dim() != 4:
        raise ValueError(f"q must be 4D, got shape {tuple(q.shape)}")
    if k_cache.dim() != 4:
        raise ValueError(f"k_cache must be 4D, got shape {tuple(k_cache.shape)}")
    if block_table.dim() != 2:
        raise ValueError(f"block_table must be 2D, got shape {tuple(block_table.shape)}")
    if cache_seqlens.dim() != 1:
        raise ValueError(
            f"cache_seqlens must be 1D, got shape {tuple(cache_seqlens.shape)}"
        )
    if num_splits.dim() != 1:
        raise ValueError(f"num_splits must be 1D, got shape {tuple(num_splits.shape)}")
    if (
        q.shape[0] != block_table.shape[0]
        or q.shape[0] != cache_seqlens.shape[0]
        or num_splits.shape[0] != q.shape[0] + 1
    ):
        raise ValueError(
            f"batch size mismatch: q batch_size is {q.shape[0]}, but block_table has "
            f"{block_table.shape[0]}, cache_seqlens has {cache_seqlens.shape[0]}, and "
            f"num_splits must have size {q.shape[0] + 1}, got {num_splits.shape[0]}"
        )
    if q.shape[-1] != k_cache.shape[-1]:
        raise ValueError(
            f"q head_dim ({q.shape[-1]}) must match k_cache head_dim ({k_cache.shape[-1]})"
        )
    if head_dim_v <= 0 or head_dim_v > k_cache.shape[-1]:
        raise ValueError(
            f"head_dim_v must be in (0, {k_cache.shape[-1]}], got {head_dim_v}"
        )
    _check_int32_tensor("block_table", block_table)
    _check_int32_tensor("cache_seqlens", cache_seqlens)
    _check_int32_tensor("tile_scheduler_metadata", tile_scheduler_metadata)
    _check_int32_tensor("num_splits", num_splits)


def get_mla_metadata(
    cache_seqlens: torch.Tensor,
    num_heads_per_head_k: int,
    num_heads_k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Arguments:
        cache_seqlens: (batch_size), dtype torch.int32.
        num_heads_per_head_k: Equals to seq_len_q * num_heads_q // num_heads_k.
        num_heads_k: num_heads_k.

    Return:
        tile_scheduler_metadata: (num_sm_parts, TileSchedulerMetaDataSize), dtype torch.int32.
        num_splits: (batch_size + 1), dtype torch.int32.
    """
    return flash_mla.get_mla_metadata(cache_seqlens, num_heads_per_head_k, num_heads_k)


def flash_mla_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int,
    tile_scheduler_metadata: torch.Tensor,
    num_splits: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Arguments:
        q: (batch_size, seq_len_q, num_heads_q, head_dim).
        k_cache: (num_blocks, page_block_size, num_heads_k, head_dim).
        block_table: (batch_size, max_num_blocks_per_seq), torch.int32.
        cache_seqlens: (batch_size), torch.int32.
        head_dim_v: Head_dim of v.
        tile_scheduler_metadata: (num_sm_parts, TileSchedulerMetaDataSize), torch.int32, return by get_mla_metadata.
        num_splits: (batch_size + 1), torch.int32, return by get_mla_metadata.
        softmax_scale: float. The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim).
        causal: bool. Whether to apply causal attention mask.

    Return:
        out: (batch_size, seq_len_q, num_heads_q, head_dim_v).
        softmax_lse: (batch_size, num_heads_q, seq_len_q), torch.float32.
    """
    _validate_flash_mla_inputs(
        q,
        k_cache,
        block_table,
        cache_seqlens,
        head_dim_v,
        tile_scheduler_metadata,
        num_splits,
    )
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    out, softmax_lse = flash_mla.fwd_kvcache_mla(
        q,
        k_cache,
        None,
        head_dim_v,
        cache_seqlens,
        block_table,
        softmax_scale,
        causal,
        tile_scheduler_metadata,
        num_splits,
    )
    return out, softmax_lse
