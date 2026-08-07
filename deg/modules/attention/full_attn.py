from typing import *
import torch
import math
from . import DEBUG, BACKEND

if BACKEND == 'xformers':
    import xformers.ops as xops
elif BACKEND == 'flash_attn':
    import flash_attn
elif BACKEND == 'sdpa':
    from torch.nn.functional import scaled_dot_product_attention as sdpa
elif BACKEND == 'naive':
    pass
else:
    raise ValueError(f"Unknown attention backend: {BACKEND}")


__all__ = [
    'scaled_dot_product_attention',
]


def _naive_sdpa(q, k, v):
    """
    Naive implementation of scaled dot product attention.
    """
    q = q.permute(0, 2, 1, 3)   # [N, H, L, C]
    k = k.permute(0, 2, 1, 3)   # [N, H, L, C]
    v = v.permute(0, 2, 1, 3)   # [N, H, L, C]
    scale_factor = 1 / math.sqrt(q.size(-1))
    attn_weight = q @ k.transpose(-2, -1) * scale_factor
    attn_weight = torch.softmax(attn_weight, dim=-1)
    out = attn_weight @ v
    out = out.permute(0, 2, 1, 3)   # [N, L, H, C]
    return out


@overload
def scaled_dot_product_attention(qkv: torch.Tensor) -> torch.Tensor:
    """
    Apply scaled dot product attention.

    Args:
        qkv (torch.Tensor): A [N, L, 3, H, C] tensor containing Qs, Ks, and Vs.
    """
    ...

@overload
def scaled_dot_product_attention(q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
    """
    Apply scaled dot product attention.

    Args:
        q (torch.Tensor): A [N, L, H, C] tensor containing Qs.
        kv (torch.Tensor): A [N, L, 2, H, C] tensor containing Ks and Vs.
    """
    ...

@overload
def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Apply scaled dot product attention.

    Args:
        q (torch.Tensor): A [N, L, H, Ci] tensor containing Qs.
        k (torch.Tensor): A [N, L, H, Ci] tensor containing Ks.
        v (torch.Tensor): A [N, L, H, Co] tensor containing Vs.

    Note:
        k and v are assumed to have the same coordinate map.
    """
    ...

def scaled_dot_product_attention(*args, **kwargs):
    arg_names_dict = {
        1: ['qkv'],
        2: ['q', 'kv'],
        3: ['q', 'k', 'v']
    }
    num_all_args = len(args) + len(kwargs)
    assert num_all_args in arg_names_dict, f"Invalid number of arguments, got {num_all_args}, expected 1, 2, or 3"
    for key in arg_names_dict[num_all_args][len(args):]:
        assert key in kwargs, f"Missing argument {key}"

    if num_all_args == 1:
        qkv = args[0] if len(args) > 0 else kwargs['qkv']
        assert len(qkv.shape) == 5 and qkv.shape[2] == 3, f"Invalid shape for qkv, got {qkv.shape}, expected [N, L, 3, H, C]"
        device = qkv.device

    elif num_all_args == 2:
        q = args[0] if len(args) > 0 else kwargs['q']
        kv = args[1] if len(args) > 1 else kwargs['kv']
        assert q.shape[0] == kv.shape[0], f"Batch size mismatch, got {q.shape[0]} and {kv.shape[0]}"
        assert len(q.shape) == 4, f"Invalid shape for q, got {q.shape}, expected [N, L, H, C]"
        assert len(kv.shape) == 5, f"Invalid shape for kv, got {kv.shape}, expected [N, L, 2, H, C]"
        device = q.device

    elif num_all_args == 3:
        q = args[0] if len(args) > 0 else kwargs['q']
        k = args[1] if len(args) > 1 else kwargs['k']
        v = args[2] if len(args) > 2 else kwargs['v']
        assert q.shape[0] == k.shape[0] == v.shape[0], f"Batch size mismatch, got {q.shape[0]}, {k.shape[0]}, and {v.shape[0]}"
        assert len(q.shape) == 4, f"Invalid shape for q, got {q.shape}, expected [N, L, H, Ci]"
        assert len(k.shape) == 4, f"Invalid shape for k, got {k.shape}, expected [N, L, H, Ci]"
        assert len(v.shape) == 4, f"Invalid shape for v, got {v.shape}, expected [N, L, H, Co]"
        device = q.device    

    if BACKEND == 'xformers':
        if num_all_args == 1:
            q, k, v = qkv.unbind(dim=2)
        elif num_all_args == 2:
            k, v = kv.unbind(dim=2)
        out = xops.memory_efficient_attention(q, k, v)
    elif BACKEND == 'flash_attn':
        if num_all_args == 1:
            out = flash_attn.flash_attn_qkvpacked_func(qkv)
        elif num_all_args == 2:
            out = flash_attn.flash_attn_kvpacked_func(q, kv)
        elif num_all_args == 3:
            out = flash_attn.flash_attn_func(q, k, v)
    elif BACKEND == 'sdpa':
        if num_all_args == 1:
            q, k, v = qkv.unbind(dim=2)
        elif num_all_args == 2:
            k, v = kv.unbind(dim=2)
        q = q.permute(0, 2, 1, 3)   # [N, H, L, C]
        k = k.permute(0, 2, 1, 3)   # [N, H, L, C]
        v = v.permute(0, 2, 1, 3)   # [N, H, L, C]
        out = sdpa(q, k, v)         # [N, H, L, C]
        out = out.permute(0, 2, 1, 3)   # [N, L, H, C]
    elif BACKEND == 'naive':
        if num_all_args == 1:
            q, k, v = qkv.unbind(dim=2)
        elif num_all_args == 2:
            k, v = kv.unbind(dim=2)
        out = _naive_sdpa(q, k, v)
    else:
        raise ValueError(f"Unknown attention module: {BACKEND}")
    
    return out

def knn_attention(q, k, v, knn_idx):
    """
    FlashAttention over k-NN neighborhoods.

    Args:
        q: (N, H, Dh) query tokens
        k: (N, H, Dh) key tokens
        v: (N, H, Dh) value tokens
        knn_idx: (N, K) neighbor indices for each query point

    Returns:
        out: (N, H, Dh) attended features
    """
    device = q.device
    N, H, Dh = q.shape
    K = knn_idx.shape[1]

    if BACKEND == 'flash_attn':
        from flash_attn.flash_attn_interface import flash_attn_varlen_kvpacked_func
        # Gather neighbors
        k_nb = k[knn_idx]  # (N, K, H, Dh)
        v_nb = v[knn_idx]  # (N, K, H, Dh)

        # Flatten neighbors
        k_flat = k_nb.reshape(N * K, H, Dh).contiguous()
        v_flat = v_nb.reshape(N * K, H, Dh).contiguous()
        kv_packed = torch.stack([k_flat, v_flat], dim=1)  # (N*K, 2, H, Dh)

        # Q is already (N, H, Dh)
        q_packed = q.contiguous()

        # Variable-length sequence offsets
        cu_seqlens_q = torch.arange(0, N + 1, 1, device=device, dtype=torch.int32)
        cu_seqlens_kv = torch.arange(0, N * K + 1, K, device=device, dtype=torch.int32)

        # Run FlashAttention varlen kernel
        out = flash_attn_varlen_kvpacked_func(
            q_packed, kv_packed, cu_seqlens_q, cu_seqlens_kv,
            max_seqlen_q=1, max_seqlen_kv=K,
            dropout_p=0.0, causal=False, softmax_scale=None
        )
    else:
        raise NotImplementedError("k-NN FlashAttention is only implemented for the FlashAttention backend.")
    print("using  knn!") # DEBUG
    return out  # (N, H, Dh)
