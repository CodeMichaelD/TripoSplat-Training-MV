from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from .full_attn import scaled_dot_product_attention
from ...modules.norm import LayerNorm32

class MultiHeadRMSNorm(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (F.normalize(x.float(), dim = -1) * self.gamma * self.scale).to(x.dtype)


class RotaryPositionEmbedder(nn.Module):
    def __init__(self, hidden_size: int, in_channels: int = 3):
        super().__init__()
        assert hidden_size % 2 == 0, "Hidden size must be divisible by 2"
        self.hidden_size = hidden_size
        self.in_channels = in_channels
        self.freq_dim = hidden_size // in_channels // 2
        self.freqs = torch.arange(self.freq_dim, dtype=torch.float32) / self.freq_dim
        self.freqs = 1.0 / (10000 ** self.freqs)
        
    def _get_phases(self, indices: torch.Tensor) -> torch.Tensor:
        self.freqs = self.freqs.to(indices.device)
        phases = torch.outer(indices, self.freqs)
        phases = torch.polar(torch.ones_like(phases), phases)
        return phases
        
    def _rotary_embedding(self, x: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        x_rotated = x_complex * phases
        x_embed = torch.view_as_real(x_rotated).reshape(*x_rotated.shape[:-1], -1).to(x.dtype)
        return x_embed
        
    def forward(self, q: torch.Tensor, k: torch.Tensor, indices: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if indices is None:
            indices = torch.arange(q.shape[-2], device=q.device)
            if len(q.shape) > 2:
                indices = indices.unsqueeze(0).expand(q.shape[:-2] + (-1,))
        
        phases = self._get_phases(indices.reshape(-1)).reshape(*indices.shape[:-1], -1)
        if phases.shape[-1] < self.hidden_size // 2:
            phases = torch.cat([phases, torch.polar(
                torch.ones(*phases.shape[:-1], self.hidden_size // 2 - phases.shape[1], device=phases.device),
                torch.zeros(*phases.shape[:-1], self.hidden_size // 2 - phases.shape[1], device=phases.device)
            )], dim=-1)
        q_embed = self._rotary_embedding(q, phases)
        k_embed = self._rotary_embedding(k, phases)
        return q_embed, k_embed
    
    def apply_rope(self, x: torch.Tensor, indices: Optional[torch.Tensor]) -> torch.Tensor:
        if indices is None:
            raise ValueError()
        phases = self._get_phases(indices.reshape(-1)).reshape(*indices.shape[:-1], -1)
        if phases.shape[-1] < self.hidden_size // 2:
            phases = torch.cat([phases, torch.polar(
                torch.ones(*phases.shape[:-1], self.hidden_size // 2 - phases.shape[-1], device=phases.device),
                torch.zeros(*phases.shape[:-1], self.hidden_size // 2 - phases.shape[-1], device=phases.device)
            )], dim=-1)
        return self._rotary_embedding(x, phases)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        ctx_channels: Optional[int]=None,
        type: Literal["self", "cross"] = "self",
        qkv_bias: bool = True,
        qk_rms_norm: bool = False,
    ):
        super().__init__()
        assert channels % num_heads == 0
        assert type in ["self", "cross"], f"Invalid attention type: {type}"

        self.channels = channels
        self.head_dim = channels // num_heads
        self.ctx_channels = ctx_channels if ctx_channels is not None else channels
        self.num_heads = num_heads
        self._type = type
        self.qk_rms_norm = qk_rms_norm

        if self._type == "self":
            self.to_qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        else:
            self.to_q = nn.Linear(channels, channels, bias=qkv_bias)
            self.to_kv = nn.Linear(self.ctx_channels, channels * 2, bias=qkv_bias)

        if self.qk_rms_norm:
            self.q_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
            self.k_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)

        self.to_out = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, indices: Optional[torch.Tensor] = None, context_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, C = x.shape
        if self._type == "self":
            qkv = self.to_qkv(x)
            qkv = qkv.reshape(B, L, 3, self.num_heads, -1)
            if self.qk_rms_norm:
                q, k, v = qkv.unbind(dim=2)
                q = self.q_rms_norm(q)
                k = self.k_rms_norm(k)
                qkv = torch.stack([q, k, v], dim=2)
            h = scaled_dot_product_attention(qkv)
        else:
            Lkv = context.shape[1]
            q = self.to_q(x)
            kv = self.to_kv(context)
            q = q.reshape(B, L, self.num_heads, -1)
            kv = kv.reshape(B, Lkv, 2, self.num_heads, -1)
            if self.qk_rms_norm:
                q = self.q_rms_norm(q)
                k, v = kv.unbind(dim=2)
                k = self.k_rms_norm(k)
                h = scaled_dot_product_attention(q, k, v)
            else:
                h = scaled_dot_product_attention(q, kv)
        h = h.reshape(B, L, -1)
        h = self.to_out(h)
        return h

  

def apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary embeddings to input tensors using the given frequencies.
    Args:
        hidden_states: [B, L, H, D]
        freqs: [B, L, 1, D/2] complex tensor
    """
    # [B, L, H, D] -> [B, L, H, D/2, 2] -> [B, L, H, D/2] (complex)
    x_rotated = torch.view_as_complex(hidden_states.float().reshape(*hidden_states.shape[:-1], -1, 2))
    
    # [B, L, H, D/2] * [B, L, 1, D/2] (broadcast) -> [B, L, H, D/2]
    x_rotated = x_rotated * freqs
    
    # [B, L, H, D/2] -> [B, L, H, D/2, 2] -> [B, L, H, D]
    x_out = torch.view_as_real(x_rotated).reshape(*x_rotated.shape[:-1], -1)
    return x_out.type_as(hidden_states)

def clamp_mul(x, f):
    f_t = f.tanh()
    return x * f_t + x.detach() * (f - f_t)

class RePo3DRotaryEmbedding(nn.Module):
    def __init__(self, model_channels: int, num_heads: int, head_dim: int, repo_hidden_ratio: float = 0.125, max_freq: float = 16.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        repo_hidden_size = int(model_channels * repo_hidden_ratio)
        
        # Lightweight Residual Network
        self.norm = LayerNorm32(model_channels)
        self.gate_map = nn.Linear(model_channels, repo_hidden_size, bias=False)
        self.content_map = nn.Linear(model_channels, repo_hidden_size, bias=False)
        self.act = nn.SiLU()
        self.final_map = nn.Linear(repo_hidden_size, 3 * num_heads, bias=False)
        
        # Split head_dim into 3 parts for x, y, z (same as Rotary3DPoseEmbedding)
        self.dim_0 = 2 * (head_dim // 6)
        self.dim_1 = 2 * (head_dim // 6)
        self.dim_2 = head_dim - self.dim_0 - self.dim_1
        self.max_freq = max_freq
        
        dims = [self.dim_0, self.dim_1, self.dim_2]
        freqs_list = []
        for d in dims:
            freq_dim = d // 2
            freqs = torch.linspace(1.0, float(max_freq), steps=freq_dim, dtype=torch.float32)
            freqs_list.append(freqs)
        
        self.freqs_0 = nn.Parameter(freqs_list[0])
        self.freqs_1 = nn.Parameter(freqs_list[1])
        self.freqs_2 = nn.Parameter(freqs_list[2])
        
    def initialize_weights(self):
        # Initialize proj_out with small values
        nn.init.constant_(self.final_map.weight, 0)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: [B, L, C]
        # base_rotary_emb: [1, L, 1, D/2] (or similar broadcastable shape)
        # pos: [B, L, 3] original 3D coordinates
        hidden_states = hidden_states.float()
        # Lightweight fusion logic
        h = self.norm(hidden_states)
        feat = self.act(self.gate_map(h)) * self.content_map(h)
        out = self.final_map(feat)
        
        # Reshape to [B, L, 3 * H] -> [B, L, H, 3]
        B, L, _ = out.shape
        delta_pos = out.reshape(B, L, self.num_heads, 3)
        
        # Compute angles for delta (rope like)
        # delta_pos[..., 0] is [B, L, H]
        # freqs_0 is [D0/2]
        # We want [B, L, H, D0/2]
        # delta_pos gradient scale clamping, avoid large freq gradient exploding
        # forward equivalent to delta_pos * self.freqs_I * torch.pi
        ang_0 = clamp_mul(delta_pos[..., 0].unsqueeze(-1), self.freqs_0) * torch.pi
        ang_1 = clamp_mul(delta_pos[..., 1].unsqueeze(-1), self.freqs_1) * torch.pi
        ang_2 = clamp_mul(delta_pos[..., 2].unsqueeze(-1), self.freqs_2) * torch.pi
        
        # Concatenate angles [B, L, H, D/2]
        ang = torch.cat([ang_0, ang_1, ang_2], dim=-1)
        
        # Complex rotation for delta
        delta_freqs_cis = torch.polar(torch.ones_like(ang), ang)
        
        # Apply residual rotation: base * delta
        # base_rotary_emb will broadcast to [B, L, H, D/2]
        return delta_freqs_cis.type(torch.complex64)


class Rotary3DPoseEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_res: int = 10):
        super().__init__()
        self.head_dim = head_dim
        
        # Split head_dim into 3 parts for x, y, z
        # Each part must be divisible by 2 for complex RoPE
        self.dim_0 = 2 * (head_dim // 6)
        self.dim_1 = 2 * (head_dim // 6)
        self.dim_2 = head_dim - self.dim_0 - self.dim_1
        
        dims = [self.dim_0, self.dim_1, self.dim_2]
        freqs_list = []
        for d in dims:
            freq_dim = d // 2
            logs = torch.linspace(0.0, float(max_res), steps=freq_dim, dtype=torch.float32)
            freqs = torch.pow(2.0, logs)
            freqs_list.append(freqs)
            
        self.register_buffer("freqs_0", freqs_list[0])
        self.register_buffer("freqs_1", freqs_list[1])
        self.register_buffer("freqs_2", freqs_list[2])

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pos: [B, L, 3] tensor of spatial positions in [0, 1]
        Returns:
            rope_emb: [B, L, 1, D/2] complex tensor
        """
        # pos[..., 0]: [B, L]
        # freqs_0: [D0/2]
        # ang: [B, L, D0/2]
        
        ang_0 = torch.outer(pos[..., 0].flatten(), self.freqs_0).view(*pos.shape[:-1], -1) * torch.pi
        ang_1 = torch.outer(pos[..., 1].flatten(), self.freqs_1).view(*pos.shape[:-1], -1) * torch.pi
        ang_2 = torch.outer(pos[..., 2].flatten(), self.freqs_2).view(*pos.shape[:-1], -1) * torch.pi
        
        # Concatenate angles [B, L, D/2]
        ang = torch.cat([ang_0, ang_1, ang_2], dim=-1)
        
        # Convert to complex exponentials
        freqs_cis = torch.polar(torch.ones_like(ang), ang)
        
        # Add head dimension for broadcasting: [B, L, 1, D/2]
        return freqs_cis.unsqueeze(-2).type(torch.complex64)


class RopeMultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        ctx_channels: Optional[int]=None,
        type: Literal["self", "cross"] = "self",
        qkv_bias: bool = True,
        qk_rms_norm: bool = False,
        use_rope: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.ctx_channels = ctx_channels if ctx_channels is not None else channels
        self._type = type
        self.qk_rms_norm = qk_rms_norm
        self.use_rope = use_rope
        
        assert channels % num_heads == 0

        if self._type == "self":
            self.qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        else:
            self.q = nn.Linear(channels, channels, bias=qkv_bias)
            self.kv = nn.Linear(self.ctx_channels, channels * 2, bias=qkv_bias)
        
        if self.qk_rms_norm:
            self.q_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
            self.k_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
            
        self.out = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, rope_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [B, L, C] query tensor
            context: [B, S, C] context tensor (for cross attention)
            rope_emb: [B, L, 1, D/2] precomputed complex RoPE embedding
        """
        B, L, C = x.shape
        
        if self._type == "self":
            # [B, L, 3, H, D]
            qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
            q, k, v = qkv.unbind(2) # [B, L, H, D]
            
            if self.use_rope:
                q = apply_rotary_emb(q, rope_emb)
                k = apply_rotary_emb(k, rope_emb)
        else:
            # Cross attention
            q = self.q(x).reshape(B, L, self.num_heads, self.head_dim)
            if context is None:
                raise ValueError("Context must be provided for cross attention")
            kv = self.kv(context).reshape(B, context.shape[1], 2, self.num_heads, self.head_dim)
            k, v = kv.unbind(2)
            # No RoPE for cross attention per instructions
            
        if self.qk_rms_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
            
        h = scaled_dot_product_attention(q, k, v)
        
        # [B, L, H, D] -> [B, L, C]
        h = h.reshape(B, L, C)
        return self.out(h)
