from typing import *
import torch
import torch.nn as nn
from ..attention import MultiHeadAttention
from ..norm import LayerNorm32


class AbsolutePositionEmbedder(nn.Module):
    """
    Embeds spatial positions into vector representations.
    """
    def __init__(self, channels: int, in_channels: int = 3):
        super().__init__()
        self.channels = channels
        self.in_channels = in_channels
        self.freq_dim = channels // in_channels // 2
        self.freqs = torch.arange(self.freq_dim, dtype=torch.float32) / self.freq_dim
        self.freqs = 1.0 / (10000 ** self.freqs)
        
    def _sin_cos_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Create sinusoidal position embeddings.

        Args:
            x: a 1-D Tensor of N indices

        Returns:
            an (N, D) Tensor of positional embeddings.
        """
        self.freqs = self.freqs.to(x.device)
        out = torch.outer(x, self.freqs)
        out = torch.cat([torch.sin(out), torch.cos(out)], dim=-1)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): (N, D) tensor of spatial positions
        """
        N, D = x.shape
        assert D == self.in_channels, "Input dimension must match number of input channels"
        embed = self._sin_cos_embedding(x.reshape(-1))
        embed = embed.reshape(N, -1)
        if embed.shape[1] < self.channels:
            embed = torch.cat([embed, torch.zeros(N, self.channels - embed.shape[1], device=embed.device)], dim=-1)
        return embed


class FeedForwardNet(nn.Module):
    def __init__(self, channels: int, mlp_ratio: float = 4.0, channels_out: Optional[int] = None):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channels, int(channels * mlp_ratio)),
            nn.GELU(approximate="tanh"),
            nn.Linear(int(channels * mlp_ratio), channels if channels_out is None else channels_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

class TransformerBlock(nn.Module):
    """
    Transformer block (MSA + FFN).
    """
    def __init__(
        self,
        channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_checkpoint: bool = False,
        qk_rms_norm: bool = False,
        qkv_bias: bool = True,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.attn = MultiHeadAttention(
            channels,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm,
        )
        self.mlp = FeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
        )

    def _forward(self, x: torch.Tensor, indices=None) -> torch.Tensor:
        h = self.norm1(x)
        h = self.attn(h, indices=indices)
        x = x + h
        h = self.norm2(x)
        h = self.mlp(h)
        x = x + h
        return x

    def forward(self, x: torch.Tensor, indices=None) -> torch.Tensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, indices, use_reentrant=False)
        else:
            return self._forward(x, indices=indices)


class TransformerCrossBlock(nn.Module):
    """
    Transformer cross-attention block (MSA + MCA + FFN).
    """
    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_checkpoint: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        qkv_bias: bool = True,
        use_2_cross_block: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.use_2_cross_block = use_2_cross_block
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm3 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.self_attn = MultiHeadAttention(
            channels,
            num_heads=num_heads,
            type="self",
            qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm,
        )
        self.cross_attn = MultiHeadAttention(
            channels,
            ctx_channels=ctx_channels,
            num_heads=num_heads,
            type="cross",
            qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm_cross,
        )
        if self.use_2_cross_block:
            self.norm2_1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
            self.cross_attn2 = MultiHeadAttention(
                channels,
                ctx_channels=ctx_channels,
                num_heads=num_heads,
                type="cross",
                qkv_bias=qkv_bias,
                qk_rms_norm=qk_rms_norm_cross,
            )
        self.mlp = FeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
        )

    def _forward(self, x: torch.Tensor, context: torch.Tensor, context2=None, indices=None, context_indices=None, context2_indices=None) -> torch.Tensor:
        h = self.norm1(x)
        h = self.self_attn(h, indices=indices)
        x = x + h
        h = self.norm2(x)
        h = self.cross_attn(h, context, context_indices=context_indices)
        x = x + h
        if self.use_2_cross_block:
            h = self.norm2_1(x)
            h = self.cross_attn2(h, context2, context_indices=context2_indices)
            x = x + h
        h = self.norm3(x)
        h = self.mlp(h)
        x = x + h
        return x

    def forward(self, x: torch.Tensor, context: torch.Tensor, context2=None, indices=None, context_indices=None, context2_indices=None) -> torch.Tensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, context, context2, indices, context_indices, context2_indices, use_reentrant=False)
        else:
            return self._forward(x, context, context2=context2, indices=indices, context_indices=context_indices, context2_indices=context2_indices)
