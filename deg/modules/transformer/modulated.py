from typing import *
import torch
import torch.nn as nn

from deg.modules.attention.modules import RopeMultiHeadAttention
from ..attention import MultiHeadAttention
from ..norm import LayerNorm32, RMSNorm32
from .blocks import FeedForwardNet


class ModulatedTransformerBlock(nn.Module):
    """
    Transformer block (MSA + FFN) with adaptive layer norm conditioning.
    """
    def __init__(
        self,
        channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
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
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 6 * channels, bias=True)
            )

    def _forward(self, x: torch.Tensor, mod: torch.Tensor) -> torch.Tensor:
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
        h = self.norm1(x)
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        h = self.attn(h)
        h = h * gate_msa.unsqueeze(1)
        x = x + h
        h = self.norm2(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        h = self.mlp(h)
        h = h * gate_mlp.unsqueeze(1)
        x = x + h
        return x

    def forward(self, x: torch.Tensor, mod: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mod, use_reentrant=False)
        else:
            return self._forward(x, mod)


class ModulatedTransformerCrossBlock(nn.Module):
    """
    Transformer cross-attention block (MSA + MCA + FFN) with adaptive layer norm conditioning.
    """
    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
        use_shift_table: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
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
        self.mlp = FeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
        )
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 6 * channels, bias=True)
            )
        if use_shift_table:
            self.shift_table = nn.Parameter(torch.randn(1, 6 * channels) / channels**0.5)
        else:
            self.shift_table = None

    def _forward(self, x: torch.Tensor, mod: torch.Tensor, context: torch.Tensor):
        if not self.share_mod:
            mod = self.adaLN_modulation(mod)
        if self.shift_table is not None:
            mod = mod + self.shift_table.type(mod.dtype)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        h = self.norm1(x)
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        h = self.self_attn(h)
        h = h * gate_msa.unsqueeze(1)
        x = x + h
        h = self.norm2(x)
        h = self.cross_attn(h, context)
        x = x + h
        h = self.norm3(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        h = self.mlp(h)
        h = h * gate_mlp.unsqueeze(1)
        x = x + h
        return x

    def forward(self, x: torch.Tensor, mod: torch.Tensor, context: torch.Tensor):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mod, context, use_reentrant=False)
        else:
            return self._forward(x, mod, context)


class UnifiedTransformerBlock(nn.Module):
    """
    Transformer block (MSA + FFN) with optional adaptive layer norm conditioning and RoPE support.
    Used for Unified MMDiT.
    """
    def __init__(
        self,
        channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qkv_bias: bool = True,
        modulation: bool = True,
        share_mod: bool = False,
        use_shift_table: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.modulation = modulation
        self.share_mod = share_mod
        
        # If modulation is used, we usually disable elementwise_affine in LN because modulation handles it.
        self.norm1 = LayerNorm32(channels, elementwise_affine=not modulation, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=not modulation, eps=1e-6)
        
        self.attn = RopeMultiHeadAttention(
            channels,
            num_heads=num_heads,
            type="self",
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
        )
        self.mlp = FeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
        )
        
        if modulation:
            if not share_mod:
                self.adaLN_modulation = nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(channels, 6 * channels, bias=True)
                )
            if use_shift_table:
                self.shift_table = nn.Parameter(torch.randn(1, 6 * channels) / channels**0.5)
            else:
                self.shift_table = None

    def _forward(self, x: torch.Tensor, mod: Optional[torch.Tensor] = None, rotary_emb: Optional[torch.Tensor] = None):
        if self.modulation:
            assert mod is not None, "Modulation tensor must be provided when modulation is enabled"
            if not self.share_mod:
                mod = self.adaLN_modulation(mod)
            if hasattr(self, 'shift_table') and self.shift_table is not None:
                mod = mod + self.shift_table.type(mod.dtype)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
            
            h = self.norm1(x)
            h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
            h = self.attn(h, rope_emb=rotary_emb)
            h = h * gate_msa.unsqueeze(1)
            x = x + h
            
            h = self.norm2(x)
            h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
            h = self.mlp(h)
            h = h * gate_mlp.unsqueeze(1)
            x = x + h
        else:
            h = self.norm1(x)
            h = self.attn(h, rope_emb=rotary_emb)
            x = x + h
            
            h = self.norm2(x)
            h = self.mlp(h)
            x = x + h
            
        return x

    def forward(self, x: torch.Tensor, mod: Optional[torch.Tensor] = None, rotary_emb: Optional[torch.Tensor] = None):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mod, rotary_emb, use_reentrant=False)
        else:
            return self._forward(x, mod, rotary_emb)


class ModulatedTransformerCrossBlockv2(nn.Module):
    """
    Transformer cross-attention block (MSA + MCA + FFN) with adaptive layer norm conditioning.
    """
    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
        use_shift_table: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm3 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.self_attn = RopeMultiHeadAttention(
            channels,
            num_heads=num_heads,
            type="self",
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
        )
        self.cross_attn = RopeMultiHeadAttention(
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
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 6 * channels, bias=True)
            )
        if use_shift_table:
            self.shift_table = nn.Parameter(torch.randn(1, 6 * channels) / channels**0.5)
        else:
            self.shift_table = None

    def _forward(self, x: torch.Tensor, mod: torch.Tensor, context: torch.Tensor, rotary_emb: Optional[torch.Tensor] = None):
        if not self.share_mod:
            mod = self.adaLN_modulation(mod)
        if self.shift_table is not None:
            mod = mod + self.shift_table.type(mod.dtype)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        h = self.norm1(x)
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        h = self.self_attn(h, rope_emb=rotary_emb)
        h = h * gate_msa.unsqueeze(1)
        x = x + h
        h = self.norm2(x)
        h = self.cross_attn(h, context)
        x = x + h
        h = self.norm3(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        h = self.mlp(h)
        h = h * gate_mlp.unsqueeze(1)
        x = x + h
        return x

    def forward(self, x: torch.Tensor, mod: torch.Tensor, context: torch.Tensor, rotary_emb: Optional[torch.Tensor] = None):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mod, context, rotary_emb, use_reentrant=False)
        else:
            return self._forward(x, mod, context, rotary_emb)
    
class ModulatedTransformerCrossOnlyBlock(nn.Module):
    """
    Transformer cross-attention block (MSA + MCA + FFN) with adaptive layer norm conditioning.
    """
    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_checkpoint: bool = False,
        qk_rms_norm_cross: bool = True,
        qkv_bias: bool = True,
        share_mod: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.cross_attn = MultiHeadAttention(
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
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 6 * channels, bias=True)
            )

    def _forward(self, x: torch.Tensor, mod: torch.Tensor, context: torch.Tensor):
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
        h = self.norm1(x)
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        h = self.cross_attn(h, context)
        h = h * gate_msa.unsqueeze(1)
        x = x + h
        h = self.norm2(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        h = self.mlp(h)
        h = h * gate_mlp.unsqueeze(1)
        x = x + h
        return x

    def forward(self, x: torch.Tensor, mod: torch.Tensor, context: torch.Tensor):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mod, context, use_reentrant=False)
        else:
            return self._forward(x, mod, context)
