from typing import *
import torch
import torch.nn as nn
from ...modules.utils import convert_module_to_f16, convert_module_to_f32
from ...modules.transformer import AbsolutePositionEmbedder, TransformerCrossBlock, TransformerBlock
from ...modules.transformer.blocks import FeedForwardNet
from ...modules.norm import LayerNorm32
from ...modules.attention import MultiHeadAttention
from ...modules.transformer.modulated import ModulatedTransformerBlock, ModulatedTransformerCrossBlock, ModulatedTransformerCrossOnlyBlock
import numpy as np

class PcdAbsolutePositionEmbedder(nn.Module):
    """
    Embeds spatial positions in [0,1] into vector representations.
    """
    def __init__(self, channels: int, in_channels: int = 3, max_res=16):
        super().__init__()
        self.channels = channels
        self.in_channels = in_channels
        self.freq_dim = channels // in_channels // 2
        self.max_res = max_res
        freqs_2exp = torch.arange(max_res, dtype=torch.float32)
        res_dim = max(0, self.freq_dim - max_res)
        freqs_res = torch.arange(res_dim, dtype=torch.float32) / res_dim * max_res
        self.freqs = torch.cat([freqs_2exp, freqs_res], dim=0)[:self.freq_dim]
        self.freqs = 2 ** self.freqs # [1, 2 ^ max_res]
        
    def _sin_cos_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Create sinusoidal position embeddings.

        Args:
            x: a 1-D Tensor of N indices (in [0,1]^3 range)

        Returns:
            an (N, D) Tensor of positional embeddings.
        """
        self.freqs = self.freqs.to(x.device)
        out = torch.outer(x, self.freqs) * 2 * torch.pi
        out = torch.cat([torch.sin(out), torch.cos(out)], dim=-1)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): (N, D) or (B, N, D) tensor of spatial positions
        """
        *dims, D = x.shape
        assert D == self.in_channels, "Input dimension must match number of input channels"
        embed = self._sin_cos_embedding(x.reshape(-1))
        embed = embed.reshape(*dims, -1)
        if embed.shape[-1] < self.channels:
            embed = torch.cat([embed, torch.zeros(*dims, self.channels - embed.shape[-1], device=embed.device)], dim=-1)
        return embed


class PcdAbsolutePositionEmbedderv2(nn.Module):
    def __init__(self, channels: int, in_channels: int = 3, max_res: int = 10):
        super().__init__()
        self.channels = channels
        self.in_channels = in_channels
        self.freq_dim = channels // in_channels // 2
        logs = torch.linspace(0.0, float(max_res), steps=self.freq_dim, dtype=torch.float32)
        self.register_buffer("freqs", torch.pow(2.0, logs), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, D = x.shape
        assert D == self.in_channels, "Input dimension must match number of input channels"
        freqs = self.freqs.to(x.device, dtype=x.dtype)
        ang = x.unsqueeze(-1) * freqs * torch.pi
        embed = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1).reshape(N, -1)
        if embed.shape[1] < self.channels:
            embed = torch.cat([embed, torch.zeros(N, self.channels - embed.shape[1], device=embed.device, dtype=embed.dtype)], dim=-1)
        return embed


class TransformerBase(nn.Module):
    """
    Transformer without output layers.
    Serve as the base class for encoder and decoder.
    """
    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        cond_channels: Optional[int],
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4.0,
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        qk_rms_norm: bool = True,
        qk_rms_norm_cross: bool = True,
        use_2_cross_block: bool = False,
        share_mod: bool = False,
    ):
        super().__init__()
        assert not use_2_cross_block or (cond_channels is not None), "use_2_cross_block requires cond_channels"
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.use_fp16 = use_fp16
        self.use_checkpoint = use_checkpoint
        self.qk_rms_norm = qk_rms_norm
        self.qk_rms_norm_cross = qk_rms_norm_cross
        self.share_mod = share_mod
        self.use_2_cross_block = use_2_cross_block

        self.dtype = torch.float16 if use_fp16 else torch.float32

        self.input_layer = nn.Linear(in_channels, model_channels)
        self.l_embedder = None
        if cond_channels is not None:
            self.blocks = nn.ModuleList([
                    TransformerCrossBlock(
                        model_channels,
                        ctx_channels=cond_channels,
                        num_heads=self.num_heads,
                        mlp_ratio=self.mlp_ratio,
                        use_checkpoint=self.use_checkpoint,
                        qk_rms_norm=self.qk_rms_norm,
                        qk_rms_norm_cross=self.qk_rms_norm_cross,
                        use_2_cross_block=use_2_cross_block,
                    )
                    for _ in range(num_blocks)
                ])
        else:
            self.blocks = nn.ModuleList([
                TransformerBlock(
                    model_channels,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    use_checkpoint=self.use_checkpoint,
                    qk_rms_norm=self.qk_rms_norm,
                )
                for _ in range(num_blocks)
            ])

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device

    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        self.blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        self.blocks.apply(convert_module_to_f32)

    def initialize_weights(self) -> None:
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)


    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor]=None, l: Optional[torch.Tensor]=None, cond2: Optional[torch.Tensor]=None, indices=None, context_indices=None, context2_indices=None) -> torch.Tensor:
        h = self.input_layer(x)
        h = h.type(self.dtype)
        cond = cond.type(self.dtype) if cond is not None else None
        cond2 = cond2.type(self.dtype) if cond2 is not None else None
        for block in self.blocks:
            if cond is None:
                h = block(h, indices=indices)
            else:
                h = block(h, cond, context2=cond2, indices=indices, context_indices=context_indices, context2_indices=context2_indices)
        h = h.type(x.dtype)
        return h

class LevelEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256, max_period=1024):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period

    @staticmethod
    def level_embedding(t, dim, max_period=1024):
        """
        Create sinusoidal timestep embeddings.

        Args:
            t: a 1-D Tensor of N indices, one per batch element.
                These may be fractional.
            dim: the dimension of the output.
            max_period: controls the minimum frequency of the embeddings.

        Returns:
            an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -np.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None] * 2 * torch.pi
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.level_embedding(t, self.frequency_embedding_size, self.max_period)
        t_emb = self.mlp(t_freq)
        return t_emb

class ModulatedCrossOnlyTransformerBase(nn.Module):
    """
    Transformer without output layers.
    Serve as the base class for encoder and decoder.
    """
    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        cond_channels: Optional[int],
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4.0,
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        share_mod: bool = False,
        additional_level_embed: bool = False,
        qk_rms_norm_cross: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.use_fp16 = use_fp16
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.qk_rms_norm_cross = qk_rms_norm_cross
        self.dtype = torch.float16 if use_fp16 else torch.float32

        self.input_layer = nn.Linear(in_channels, model_channels)
        self.l_embedder = LevelEmbedder(model_channels)
        if additional_level_embed:
            self.l_embedder2 = LevelEmbedder(model_channels, max_period=100)
        else:
            self.l_embedder2 = None
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(model_channels, 6 * model_channels, bias=True)
            )
            
        if cond_channels is not None:
            self.blocks = nn.ModuleList([
                ModulatedTransformerCrossOnlyBlock(
                    model_channels,
                    ctx_channels=cond_channels,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    use_checkpoint=self.use_checkpoint,
                    qk_rms_norm_cross=self.qk_rms_norm_cross,
                    share_mod=self.share_mod
                )
                for _ in range(num_blocks)
            ])
        else:
            raise ValueError("ModulatedCrossOnlyTransformerBase requires cond_channels to be not None.")

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device

    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        self.blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        self.blocks.apply(convert_module_to_f32)

    def initialize_weights(self) -> None:
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        
        # Initialize timestep embedding MLP:
        nn.init.normal_(self.l_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.l_embedder.mlp[2].weight, std=0.02)
        
        if self.l_embedder2 is not None:
            # zero projection init
            nn.init.normal_(self.l_embedder2.mlp[0].weight, std=0.02)
            nn.init.constant_(self.l_embedder2.mlp[2].weight, 0)
            nn.init.constant_(self.l_embedder2.mlp[2].bias, 0)
        
        # Zero-out adaLN modulation layers in DiT blocks:
        if self.share_mod:
            nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        else:
            for block in self.blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
    def forward(self, x: torch.Tensor, l: torch.Tensor, cond: torch.Tensor, l2: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.input_layer(x)
        l = l.type(h.dtype)
        l_emb = self.l_embedder(l)
        if self.l_embedder2 is not None and l2 is not None:
            l2 = l2.type(h.dtype)
            l_emb = l_emb + self.l_embedder2(l2)
        if self.share_mod:
            l_emb = self.adaLN_modulation(l_emb)
        l_emb = l_emb.type(self.dtype)
        h = h.type(self.dtype)
        cond = cond.type(self.dtype)
        for block in self.blocks:
            h = block(h, l_emb, cond)
        h = h.type(x.dtype)
        return h
    
class ModulatedBlock(nn.Module):
    """A block that performs modulation and MLP processing."""
    def __init__(self, token_channels, mlp_ratio=4.0):
        super().__init__()
        hidden_channels = int(token_channels * mlp_ratio)
        self.norm = nn.LayerNorm(token_channels)
        self.mlp = nn.Sequential(
            nn.Linear(token_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, token_channels),
        )

    def forward(self, x, scale, shift):
        # x shape: (B, N, C)
        # scale, shift shape: (B, C)
        
        # Normalize tokens
        norm_x = self.norm(x)

        # Modulate: Apply the scale and shift
        # We unsqueeze scale and shift to broadcast them across the token dimension (N)
        # (B, N, C) * (B, 1, C) + (B, 1, C) -> (B, N, C)
        modulated_x = norm_x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        
        # Process through MLP and add residual connection
        return x + self.mlp(modulated_x)


class TokenlenModulatedMLPHead(nn.Module):
    def __init__(self, token_channels, channels_out=1, time_embedding_dim=256, max_period=32768):
        super().__init__()
        self.token_channels = token_channels
        
        # 1. Time Embedding Modules
        self.time_embedding = LevelEmbedder(time_embedding_dim, max_period=max_period)
        
        self.time_mlp = nn.Sequential(
            nn.GELU(),
            # Output will be split into scale and shift, so size is 2 * token_channels
            nn.Linear(time_embedding_dim, token_channels * 2),
        )
        
        # 2. Token Processing Block
        self.processing_block = ModulatedBlock(token_channels)
        self.norm = nn.LayerNorm(token_channels)
        self.out_proj = nn.Linear(token_channels, channels_out)

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        
        return next(self.parameters()).device
    
    
    def forward(self, tokens):
        # tokens shape: (B, N, C)
        # prepare time_scalars shape: (B)
        B, N, C = tokens.shape
        time_scalers = torch.full((B,), N, device=self.device)
        
        # A. Process time to get scale and shift
        # (B,) -> (B, time_embed_dim)
        time_emb = self.time_embedding(time_scalers)
        
        # (B, time_embed_dim) -> (B, 2 * C)
        time_params = self.time_mlp(time_emb)
        
        # Split into scale and shift
        # (B, 2 * C) -> (B, C), (B, C)
        scale, shift = time_params.chunk(2, dim=1)
        
        # B. Process tokens using the modulation
        # (B, N, C) -> (B, N, C)
        output_features = self.processing_block(tokens, scale, shift)
        
        # C. Normalize and project to out_channels
        # (B, N, C) -> (B, N, out_channels)
        output_features = self.norm(output_features)
        output_features = self.out_proj(output_features)
        
        return output_features


class TransformerDeepCrossBase(nn.Module):
    """
    Deep Cross Transformer.
    """
    def __init__(
        self,
        in_channels: int,
        in_channels_y: int,
        model_channels: int,
        cond_channels: Optional[int],
        num_blocks: int,
        num_blocks_y: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4.0,
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        qk_rms_norm: bool = True,
        qk_rms_norm_cross: bool = True,
        use_2_cross_block: bool = False,
    ):
        super().__init__()
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.num_blocks = num_blocks
        self.num_blocks_y = num_blocks_y
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.use_fp16 = use_fp16
        self.use_checkpoint = use_checkpoint
        self.qk_rms_norm = qk_rms_norm
        self.qk_rms_norm_cross = qk_rms_norm_cross
        
        self.dtype = torch.float16 if use_fp16 else torch.float32

        self.input_layer_x = nn.Linear(in_channels, model_channels)
        self.input_layer_y = nn.Linear(in_channels_y, model_channels)
        
        # blocks_y: Cross attend to cond (and cond2)
        # Use TransformerCrossBlock to handle optional cond2
        self.blocks_y = nn.ModuleList([
            TransformerCrossBlock(
                model_channels,
                ctx_channels=cond_channels,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                use_checkpoint=self.use_checkpoint,
                qk_rms_norm=self.qk_rms_norm,
                qk_rms_norm_cross=self.qk_rms_norm_cross,
                use_2_cross_block=use_2_cross_block,
            )
            for _ in range(num_blocks_y)
        ])

        # blocks_x: Cross attend to z (output of blocks_y)
        # z has model_channels, so ctx_channels=model_channels
        self.blocks_x = nn.ModuleList([
            TransformerCrossBlock(
                model_channels,
                ctx_channels=model_channels, # z channels
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                use_checkpoint=self.use_checkpoint,
                qk_rms_norm=self.qk_rms_norm,
                qk_rms_norm_cross=self.qk_rms_norm_cross,
                use_2_cross_block=False,
            )
            for _ in range(num_blocks)
        ])

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def convert_to_fp16(self) -> None:
        self.blocks_y.apply(convert_module_to_f16)
        self.blocks_x.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        self.blocks_y.apply(convert_module_to_f32)
        self.blocks_x.apply(convert_module_to_f32)

    def initialize_weights(self) -> None:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

    def forward(self, x: torch.Tensor, y: torch.Tensor, cond: torch.Tensor, cond2: Optional[torch.Tensor]=None) -> torch.Tensor:
        # x: [B, Lx, C_in_x]
        # y: [B, Ly, C_in_y]
        # cond: [B, Lc, C_cond]
        # cond2: [B, Lc2, C_cond] (Optional)
        
        # 1. Process y -> z
        h_y = self.input_layer_y(y)
        h_y = h_y.type(self.dtype)
        
        cond = cond.type(self.dtype)
        cond2 = cond2.type(self.dtype) if cond2 is not None else None
        
        for block in self.blocks_y:
            # Note: blocks_y are TransformerCrossBlock
            h_y = block(h_y, cond, context2=cond2)
            
        z = h_y # [B, Ly, model_channels]
        
        # 2. Process x -> output, using z as context
        h_x = self.input_layer_x(x)
        h_x = h_x.type(self.dtype)
        
        for block in self.blocks_x:
            h_x = block(h_x, z) # context is z
            
        h_x = h_x.type(x.dtype)
        return h_x

class CrossAttentionBlock(nn.Module):
    """
    Transformer cross-attention block.
    """
    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        qkv_bias: bool = True,
        qk_rms_norm_cross: bool = False,
    ):
        super().__init__()
        self.norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.cross_attn = MultiHeadAttention(
            channels,
            ctx_channels=ctx_channels,
            num_heads=num_heads,
            type="cross",
            qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm_cross,
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor, context_indices=None) -> torch.Tensor:
        h = self.norm1(x)
        h = self.cross_attn(h, context, context_indices=context_indices)
        x = x + h
        return x
    
    
class SelfAttentionBlock(nn.Module):
    """
    Transformer self-attention block.
    """
    def __init__(
        self,
        channels: int,
        num_heads: int,
        qkv_bias: bool = True,
        qk_rms_norm: bool = False,
    ):
        super().__init__()
        self.norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.self_attn = MultiHeadAttention(
            channels,
            num_heads=num_heads,
            type="self",
            qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = self.self_attn(h)
        x = x + h
        return x

class FFN(nn.Module):
    def __init__(
        self,
        channels: int,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, int(channels * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(channels * mlp_ratio), channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = self.mlp(h)
        x = x + h
        return x

class ProgressiveDecoderBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        cond_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_checkpoint: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        use_2_cross_block: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.use_2_cross_block = use_2_cross_block

        # 1. Update patches (y) with point features (cond)
        self.points2patch = CrossAttentionBlock(
            channels, cond_channels, num_heads,
            qk_rms_norm_cross=qk_rms_norm_cross
        )
        if use_2_cross_block:
            self.points2patch2 = CrossAttentionBlock(
                channels, cond_channels, num_heads,
                qk_rms_norm_cross=qk_rms_norm_cross
            )

        # 2. Process patches (y) with self-attention
        self.processing_layers = nn.ModuleList([
            SelfAttentionBlock(channels, num_heads, qk_rms_norm=qk_rms_norm) for _ in range(3)
        ])
        self.patch_ffn = FFN(channels, mlp_ratio)

        # 3. Update latents (x) with patches (y)
        self.patch2latents = TransformerCrossBlock(
            channels, channels, num_heads, mlp_ratio,
            use_checkpoint=use_checkpoint, qk_rms_norm=qk_rms_norm,
        )
        

    def _forward(self, x, y, cond, cond2=None):
        # x: z (latents)
        # y: patches
        # cond: point_features
        
        # 1. Update patches (y) with point features (cond)
        y = self.points2patch(y, cond)
        if self.use_2_cross_block and cond2 is not None:
            y = self.points2patch2(y, cond2)
            
        # 2. Process patches (y) with self-attention
        for sa in self.processing_layers:
            y = sa(y) 
        y = self.patch_ffn(y)
            
        # 3. Update latents (x) with patches (y)
        x = self.patch2latents(x, y)
        
        return x, y, cond, cond2

    def forward(self, x, y, cond, cond2=None):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(
                self._forward, x, y, cond, cond2, use_reentrant=False
            )
        else:
            return self._forward(x, y, cond, cond2)


class TransformerDeepCrossBasev2(nn.Module):
    """
    Deep Cross Transformer V2 based on the figure.
    """
    def __init__(
        self,
        in_channels: int,
        in_channels_y: int,
        model_channels: int,
        cond_channels: Optional[int],
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4.0,
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        qk_rms_norm: bool = True,
        qk_rms_norm_cross: bool = True,
        use_2_cross_block: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.in_channels_y = in_channels_y
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.use_fp16 = use_fp16
        self.use_checkpoint = use_checkpoint
        self.qk_rms_norm = qk_rms_norm
        self.qk_rms_norm_cross = qk_rms_norm_cross

        self.dtype = torch.float16 if use_fp16 else torch.float32

        self.input_layer_x = nn.Linear(in_channels, model_channels)
        self.input_layer_y = nn.Linear(in_channels_y, model_channels)

        self.layers = nn.ModuleList([
            ProgressiveDecoderBlock(
                channels=model_channels,
                cond_channels=cond_channels,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                use_checkpoint=self.use_checkpoint,
                qk_rms_norm=self.qk_rms_norm,
                qk_rms_norm_cross=self.qk_rms_norm_cross,
                use_2_cross_block=use_2_cross_block
            )
            for i in range(num_blocks)
        ])

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def convert_to_fp16(self) -> None:
        self.layers.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        self.layers.apply(convert_module_to_f32)

    def initialize_weights(self) -> None:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

    def forward(self, x: torch.Tensor, y: torch.Tensor, cond: torch.Tensor, cond2: Optional[torch.Tensor]=None) -> torch.Tensor:
        h_x = self.input_layer_x(x).type(self.dtype)
        h_y = self.input_layer_y(y).type(self.dtype)
        
        cond = cond.type(self.dtype)
        cond2 = cond2.type(self.dtype) if cond2 is not None else None
        
        for layer in self.layers:
            h_x, h_y, cond, cond2 = layer(h_x, h_y, cond, cond2)
            
        h_x = h_x.type(x.dtype)
        return h_x
