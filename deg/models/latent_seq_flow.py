# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_wan.py
# The original license is as follows:
# 
# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tensordict import TensorDict

from ..modules.mlp import MLP
from ..modules.transformer.modulated import UnifiedTransformerBlock
from ..modules.attention.modules import RePo3DRotaryEmbedding, Rotary3DPoseEmbedding
from ..modules.utils import convert_module_to_f16, convert_module_to_f32
from .gs_seqence_vae.base import PcdAbsolutePositionEmbedder

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
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
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class LatentSeqMMFlowModel(nn.Module):
    """
    Latent Sequence Multimodal Flow Model (Unified MMDiT).
    Adapts ZImageTransformer2DModel architecture for 3D-Image unification.
    MMDiT adapted from diffusers/models/transformers/transformer_z_image.py
    REPO adapted from https://github.com/SakanaAI/repo
    """
    def __init__(
        self,
        q_token_length: int,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        out_channels: int,
        num_blocks: int,
        num_refiner_blocks: int = 2,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        cam_channels: Optional[int] = None,
        cond2_channels: Optional[int] = None,
        ctrl_channels: Optional[int] = None,
        mlp_ratio: float = 4,
        pe_mode: Literal["3d_ot_rope", "3d_ot_repo", "3d_ot_ape", "repo", "learnable", None] = "3d_ot_ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        share_mod: bool = True,
        qk_rms_norm: bool = False,
        use_shift_table: bool = False,
    ):
        super().__init__()
        self.q_token_length = q_token_length
        self.in_channels = in_channels
        self.cam_channels = cam_channels
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.cond2_channels = cond2_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_refiner_blocks = num_refiner_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.pe_mode = pe_mode
        self.use_fp16 = use_fp16
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.qk_rms_norm = qk_rms_norm
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.use_shift_table = use_shift_table

        self.t_embedder = TimestepEmbedder(model_channels)
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(model_channels, 6 * model_channels, bias=True)
            )

        self.input_layer = nn.Linear(in_channels, model_channels)
        self.cond_embedder = nn.Linear(cond_channels, model_channels)
        if cond2_channels is not None:
            self.cond_embedder2 = nn.Linear(cond2_channels, model_channels)
        else:
            self.cond_embedder2 = None

        self.ctrl_channels = ctrl_channels
        if ctrl_channels is not None:
            self.ctrl_embedder = nn.Linear(ctrl_channels, model_channels)
            # Zero-init so untrained adapter does nothing initially
            nn.init.zeros_(self.ctrl_embedder.weight)
            if self.ctrl_embedder.bias is not None:
                nn.init.zeros_(self.ctrl_embedder.bias)
        else:
            self.ctrl_embedder = None
        
        # Positional Embeddings
        self.use_rope = (self.pe_mode in ["3d_ot_rope", "3d_ot_repo", "repo"])
        if pe_mode == "3d_ot_rope":
            sobol_seq = torch.quasirandom.SobolEngine(dimension=3, scramble=True, seed=123).draw(q_token_length)
            self.register_buffer("pos_pe", sobol_seq[None])
            self.pos_embedder = Rotary3DPoseEmbedding(num_head_channels)
            self.cond_pos_embedder = None
        elif pe_mode == "3d_ot_ape":
            sobol_seq = torch.quasirandom.SobolEngine(dimension=3, scramble=True, seed=123).draw(q_token_length)
            self.register_buffer("pos_pe", sobol_seq[None])
            self.pos_embedder = PcdAbsolutePositionEmbedder(model_channels)
            self.cond_pos_embedder = None
        elif pe_mode == "repo":
            self.pos_embedder = None
            self.cond_pos_embedder = None
            self.noise_repo_layers = nn.ModuleList([
                RePo3DRotaryEmbedding(model_channels, num_heads=self.num_heads, head_dim=num_head_channels)
                for _ in range(num_refiner_blocks)
            ])
            self.context_repo_layers = nn.ModuleList([
                RePo3DRotaryEmbedding(model_channels, num_heads=self.num_heads, head_dim=num_head_channels)
                for _ in range(num_refiner_blocks)
            ])
            self.repo_layers = nn.ModuleList([
                RePo3DRotaryEmbedding(model_channels, num_heads=self.num_heads, head_dim=num_head_channels)
                for _ in range(num_blocks)
            ])
        elif pe_mode == "3d_ot_repo":
            sobol_seq = torch.quasirandom.SobolEngine(dimension=3, scramble=True, seed=123).draw(q_token_length)
            self.register_buffer("pos_pe", sobol_seq[None])
            self.pos_embedder = PcdAbsolutePositionEmbedder(model_channels)
            self.cond_pos_embedder = None
            self.noise_repo_layers = nn.ModuleList([
                RePo3DRotaryEmbedding(model_channels, num_heads=self.num_heads, head_dim=num_head_channels)
                for _ in range(num_refiner_blocks)
            ])
            self.context_repo_layers = nn.ModuleList([
                RePo3DRotaryEmbedding(model_channels, num_heads=self.num_heads, head_dim=num_head_channels)
                for _ in range(num_refiner_blocks)
            ])
            self.repo_layers = nn.ModuleList([
                RePo3DRotaryEmbedding(model_channels, num_heads=self.num_heads, head_dim=num_head_channels)
                for _ in range(num_blocks)
            ])
        elif pe_mode == "learnable":
            # Not fully implemented adaptation for learnable
            self.pos_pe = None
            self.pos_embedder = None
        else:
            self.pos_pe = None
            self.pos_embedder = None

        # Refiners
        self.noise_refiner = nn.ModuleList([
            UnifiedTransformerBlock(
                model_channels,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                use_checkpoint=self.use_checkpoint,
                use_rope=self.use_rope,
                qk_rms_norm=self.qk_rms_norm,
                modulation=True,
                share_mod=self.share_mod,
                use_shift_table=self.use_shift_table,
            )
            for _ in range(num_refiner_blocks)
        ])
        
        self.context_refiner = nn.ModuleList([
            UnifiedTransformerBlock(
                model_channels,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                use_checkpoint=self.use_checkpoint,
                use_rope=self.use_rope,
                qk_rms_norm=self.qk_rms_norm,
                modulation=False,
            )
            for _ in range(num_refiner_blocks)
        ])
        if self.cam_channels is not None:
            self.cam_refiner = MLP(
                self.cam_channels,
                model_channels,
                model_channels,
                mlp_layer_num=num_refiner_blocks,
            )
        
        self.blocks = nn.ModuleList([
            UnifiedTransformerBlock(
                model_channels,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                use_checkpoint=self.use_checkpoint,
                use_rope=self.use_rope,
                qk_rms_norm=self.qk_rms_norm,
                modulation=True,
                share_mod=self.share_mod,
                use_shift_table=self.use_shift_table,
            )
            for _ in range(num_blocks)
        ])

        if self.use_shift_table:
            self.shift_table = nn.Parameter(torch.randn(1, 2, model_channels) / model_channels**0.5)
        else:
            self.shift_table = None

        self.out_layer = nn.Linear(model_channels, out_channels)
        if cam_channels is not None:
            self.cam_out_layer = nn.Linear(model_channels, cam_channels)
        
        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def convert_to_fp16(self) -> None:
        self.blocks.apply(convert_module_to_f16)
        self.noise_refiner.apply(convert_module_to_f16)
        self.context_refiner.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        self.blocks.apply(convert_module_to_f32)
        self.noise_refiner.apply(convert_module_to_f32)
        self.context_refiner.apply(convert_module_to_f32)

    def initialize_weights(self) -> None:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        if self.share_mod:
            nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        else:
            for block in self.blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            for block in self.noise_refiner:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        if self.use_shift_table:
            for block in self.blocks:
                nn.init.constant_(block.shift_table, 0)
            for block in self.noise_refiner:
                nn.init.constant_(block.shift_table, 0)
            nn.init.constant_(self.shift_table, 0)
            
        if hasattr(self, 'repo_layers'):
            for block in self.noise_repo_layers:
                block.initialize_weights()
            for block in self.context_repo_layers:
                block.initialize_weights()
            for block in self.repo_layers:
                block.initialize_weights()
                
        if self.cond_embedder2 is not None:
            nn.init.constant_(self.cond_embedder2.weight, 0)
            nn.init.constant_(self.cond_embedder2.bias, 0)

        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)

        if self.cam_channels is not None:
            nn.init.constant_(self.cam_out_layer.weight, 0)
            nn.init.constant_(self.cam_out_layer.bias, 0)

    def forward(
        self,
        x: TensorDict,
        t: torch.Tensor,
        cond: TensorDict,
        points: Optional[torch.Tensor] = None,
        return_mid_features: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            x: TensorDict with 'latent' [B, L_x, C_in] key, optional 'cam' [B, 1, C_cam] key
            cond: TensorDict with 'feature1' [B, L_c, C_cond1] key, optional 'feature2' [B, L_c, C_cond2] key
        """
        
        z = x['latent']
        h_x = self.input_layer(z).type(self.dtype)
        if self.cond_embedder2 is not None:
            h_cond = self.cond_embedder(cond['feature1']) + self.cond_embedder2(cond['feature2'])
        else:
            h_cond = self.cond_embedder(cond['feature1'])
        h_cond = h_cond.type(self.dtype)
        
        t_emb = self.t_embedder(t)
        if self.share_mod:
            t_mod = self.adaLN_modulation(t_emb)
        else:
            t_mod = t_emb
        t_mod = t_mod.type(self.dtype)

        rotary_emb = None
        rope_x = None
        rope_cond = None

        if self.pos_embedder is not None:
            if self.pe_mode == "3d_ot_rope":
                rope_x = self.pos_embedder(self.pos_pe) # [1, L, 1, D/2]
                rope_x = rope_x.expand(z.shape[0], -1, -1, -1)
                
                # cond RoPE (Identity)
                rope_cond = torch.ones(z.shape[0], h_cond.shape[1], 1, rope_x.shape[-1], device=z.device, dtype=rope_x.real.dtype)
                rope_cond = torch.polar(rope_cond, torch.zeros_like(rope_cond)) 
                
                parts = [rope_x, rope_cond]
                if self.cam_channels is not None:
                    rope_cam = torch.ones(z.shape[0], x['camera'].shape[1], 1, rope_x.shape[-1], device=z.device, dtype=rope_x.real.dtype)
                    rope_cam = torch.polar(rope_cam, torch.zeros_like(rope_cam))
                    parts.append(rope_cam)
                
                rotary_emb = torch.cat(parts, dim=1)

            elif self.pe_mode in ["3d_ot_ape", "3d_ot_repo"]:
                h_x = h_x + self.pos_embedder(self.pos_pe).type(self.dtype)
        
        # Refine noise (z)
        for i, block in enumerate(self.noise_refiner):
            if "repo" in self.pe_mode:
                rope_x = self.noise_repo_layers[i](h_x)
            h_x = block(h_x, mod=t_mod, rotary_emb=rope_x)
            
        # Refine context (cond)
        # Note: We pass rope_cond (identity) if rope is used, though it might not affect much if it's identity
        for i, block in enumerate(self.context_refiner):
            if "repo" in self.pe_mode:
                rope_cond = self.context_repo_layers[i](h_cond)
            h_cond = block(h_cond, mod=None, rotary_emb=rope_cond)
            
        # Refine camera
        if self.cam_channels is not None:
            cam = x['camera']
            h_cam = self.cam_refiner(cam)
            h_cam = h_cam.type(self.dtype)

        # Concat
        h = torch.cat([h_x, h_cond], dim=1)
        if self.cam_channels is not None:
            h = torch.cat([h, h_cam], dim=1)

        # Handle Control Tokens
        ctrl_tokens = kwargs.get('ctrl_tokens', None)
        if ctrl_tokens is not None and self.ctrl_embedder is not None:
            # FIX: Cast to self.dtype (float16) to match the transformer blocks!
            h_ctrl = self.ctrl_embedder(ctrl_tokens).type(self.dtype)
            h = torch.cat([h, h_ctrl], dim=1)
        
        mid_features = None
        mid_idx = len(self.blocks) // 2
        for i, block in enumerate(self.blocks):
            if "repo" in self.pe_mode:
                rotary_emb = self.repo_layers[i](h)
            h = block(h, mod=t_mod, rotary_emb=rotary_emb)
            if i == mid_idx:
                mid_features = h
            
        # Split
        h_x = h[:, :z.shape[1]]
        h_x = h_x.type(z.dtype)
        h_x = F.layer_norm(h_x, h_x.shape[-1:])
        # camera out
        if self.cam_channels is not None:
            h_cam = h[:, -cam.shape[1]:]
            h_cam = h_cam.type(cam.dtype)
            h_cam = F.layer_norm(h_cam, h_cam.shape[-1:])
            
        if self.use_shift_table:
            shift, scale = (self.shift_table + t_emb.unsqueeze(1)).chunk(2, dim=1)
            h_x = h_x * (1 + scale) + shift
            if self.cam_channels is not None:
                h_cam = h_cam * (1 + scale) + shift
        
        h_x = self.out_layer(h_x)
        if self.cam_channels is not None:
            h_cam = self.cam_out_layer(h_cam)
        
        out = TensorDict({
            'latent': h_x,
        }, batch_size=x.batch_size)
        if self.cam_channels is not None:
            out['camera'] = h_cam
        if return_mid_features:
            return out, mid_features
        return out
