from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...utils.random_utils import hammersley_sequence, sample_probs
from ...representations import Gaussian
from .base import TransformerBase, ModulatedCrossOnlyTransformerBase, PcdAbsolutePositionEmbedder, PcdAbsolutePositionEmbedderv2
import tensordict
from tensordict import TensorDict
from .elastic_mixin import TransformerElasticMixin

class FixedlenEncoder(TransformerBase):
    def __init__(
        self,
        model_channels: int,
        latent_channels: int,
        cond_channels: int,
        num_blocks: int,
        cond_channels2: Optional[int] = None,
        in_channels: Optional[int] = None,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4.0,
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        qk_rms_norm: bool = True,
        qk_rms_norm_cross: bool = True,
        use_2_cross_block: bool = False,
        q_token_length: Optional[int] = 2048,
        pcd_pe_mode: Literal["pcd_ape", "pcd_ape_v2"] = "pcd_ape_v2",
        query_mode: Literal["learned", "fps"] = "learned",
    ):
        """
        Gaussian Feature Fixed Length Latent Encoder.
        Convert fixed length query token to fixed length latent, encode condition with cross attention.
        """
        super().__init__(
            in_channels=model_channels,
            model_channels=model_channels,
            cond_channels=model_channels,
            num_blocks=num_blocks,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            mlp_ratio=mlp_ratio,
            use_fp16=use_fp16,
            use_checkpoint=use_checkpoint,
            qk_rms_norm=qk_rms_norm,
            qk_rms_norm_cross=qk_rms_norm_cross,
            use_2_cross_block=use_2_cross_block,
        )
        self.query_mode = query_mode
        if self.query_mode == "fps":
            self.in_channels = 3
        else:
            self.in_channels = in_channels
        self.out_channels = latent_channels
        self.pcd_pe_mode = pcd_pe_mode
        # input projection
        self.in_proj = nn.Linear(self.in_channels, model_channels)
        # learnable query tokens
        self.q_token_length = q_token_length
        if q_token_length is not None and self.query_mode == "learned":
            self.q_tokens = nn.Parameter(torch.randn(1, q_token_length, self.in_channels) / self.in_channels**0.5)
        else:
            self.q_tokens = None
        # pcd condition
        self.cond_pcd_proj = nn.Linear(cond_channels, model_channels)
        if cond_channels2 is not None:
            self.cond_pcd_proj2 = nn.Linear(cond_channels2, model_channels)
        else:
            self.cond_pcd_proj2 = None
        if pcd_pe_mode == "pcd_ape":
            self.pos_embedder = PcdAbsolutePositionEmbedder(
                channels=model_channels,
                in_channels=3,
            )
        elif pcd_pe_mode == "pcd_ape_v2":
            self.pos_embedder = PcdAbsolutePositionEmbedderv2(
                channels=model_channels,
                in_channels=3,
            )
        else:
            self.pos_embedder = None
        # output projection
        self.mean_out = nn.Linear(model_channels, latent_channels)
        self.logvar_out = nn.Linear(model_channels, latent_channels)
        
        # weight init
        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()
        
        
    def initialize_weights(self):
        super().initialize_weights()
        # zero init logvar
        nn.init.constant_(self.logvar_out.weight, 0)
        nn.init.constant_(self.logvar_out.bias, 0)
        
    
    def forward(self, x: Optional[torch.Tensor] = None, cond: Optional[tensordict.TensorDict] = None, sample_posterior=True, return_raw=False, q_token_length=None, return_fps=False) -> torch.Tensor:
        """
        Args:
            x: [B, Lq, C] or None (if None, use learnable query tokens)
            cond: a dict containing (some keys may be missing):
                - cond_hidden_states_dino: [B, Lv, C]
                - ray_embeddings: [B, Lv, 6]
                - features: [B, L, C]
                - points: [B, L, 3]
                - mv_features: [B, Nv, C, H, W]
                - mv_extrinsics: [B, Nv, 4, 4]
                - mv_intrinsics: [B, Nv, 3, 3]
        """
        assert cond is not None, "Condition is required for GaussianFixlenEncoder"
        assert 'points' in cond and 'features' in cond, "Condition must contain 'points' and 'features'"
        B = cond['points'].shape[0]
        if self.query_mode == "fps":
            from pytorch3d.ops import sample_farthest_points
            q_token_length = self.q_token_length if q_token_length is None else q_token_length
            query_points, _ = sample_farthest_points(cond['points'], K=q_token_length)
            h = self.in_proj(query_points)
            h = h + self.pos_embedder(query_points.view(-1,3)).view(B, -1, h.shape[-1])
        elif self.query_mode == "learned":
            q_token_length = self.q_token_length if q_token_length is None else q_token_length
            if self.q_tokens is None:
                raise ValueError("learned query_mode requires q_tokens")
            query_points = self.q_tokens[:, :q_token_length, :].repeat(B, 1, 1)
            h = self.in_proj(query_points)

        ctx = cond['features']
        ctx = self.cond_pcd_proj(ctx)
        ctx = ctx + self.pos_embedder(cond['points'].view(-1,3)).view(B, -1, ctx.shape[-1])
        if self.cond_pcd_proj2 is not None:
            ctx2 = cond['features2']
            ctx2 = self.cond_pcd_proj2(ctx2)
            ctx2 = ctx2 + self.pos_embedder(cond['points2'].view(-1,3)).view(B, -1, ctx2.shape[-1])

        indices = None
        context_indices = None
        context2_indices = None

        if self.use_2_cross_block:
            h = super().forward(h, ctx, cond2=ctx2, indices=indices, context_indices=context_indices, context2_indices=context2_indices)
        else:
            if self.cond_pcd_proj2 is not None:
                ctx = torch.cat([ctx, ctx2], dim=1)
            h = super().forward(h, ctx, indices=indices, context_indices=context_indices, context2_indices=context2_indices)
        h = h.type(cond['points'].dtype)
            
        h = F.layer_norm(h, h.shape[-1:])
        mean = self.mean_out(h)
        logvar = self.logvar_out(h)
        if sample_posterior:
            std = torch.exp(0.5 * logvar)
            z = mean + std * torch.randn_like(std)
        else:
            z = mean
            
        if return_fps and not return_raw:
            return z, query_points
        elif return_fps and return_raw:
            return z, query_points, mean, logvar
        elif return_raw:
            return z, mean, logvar
        else:
            return z

class ElasticFixedlenEncoder(TransformerElasticMixin, FixedlenEncoder):
    def _get_input_size(self, x: Optional[torch.Tensor] = None, cond: Optional[tensordict.TensorDict] = None, sample_posterior=True, return_raw=False, q_token_length=None, *args, **kwargs):
        q_token_length = self.q_token_length if q_token_length is None else q_token_length
        return q_token_length

class FixedlenDecoder(TransformerBase):
    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full"] = "full",
        window_num: Optional[int] = None,
        pcd_pe_mode: Literal["pcd_ape", "pcd_ape_v2"] = "pcd_ape_v2",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        qk_rms_norm: bool = True,
        qk_rms_norm_cross: bool = True,
        share_mod: bool = False,
    ):
        """
        Gaussian Feature Fixed Length Latent Decoder.
        Decode fixed length latent to 3D Gaussian representation, encode condition with cross attention.
        Conditions:
            - latent: [B, Lq, C]
        Input:
            - x: [B, L, 3] (points to decode)
        """
        if attn_mode != "full":
            raise ValueError(f"Only attn_mode='full' is supported in the public decoder, got {attn_mode!r}.")
        super().__init__(
            in_channels=model_channels,
            model_channels=model_channels,
            cond_channels=cond_channels,
            num_blocks=num_blocks,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            mlp_ratio=mlp_ratio,
            use_fp16=use_fp16,
            use_checkpoint=use_checkpoint,
            qk_rms_norm=qk_rms_norm,
            qk_rms_norm_cross=qk_rms_norm_cross,
            share_mod=share_mod,
        )
        self.in_channels = in_channels
        self.in_proj = nn.Linear(in_channels, model_channels)
        self.pcd_pe_mode = pcd_pe_mode
        if pcd_pe_mode == "pcd_ape":
            self.pos_embedder = PcdAbsolutePositionEmbedder(
                channels=model_channels,
                in_channels=3,
            )
        elif pcd_pe_mode == "pcd_ape_v2":
            self.pos_embedder = PcdAbsolutePositionEmbedderv2(
                channels=model_channels,
                in_channels=3,
            )
        else:
            self.pos_embedder = None
            
    def forward(self, x: tensordict.TensorDict = None, cond: Optional[torch.Tensor]= None) -> torch.Tensor:
        """
        Args:
            x: a dict containing:
                - points: [B, L, 3] the query points to decode
            cond: [B, Lq, C] the latent features as condition
        Returns:
            list of Gaussian representations
        """
        pcd = x["points"]
        B, L, C = pcd.shape
        h = pcd
        
        # token num log scaling
        l = torch.full((B,), L).to(h)
        l = torch.log2(l).to(h)
        
        h = self.in_proj(h)
        h = h + self.pos_embedder(pcd.view(-1,C)).view(B, L, -1) if hasattr(self, 'pos_embedder') else h
        indices = None
        # prepare spatial cache

        h = super().forward(h, cond, l=l, indices=indices)
        h = h.type(cond.dtype)
        return h

class GaussianFixedlenDecoder(FixedlenDecoder):
    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full"] = "full",
        window_num: Optional[int] = None,
        pcd_pe_mode: Literal["pcd_ape", "pcd_ape_v2"] = "pcd_ape_v2",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        qk_rms_norm: bool = True,
        qk_rms_norm_cross: bool = True,
        share_mod: bool = False,
        *,
        representation_config: dict = None,
        use_learned_offset_scale: bool = False,
        gaussian_center_pertube_scale: float = 1.0,
        use_per_offset=False,
    ):
        self.rep_config = representation_config
        self.gaussian_center_pertube_scale = gaussian_center_pertube_scale
        self.use_learned_offset_scale = use_learned_offset_scale
        self.use_per_offset = use_per_offset
        self.out_channels = self._calc_layout()
        super().__init__(
            in_channels = in_channels,
            model_channels = model_channels,
            cond_channels = cond_channels,
            num_blocks = num_blocks,
            num_heads = num_heads,
            num_head_channels = num_head_channels,
            mlp_ratio = mlp_ratio,
            attn_mode = attn_mode,
            window_num = window_num,
            pcd_pe_mode = pcd_pe_mode,
            use_fp16 = use_fp16,
            use_checkpoint = use_checkpoint,
            qk_rms_norm = qk_rms_norm,
            qk_rms_norm_cross = qk_rms_norm_cross,
            share_mod = share_mod,
        )
        self.out_proj = nn.Linear(model_channels, self.out_channels)
        self._build_perturbation()
        
        # weight init
        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()

    def initialize_weights(self) -> None:
        super().initialize_weights()
        # zero-out output layers:
        nn.init.constant_(self.out_proj.weight, 0)
        nn.init.constant_(self.out_proj.bias, 0)
        
    def _calc_layout(self) -> None:
        self.layout = {
            '_xyz' : {'shape': (self.rep_config['num_gaussians'], 3), 'size': self.rep_config['num_gaussians'] * 3},
            '_features_dc' : {'shape': (self.rep_config['num_gaussians'], 1, 3), 'size': self.rep_config['num_gaussians'] * 3},
            '_scaling' : {'shape': (self.rep_config['num_gaussians'], 3), 'size': self.rep_config['num_gaussians'] * 3},
            '_rotation' : {'shape': (self.rep_config['num_gaussians'], 4), 'size': self.rep_config['num_gaussians'] * 4},
            '_opacity' : {'shape': (self.rep_config['num_gaussians'], 1), 'size': self.rep_config['num_gaussians']},
        }
        if self.use_learned_offset_scale:
            if self.use_per_offset:
                self.layout['_offset_scale'] = {'shape': (self.rep_config['num_gaussians'], 1), 'size': self.rep_config['num_gaussians']}
            else:
                raise ValueError("use_per_offset must be True when use_learned_offset_scale is True")
        start = 0
        for k, v in self.layout.items():
            v['range'] = (start, start + v['size'])
            start += v['size']
        out_channels = start
        return out_channels
    
    
    @torch.no_grad()
    def aggregrate_opacity_grad(self, opacity, opacity_grad, reduction="sum") -> torch.Tensor:
        """
        opacity: [B x Ns x 1]
        opacity_grad: [B x Ns x 1]
        """
        if not isinstance(opacity, torch.Tensor):
            opacity = torch.stack(opacity, dim=0) # [B x Ns x 1]
        if not isinstance(opacity_grad, torch.Tensor):
            opacity_grad = torch.stack(opacity_grad, dim=0) # [B x Ns x 1]
        # reshape to [B x P x N x 1]
        B = opacity.shape[0]
        opacity = opacity.reshape(B, -1, *self.layout['_opacity']['shape'])
        opacity_grad = opacity_grad.reshape(B, -1, *self.layout['_opacity']['shape'])
        d_loss = opacity_grad * opacity
        return getattr(torch, reduction)(d_loss, dim=-2).squeeze(-1) # [B x P]
    
    @torch.no_grad()
    def feature_to_offset_norm(self, features) -> Tuple[torch.Tensor, torch.Tensor]:
        offset = self._get_offset(features)
        offset_mean = offset.mean(dim=-2)
        offset_norm = torch.linalg.vector_norm(offset_mean, dim=-1)
        return offset_norm
        
        
    def _build_perturbation(self) -> None:
        perturbation = [hammersley_sequence(3, i, self.rep_config['num_gaussians']) for i in range(self.rep_config['num_gaussians'])]
        perturbation = torch.tensor(perturbation).float() * 2 - 1
        perturbation = perturbation / self.rep_config['perturbe_size']
        perturbation = torch.atanh(perturbation).to(self.device)
        self.register_buffer('points_offset_perturbation', perturbation)
        
        if self.use_learned_offset_scale:
             base_offset_scale = torch.tensor(self.rep_config['offset_scale'])
             base_offset_scale = torch.log(torch.exp(base_offset_scale) - 1.0) # inverse softplus
             self.register_buffer('base_offset_scale', base_offset_scale)
    
    def _get_offset(self, h):
        B = h.shape[0]
        if self.use_learned_offset_scale:
            _offset_scale = h[:,:, self.layout['_offset_scale']['range'][0]:self.layout['_offset_scale']['range'][1]].reshape(B, -1, *self.layout['_offset_scale']['shape'])
            _offset_scale = F.softplus(_offset_scale+self.base_offset_scale)
            
        offset = h[:,:, self.layout['_xyz']['range'][0]:self.layout['_xyz']['range'][1]].reshape(B, -1, *self.layout['_xyz']['shape'])
        offset = offset * self.rep_config['lr']['_xyz']
        if self.rep_config['perturb_offset']:
            offset = offset + self.points_offset_perturbation
        offset = torch.tanh(offset) * 0.5 * self.rep_config['perturbe_size']
        if self.use_learned_offset_scale:
            offset = offset * _offset_scale
        else:
            offset = offset * self.rep_config['offset_scale']
        return offset
        

    def to_representation(self, x: torch.Tensor, h: torch.Tensor, return_status: bool = False) -> List[Gaussian]:
        """
        Convert a batch of network outputs to 3D representations.

        Args:
            x: the input tensor of shape [B, L, C].
            h: the output tensor of shape [B, L, C'] from the MLP.

        Returns:
            list of representations
        """
        x = h["points"] if "points" in h else x["points"]
        offset = self._get_offset(h['features'])
        h = h["features"]
        ret = []
        for i in range(h.shape[0]):
            representation = Gaussian(
                sh_degree=0,
                aabb=[-0.5, -0.5, -0.5, 1.0, 1.0, 1.0],
                mininum_kernel_size = self.rep_config['3d_filter_kernel_size'],
                scaling_bias = self.rep_config['scaling_bias'],
                opacity_bias = self.rep_config['opacity_bias'],
                scaling_activation = self.rep_config['scaling_activation']
            )
            _x = x[i,:,None,:]

            for k, v in self.layout.items():
                if k == '_xyz':
                    _xyz = offset[i] + _x # NOTE: based on x data range == [0, 1]
                    setattr(representation, k, _xyz.flatten(0, 1))
                elif k == '_xyz_center':
                    continue
                elif k == '_offset_scale':
                    continue
                else:
                    feats = h[i][:, v['range'][0]:v['range'][1]].reshape(-1, *v['shape']).flatten(0, 1)
                    feats = feats * self.rep_config['lr'][k]
                    setattr(representation, k, feats)
            ret.append(representation)
        if return_status:
            with torch.no_grad():
                status = {}
                offset_center = offset.mean(dim=-2)
                offset_center_norm = torch.linalg.vector_norm(offset_center, dim=-1)
                status['offset_center_norm_mean'] = offset_center_norm.mean().item()
                status['offset_center_norm_max'] = offset_center_norm.max().item()
                offset_norm = torch.linalg.vector_norm(offset, dim=-1)
                status['offset_norm_mean'] = offset_norm.mean().item()
                status['offset_norm_max'] = offset_norm.max().item()
            return ret, status
        return ret
    
    def forward(self, x = None, cond = None) -> TensorDict:
        h = super().forward(x, cond)
        
        h = F.layer_norm(h, h.shape[-1:])
        h_out = self.out_proj(h)
        return TensorDict({
            'features': h_out
        }, batch_size=h_out.shape[:1], device=h.device)

class ElasticGaussianFixedlenDecoder(TransformerElasticMixin, GaussianFixedlenDecoder):
    def _get_input_size(self, x = None, cond = None, *args, **kwargs):
        return cond.shape[1]

class OctreeProbabilityFixedlenDecoder(ModulatedCrossOnlyTransformerBase):
    def __init__(
        self,
        model_channels: int,
        cond_channels: Optional[int],
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4.0,
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        share_mod: bool = False,
        qk_rms_norm_cross: bool = True,
        *,
        pcd_pe_mode: Literal["pcd_ape", "pcd_ape_v2"] = "pcd_ape_v2",
        **kwargs,
    ):
        super().__init__(
            in_channels = model_channels, # pe embedded
            model_channels = model_channels,
            cond_channels = cond_channels,
            num_blocks = num_blocks,
            num_heads = num_heads,
            num_head_channels = num_head_channels,
            mlp_ratio = mlp_ratio,
            use_fp16 = use_fp16,
            use_checkpoint = use_checkpoint,
            share_mod = share_mod,
            qk_rms_norm_cross = qk_rms_norm_cross,
        )
        self.out_proj = nn.Linear(self.model_channels, 2 ** 3) # to 8 logits
        
        self.in_proj = nn.Linear(3, self.model_channels)
        self.pcd_pe_mode = pcd_pe_mode
        if pcd_pe_mode == "pcd_ape":
            self.pos_embedder = PcdAbsolutePositionEmbedder(
                channels=model_channels,
                in_channels=3,
            )
        elif pcd_pe_mode == "pcd_ape_v2":
            self.pos_embedder = PcdAbsolutePositionEmbedderv2(
                channels=model_channels,
                in_channels=3,
            )
        else:
            self.pos_embedder = None
            
        # weight init
        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()
            
    def initialize_weights(self) -> None:
        super().initialize_weights()
        # zero-out output layers:
        nn.init.constant_(self.out_proj.weight, 0)
        nn.init.constant_(self.out_proj.bias, 0)
        
    def forward(self, x: torch.Tensor, l: torch.Tensor, cond: Optional[torch.Tensor]= None, l2: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: query points voxel centers, [B(=1), L, 3]
            l: query points level, [B(=1)]
            cond: [B(=1), Lq, C] the latent features as condition
        Returns:
            list of Gaussian representations
        """
        B, L, C = x.shape
        h = self.in_proj(x)
        h = h + self.pos_embedder(x.view(-1,3)).view(B, L, -1) if hasattr(self, 'pos_embedder') else h
        # FIXME: elegant l2 log scaling
        if l2 is not None:
            l2 = torch.log2(l2)
        h = super().forward(h, l, cond, l2)
        h = h.type(cond.dtype)
        h = F.layer_norm(h, h.shape[-1:])
        logits = self.out_proj(h)
        probs = torch.softmax(logits, dim=-1)
        return TensorDict({'logits': logits, 'probs': probs}, batch_size=logits.shape[:1], device=logits.device)

    @staticmethod
    def sample(model, cond, num_points, level, temperature=1.0, algo: Literal["iid", "residual", "systematic"] = "systematic") -> TensorDict:
        """
        Sample points from the model.
        Args:
            cond: object latent condition, [B, Lc, C]
            num_points: number of points to sample
            level: level of the octree
            temperature: temperature of the sampling
        Returns:
            points: sampled points, [B, num_points, 3]
            log_probs: log probabilities of the sampled points, [B, num_points]
        """
        B = cond.shape[0]
        device = cond.device
        child_offset = torch.tensor([[i, j, k] for k in [0,1] for j in [0,1] for i in [0,1]], dtype=torch.long, device=device)
        
        prev_coords_int = torch.zeros(B, 1, 3, dtype=torch.long, device=device) # [B, 1, 3]
        prev_counts = torch.full((B, 1), num_points, dtype=torch.long, device=device) # [B, 1]
        prev_log_probs = torch.zeros(B, 1, dtype=torch.float32, device=device) # [B, 1]
        
        batch_indices_range = torch.arange(B, device=device).unsqueeze(1)
        num_tensor = torch.full((B,), num_points, dtype=torch.long, device=device)

        for l in range(1, level + 1):
            res_p = 1 << (l-1)
            res = 1 << l
                
            parent_coords_int = prev_coords_int # [B, Np, 3]
            parent_coords_norm = (parent_coords_int.to(torch.float32) + 0.5) / res_p
            res_tensor = torch.full((B,), res, dtype=torch.long, device=device)
            
            pred_logits = model(parent_coords_norm, res_tensor, cond, num_tensor)['logits'] # [B, Np, 8]
            pred_logits = pred_logits / temperature
            pred_probs = torch.softmax(pred_logits, dim=-1) # [B, Np, 8]
            pred_log_probs = torch.log_softmax(pred_logits, dim=-1) # [B, Np, 8]
            sampled = sample_probs(pred_probs, prev_counts, algo=algo).flatten(1,2) # [B, Np*8]
            pred_log_probs = pred_log_probs.flatten(1,2) # [B, Np*8]
            prev_log_probs_expanded = prev_log_probs.repeat_interleave(8, dim=1) # [B, Np*8]

            # prepare coords
            child_coords_int = parent_coords_int[:, :, None, :] * 2 + child_offset[None, None, :, :] # [B,. Np, 8, 3]
            child_coords_int = child_coords_int.flatten(1,2) # [B, Np*8, 3]
                
            # mask out and padding
            # Vectorized implementation
            mask = sampled > 0 # [B, K]
            valid_counts = mask.sum(dim=1) # [B]
            max_valid = valid_counts.max().item()
            
            # Create scatter indices
            scatter_indices = mask.cumsum(dim=1) - 1
            valid_mask = mask # just mask
            
            valid_scatter_indices = scatter_indices[valid_mask] # [Total_Valid]
            
            # Batch indices
            valid_batch_indices = batch_indices_range.expand_as(mask)[valid_mask]
            
            next_prev_coords_int = torch.zeros(B, max_valid, 3, dtype=child_coords_int.dtype, device=device)
            next_prev_coords_int[valid_batch_indices, valid_scatter_indices] = child_coords_int[valid_mask]
            
            next_prev_counts = torch.zeros(B, max_valid, dtype=sampled.dtype, device=device)
            next_prev_counts[valid_batch_indices, valid_scatter_indices] = sampled[valid_mask]
            
            next_prev_log_probs = torch.zeros(B, max_valid, dtype=prev_log_probs.dtype, device=device)
            next_prev_log_probs[valid_batch_indices, valid_scatter_indices] = (prev_log_probs_expanded + pred_log_probs)[valid_mask]
            
            prev_coords_int = next_prev_coords_int
            prev_counts = next_prev_counts
            prev_log_probs = next_prev_log_probs

            
        # finially randomly put points in each voxel
        res = 1 << level
        # unflatten the coords according to counts
        prev_log_probs = torch.repeat_interleave(prev_log_probs.flatten(0,1), prev_counts.flatten(0,1), dim=0).reshape(B, num_points,) # [B, num_points]
        coords_int = torch.repeat_interleave(prev_coords_int.flatten(0,1), prev_counts.flatten(0,1), dim=0).reshape(B, num_points, -1) # [B, num_points, 3]
        coords_norm = (coords_int.to(torch.float32) + torch.rand_like(coords_int, dtype=torch.float32)) / res

        points = coords_norm
        log_probs = prev_log_probs
        return TensorDict({
            'points': points,
            'log_probs': log_probs,
        }, batch_size=points.shape[:1], device=points.device)
