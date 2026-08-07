from typing import *
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from deg.utils.general_utils import edict
import utils3d.torch
from tensordict import TensorDict
import tensordict
import einops
from ..basic import BasicTrainer
from ...representations import Gaussian
from ...renderers import GaussianRenderer
from ...utils.loss_utils import l1_loss, l2_loss, ssim, lpips, chamfer_distance, psnr
from tqdm import tqdm
from ...utils.data_utils import recursive_to_device
from ..utils import *
from ...utils.general_utils import *
from ...utils.postprocessing_utils import octree_from_points
from ...models.gs_seqence_vae.gs_fixlen_vae import OctreeProbabilityFixedlenDecoder


def grad_normalization(grad_output, clamp=False, mean=False, std=False, normalize01=False, clamp_percentile=None):
    if normalize01:
        eps = 1e-3
        scale = grad_output.abs().mean()
        bad_mask = grad_output > - eps * scale # negative scale
        mean_scale = grad_output.numel()
        grad_normalized = torch.zeros_like(grad_output)
        grad_normalized[bad_mask] = 1 / mean_scale
        grad_output = grad_normalized
    if clamp:
        grad_output = grad_output.clamp(max=0.0)
    if clamp_percentile is not None and clamp_percentile > 0.0:
        q = 1.0 - clamp_percentile
        threshold = torch.quantile(grad_output.abs(), q, dim=-1, keepdim=True)
        grad_output = torch.clamp(grad_output, -threshold, threshold)
    if mean:
        grad_output = grad_output - grad_output.mean(dim=-1, keepdim=True)
    if std:
        grad_output = grad_output / grad_output.std(dim=-1, keepdim=True)
    return grad_output


class GradNormalization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, clamp=False, mean=False, std=False, normalize01=False, clamp_percentile=None):
        ctx.clamp = clamp
        ctx.mean = mean
        ctx.std = std
        ctx.normalize01 = normalize01
        ctx.clamp_percentile = clamp_percentile
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_normalization(grad_output, ctx.clamp, ctx.mean, ctx.std, ctx.normalize01, ctx.clamp_percentile)
        return grad_output, None, None, None, None, None

class OctreeFixlenVaeTrainer(BasicTrainer):
    def __init__(
        self,
        *args,
        loss_type: str = 'l1',
        lambda_ssim: float = 0,
        lambda_lpips: float = 0,
        lambda_alpha: float = 0,
        lambda_depth: float = 0,
        lambda_alphaed_diff: float = 0,
        lambda_kl: float = 0,
        lambda_offset_reg: float = 0,
        lambda_offset_nn_reg: float = 0,
        lambda_offset_sphere_reg: float = 0,
        offset_reg_k: int = 8,
        offset_reg_dist_factor: float = 1.0,
        regularizations: Dict = {},
        varlen_max_log_scale: Optional[int] = None,
        varlen_log_scale_uniform: bool = True,
        use_render_loss: bool = True,
        use_geo_loss: bool = False,
        lambda_chamfer: float = 0.0,
        lambda_earth_mover: float = 0.0,
        geo_warmup_steps: int = 0,
        dual_decoder: bool = False,
        dual_encoder: bool = False,
        detach_render_grad: bool = False,
        latent_token_varlen_max_log_scale: Optional[int] = None,
        lambda_entropy: float = 1.0,
        max_voxel_level: int = 6,
        max_sampled_points: int = 8192,
        sample_algo: Literal["iid", "residual", "systematic"] = "systematic",
        lambda_learned_fps: float = 0.0,
        **kwargs
    ):
        self.loss_type = loss_type
        self.lambda_ssim = lambda_ssim
        self.lambda_lpips = lambda_lpips
        self.lambda_kl = lambda_kl
        self.lambda_alpha = lambda_alpha
        self.lambda_depth = lambda_depth
        self.lambda_alphaed_diff = lambda_alphaed_diff
        self.lambda_offset_reg = lambda_offset_reg
        self.lambda_offset_nn_reg = lambda_offset_nn_reg
        self.lambda_offset_sphere_reg = lambda_offset_sphere_reg
        self.offset_reg_k = offset_reg_k
        self.offset_reg_dist_factor = offset_reg_dist_factor
        self.regularizations = regularizations
        self.varlen_max_log_scale = varlen_max_log_scale
        self.varlen_log_scale_uniform = varlen_log_scale_uniform
        self.dual_encoder = dual_encoder
        self.use_render_loss = use_render_loss
        self.use_geo_loss = use_geo_loss
        self.lambda_chamfer = lambda_chamfer
        self.lambda_earth_mover = lambda_earth_mover
        self.geo_warmup_steps = geo_warmup_steps
        self.dual_decoder = dual_decoder
        self.detach_render_grad = detach_render_grad
        self.latent_token_varlen_max_log_scale = latent_token_varlen_max_log_scale
        self.latent_token_len_random_generator = torch.Generator()
        self.latent_token_len_random_generator.manual_seed(233)
        self.lambda_entropy = lambda_entropy
        self.max_voxel_level = max_voxel_level
        self.max_sampled_points = max_sampled_points
        self.sample_algo = sample_algo
        self.lambda_learned_fps = lambda_learned_fps
        self.pcd_num_random_generator = torch.Generator()
        self.pcd_num_random_generator.manual_seed(666)

        super().__init__(*args, **kwargs)
        assert not self.dual_encoder, "dual encoder is not supported"
        self._init_renderer()

    # ------------------------------------------------------------------ #
    # Renderer                                                             #
    # ------------------------------------------------------------------ #
    def _init_renderer(self):
        rendering_options = {"near": 0.8, "far": 1.6, "bg_color": 'random'}
        self.renderer = GaussianRenderer(rendering_options)
        if self.use_render_loss:
            self.renderer.pipe.kernel_size = self.models['decoder_gs'].rep_config['2d_filter_kernel_size']
        else:
            self.renderer.pipe.kernel_size = 0.1

    def _render_batch(self, reps: List[Gaussian], extrinsics: torch.Tensor, intrinsics: torch.Tensor,
                      return_opacity: bool = False, return_depth: bool = False) -> torch.Tensor:
        B, Nv, _, _ = extrinsics.shape
        ret = []
        for i, representation in enumerate(reps):
            ret_nv = []
            for nv in range(Nv):
                render_pack = self.renderer.render(
                    representation,
                    extrinsics[i, nv],
                    intrinsics[i, nv],
                    return_opacity=return_opacity,
                    return_depth=return_depth,
                )
                render_pack['bg_color'] = self.renderer.bg_color
                ret_nv.append(TensorDict(render_pack))
            ret.append(tensordict.stack(ret_nv, dim=0))
        return tensordict.stack(ret, dim=0)

    # ------------------------------------------------------------------ #
    # Status helpers                                                       #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _get_status(self, reps: List[Gaussian]) -> Dict:
        xyz = torch.cat([g.get_xyz for g in reps], dim=0)
        status = {
            'xyz': xyz,
            'scale': torch.cat([g.get_scaling for g in reps], dim=0),
            'opacity': torch.cat([g.get_opacity for g in reps], dim=0),
        }
        for k in list(status.keys()):
            status[k] = {
                'mean': status[k].mean().item(),
                'max': status[k].max().item(),
                'min': status[k].min().item(),
            }
        status['num_gs'] = reps[0].get_xyz.shape[0]
        return status

    @torch.no_grad()
    def _get_latent_status(self, z: torch.Tensor, mean=None, logvar=None) -> Dict:
        status = edict()
        if z is not None:
            status.z_mean = z.mean().detach().float()
            status.z_std = z.std().detach().float()
            dim = z.shape[-1]
            for i in range(dim):
                status[f'z_mean_{i}'] = z[..., i].mean().detach().float()
                status[f'z_std_{i}'] = z[..., i].std().detach().float()
            status.z_token_length = z.shape[1]
        if mean is not None:
            status.z_mu_mean = mean.mean().detach().float()
            status.z_mu_std = mean.std().detach().float()
        if logvar is not None:
            status.z_sigma_mean = logvar.exp().mean().detach().float()
            status.z_sigma_std = logvar.exp().std().detach().float()
            status.z_logvar_mean = logvar.mean().detach().float()
            status.z_logvar_std = logvar.std().detach().float()
        return status

    # ------------------------------------------------------------------ #
    # Regularization                                                       #
    # ------------------------------------------------------------------ #
    def _get_regularization_loss(self, reps: List[Gaussian]) -> Tuple[torch.Tensor, Dict]:
        loss = 0.0
        terms = {}
        if 'lambda_vol' in self.regularizations:
            scales = torch.cat([g.get_scaling for g in reps], dim=0)
            volume = torch.prod(scales, dim=1)
            terms['reg_vol'] = volume.mean()
            loss = loss + self.regularizations['lambda_vol'] * terms['reg_vol']
        if 'lambda_opacity' in self.regularizations:
            opacity = torch.cat([g.get_opacity for g in reps], dim=0)
            terms['reg_opacity'] = 1.0 - opacity.mean()
            loss = loss + self.regularizations['lambda_opacity'] * terms['reg_opacity']
        return loss, terms

    def get_offset_reg(self, points_center, offset) -> Dict:
        terms = edict()
        loss = 0.0
        if self.lambda_offset_reg > 0:
            with torch.no_grad():
                dist = torch.cdist(points_center, points_center, compute_mode='donot_use_mm_for_euclid_dist')
                knn_dist, _ = torch.topk(dist, self.offset_reg_k, largest=False, sorted=True)
                largest_dist = knn_dist[:, :, -1]
            offset_norm = torch.linalg.vector_norm(offset, dim=-1)
            offset_reg_loss = F.relu(offset_norm - self.offset_reg_dist_factor * largest_dist[..., None])
            offset_reg_loss = offset_reg_loss.mean()
            terms['offset_reg_loss'] = offset_reg_loss
            loss = loss + self.lambda_offset_reg * offset_reg_loss
        if self.lambda_offset_nn_reg > 0:
            B, N, G, _ = offset.shape
            with torch.no_grad():
                gs_pos = points_center[:, :, None, :] + offset
                gs_pos = gs_pos.flatten(1, 2)
                dist = torch.cdist(gs_pos, points_center, compute_mode='donot_use_mm_for_euclid_dist')
                nearest_dist = dist.min(dim=-1).values.reshape(B, N, G)
            offset_scaled = self.offset_reg_dist_factor * torch.linalg.vector_norm(offset, dim=-1)
            offset_activated = F.relu(offset_scaled - nearest_dist)
            offset_nn_loss = offset_activated.mean()
            terms['offset_nn_reg_loss'] = offset_nn_loss
            loss = loss + self.lambda_offset_nn_reg * offset_nn_loss
        if self.lambda_offset_sphere_reg > 0:
            B, N, G, _ = offset.shape
            offset_std = torch.sqrt((offset ** 2).sum(dim=-1).mean(dim=-1))
            offset_center = offset.mean(dim=-2)
            offset_sphere_reg_diag = F.relu(torch.linalg.vector_norm(offset_center, dim=-1) - self.offset_reg_dist_factor * offset_std).mean() / N
            dist_to_center = torch.cdist(points_center, points_center, compute_mode='donot_use_mm_for_euclid_dist')
            diag_mask = torch.eye(N, device=dist_to_center.device) * 1e8
            effective_dist = dist_to_center + diag_mask.unsqueeze(0)
            offset_sphere_reg_offdiag = F.relu(offset_std[..., None] - effective_dist).mean()
            offset_sphere_reg_loss = offset_sphere_reg_diag + offset_sphere_reg_offdiag
            terms['offset_sphere_reg_diag'] = offset_sphere_reg_diag
            terms['offset_sphere_reg_offdiag'] = offset_sphere_reg_offdiag
            terms['offset_sphere_reg_loss'] = offset_sphere_reg_loss
            loss = loss + self.lambda_offset_sphere_reg * offset_sphere_reg_loss
        return loss, terms

    # ------------------------------------------------------------------ #
    # Rendering loss helpers                                               #
    # ------------------------------------------------------------------ #
    def get_rec_loss_terms(self, rec_image, gt_image) -> Dict:
        B, Nv, _, H, W = gt_image.shape
        terms = edict(loss=0.0, rec=0.0)
        if self.loss_type == 'l1':
            terms["l1"] = l1_loss(rec_image, gt_image)
            terms["rec"] = terms["rec"] + terms["l1"]
        elif self.loss_type == 'l2':
            terms["l2"] = l2_loss(rec_image, gt_image)
            terms["rec"] = terms["rec"] + terms["l2"]
        else:
            raise ValueError(f"Invalid loss type: {self.loss_type}")
        if self.lambda_ssim > 0:
            terms["ssim"] = 1 - ssim(rec_image.view(-1, 3, H, W), gt_image.view(-1, 3, H, W))
            terms["rec"] = terms["rec"] + self.lambda_ssim * terms["ssim"]
        if self.lambda_lpips > 0:
            terms["lpips"] = lpips(rec_image.view(-1, 3, H, W), gt_image.view(-1, 3, H, W))
            terms["rec"] = terms["rec"] + self.lambda_lpips * terms["lpips"]
        terms['loss'] = terms['rec']
        return terms

    def _get_rendered_image_and_gt_image(self, reps, target_images):
        self.renderer.rendering_options.resolution = target_images["images"].shape[-1]
        render_results = self._render_batch(
            reps, target_images["extrinsics"], target_images["intrinsics"],
            return_depth=self.lambda_depth > 0, return_opacity=self.lambda_alpha > 0)
        rec_image = render_results['color']
        gt_image = (target_images["images"] * target_images["alphas"][..., None, :, :]
                    + (1 - target_images["alphas"][..., None, :, :]) * render_results['bg_color'][..., None, None])
        return render_results, rec_image, gt_image

    def get_other_rec_loss_terms(self, reps, render_results, target_images):
        terms = edict(loss=0.0)
        if self.lambda_alpha > 0:
            if self.loss_type == 'l1':
                terms["alpha_l1"] = l1_loss(render_results['opacity'], target_images["alphas"])
                terms["loss"] = terms["loss"] + self.lambda_alpha * terms["alpha_l1"]
            elif self.loss_type == 'l2':
                terms["alpha_l2"] = l2_loss(render_results['opacity'], target_images["alphas"])
                terms["loss"] = terms["loss"] + self.lambda_alpha * terms["alpha_l2"]
        if self.lambda_depth > 0:
            if self.loss_type == 'l1':
                terms["depth_l1"] = l1_loss(render_results['depth'], target_images["depths"])
                terms["loss"] = terms["loss"] + self.lambda_depth * terms["depth_l1"]
            elif self.loss_type == 'l2':
                terms["depth_l2"] = l2_loss(render_results['depth'], target_images["depths"])
                terms["loss"] = terms["loss"] + self.lambda_depth * terms["depth_l2"]
        reg_loss, reg_terms = self._get_regularization_loss(reps)
        terms.update(reg_terms)
        terms["loss"] = terms["loss"] + reg_loss
        return terms

    def get_render_loss_terms(self, reps, target_images) -> Dict:
        terms = edict(loss=0.0)
        render_results, rec_image, gt_image = self._get_rendered_image_and_gt_image(reps, target_images)
        rec_loss_terms = self.get_rec_loss_terms(rec_image, gt_image)
        other_rec_loss_terms = self.get_other_rec_loss_terms(reps, render_results, target_images)
        terms = terms_merge(terms, rec_loss_terms, other_rec_loss_terms)
        return terms

    # ------------------------------------------------------------------ #
    # Condition / cond helpers                                             #
    # ------------------------------------------------------------------ #
    def get_cond(self, cond: Optional[TensorDict] = None, varlen_log_scale: Optional[int] = None) -> Optional[TensorDict]:
        return cond

    def vis_cond(self, cond, **kwargs):
        return {}

    # ------------------------------------------------------------------ #
    # Latent token length sampling                                         #
    # ------------------------------------------------------------------ #
    def _sample_tgt_latent_token_length(self, encoder, latent_token_varlen_log_scale: Optional[int] = None) -> int:
        if latent_token_varlen_log_scale is None:
            if self.latent_token_varlen_max_log_scale is None:
                return encoder.q_token_length
            else:
                random_log_scale = torch.randint(
                    0, self.latent_token_varlen_max_log_scale + 1, (1,),
                    generator=self.latent_token_len_random_generator).item()
                return encoder.q_token_length // (2 ** random_log_scale)
        return encoder.q_token_length // (2 ** latent_token_varlen_log_scale)

    # ------------------------------------------------------------------ #
    # PCD visualisation helper                                             #
    # ------------------------------------------------------------------ #
    def pcd_to_representation(self, pcds, color=None, render_mode="xyz", scale=0.004):
        reps = []
        for i in range(pcds.shape[0]):
            pcd = pcds[i]
            representation = Gaussian(
                sh_degree=0,
                aabb=[-0.5, -0.5, -0.5, 1.0, 1.0, 1.0],
                scaling_bias=scale,
            )
            if render_mode == "xyz":
                setattr(representation, '_xyz', pcd)
                setattr(representation, '_features_dc', pcd[..., None, :] * 2 - 1)
                setattr(representation, '_scaling', torch.zeros_like(pcd))
                setattr(representation, '_rotation', torch.zeros(pcd.shape[0], 4).to(pcd.device))
                setattr(representation, '_opacity', torch.ones(pcd.shape[0], 1).to(pcd.device) * 10.0)
            elif render_mode == "custom":
                assert color is not None
                if color.dim() == 2:
                    color = color[:, :, None]
                color_i = color[i, :, None, :]
                if color_i.shape[-1] == 1:
                    color_i = color_i.repeat(1, 1, 3)
                setattr(representation, '_xyz', pcd)
                setattr(representation, '_features_dc', color_i)
                setattr(representation, '_scaling', torch.zeros_like(pcd))
                setattr(representation, '_rotation', torch.zeros(pcd.shape[0], 4).to(pcd.device))
                setattr(representation, '_opacity', torch.ones(pcd.shape[0], 1).to(pcd.device) * 10.0)
            reps.append(representation)
        return reps
        
    def _sample_tgt_pcd_num(self, varlen_log_scale: Optional[int] = None) -> int:
        """
        get the target point cloud number for varlen training
        Args:
            varlen_log_scale: the varlen log scale. When doing inference, varlen_log_scale is not None, return max_sampled_points // (2 ** varlen_log_scale). When training with random varlen, varlen_log_scale is None, return max_sampled_points // (2 ** random_log_scale). If self.varlen_max_log_scale is None, always return max_sampled_points.
        """
        if varlen_log_scale is None:
            # training
            if self.varlen_max_log_scale is None:
                return self.max_sampled_points
            else:
                if self.varlen_log_scale_uniform:
                    # use rank consistent random generator
                    random_log_scale = self.varlen_max_log_scale * torch.rand(1, generator=self.pcd_num_random_generator).item()
                    return int(self.max_sampled_points / (2 ** random_log_scale))
                else:
                    min_token_num = self.max_sampled_points // (2 ** self.varlen_max_log_scale)
                    pcd_num = torch.randint(min_token_num, self.max_sampled_points, (1,), generator=self.pcd_num_random_generator).item()
                    return pcd_num
        return int(self.max_sampled_points / (2 ** varlen_log_scale))
        
    def get_geo_prob_loss_terms(self, x_0: TensorDict, z: Optional[TensorDict] = None) -> Dict:
        terms = edict(loss = 0.0)
        lambda_entropy = self.lambda_entropy if self.geo_warmup_steps < self.step else 0.001 # add reg on warmup stage
        if lambda_entropy > 0:
            terms['entropy_loss'] = 0.0
            B = len(x_0)
            octrees = [octree_from_points(x_0['points'][i], L=self.max_voxel_level) for i in range(B)]
            for l in range(self.max_voxel_level):
                res = [octrees[i][l]['res'] for i in range(B)]
                coords_norm = [octrees[i][l]['parent_coords_norm'] for i in range(B)]
                probs = [octrees[i][l]['child8_probs_global'] for i in range(B)]
                child8_probs = [octrees[i][l]['child8_probs_cond'] for i in range(B)]
                res = torch.stack(res)
                padded_coords = torch.nn.utils.rnn.pad_sequence(coords_norm, batch_first=True, padding_value=0.0) # [B, M, 3]
                target_probs = torch.nn.utils.rnn.pad_sequence(probs, batch_first=True, padding_value=0.0) # [B, M, 8]
                
                # sample random num_gs
                num_points = self._sample_tgt_pcd_num()
                num_points = torch.full((B,), num_points, dtype=torch.long, device=res.device)
                
                preds = self.training_models["decoder"](x=padded_coords, l=res, cond=z, l2=num_points)
                
                # compute entropy loss
                log_prob_preds = F.log_softmax(preds["logits"], dim=-1) # [B, M, 8]
                entropy_loss = -(target_probs * log_prob_preds).sum(dim=-1) # [B, M]
                base_entropy = [(- p * torch.log(cp + 1e-8)).sum() for p, cp in zip(probs, child8_probs)]
                base_entropy = torch.stack(base_entropy).mean()
                terms['entropy_loss_l'+str(l)] = entropy_loss.sum() / B - base_entropy
                terms['entropy_loss'] = terms['entropy_loss'] + terms['entropy_loss_l'+str(l)]
            terms['loss'] = terms['loss'] + lambda_entropy * terms['entropy_loss']
        return terms
    
    def training_losses(
        self,
        x_0: TensorDict,
        target_images: TensorDict,
        cond: Optional[TensorDict] = None,
        **kwargs
    ) -> Tuple[Dict, Dict]:
        """
        Compute training losses.

        Args:
            x_0: empty now
            cond:
                - points: The [B x P x 3] tensor of point clouds.
                - feats: The [B x P x C] tensor of point features.
                - cond_hidden_states_dino: The [B x Lv x C] tensor of image features.
                - ray_embeddings: The [B x Lv x 6] tensor of ray embeddings.
            target_images:
                image: The [B x Nv x 3 x H x W] tensor of images.
                alphas: The [B x Nv x H x W] tensor of alphas channels.
                extrinsics: The [B x Nv x 4 x 4] tensor of extrinsics.
                intrinsics: The [B x Nv x 3 x 3] tensor of intrinsics.
            return_aux: Whether to return auxiliary information.

        Returns:
            a dict with the key "loss" containing a scalar tensor.
            may also contain other keys for different terms.
        """
        terms = edict(loss = 0.0)
        status = edict()
        cond = self.get_cond(cond)
        latent_token_length = self._sample_tgt_latent_token_length(self.models["encoder"])
        if self.lambda_learned_fps > 0.0:
            z, query_points, mean, logvar = self.training_models["encoder"](x=None, cond=cond, sample_posterior=True, return_raw=True, q_token_length=latent_token_length, return_fps=True)
            
            # chamfer reg
            terms['loss_learned_fps'] = chamfer_distance(query_points, cond['points'], training=True)
            terms['loss'] = terms['loss'] + self.lambda_learned_fps * terms['loss_learned_fps']
            print("using fps")
        else:
            z, mean, logvar = self.training_models["encoder"](x=None, cond=cond, sample_posterior=True, return_raw=True, q_token_length=latent_token_length)
        
        terms_prob = self.get_geo_prob_loss_terms(x_0, z)
        for k, v in terms_prob.items():
            if k in terms:
                terms[k] = terms[k] + v
            else:
                terms[k] = v
        status.update(self._get_latent_status(z, mean=mean, logvar=logvar))
                
        # kl
        terms["loss_kl"] = 0.5 * torch.mean(mean.pow(2) + logvar.exp() - logvar - 1)
        terms["loss"] = terms["loss"] + self.lambda_kl * terms["loss_kl"]
                
        return terms, status
    
    # FIXME: change vl to token length
    @torch.no_grad()
    def run_snapshot(
        self,
        num_samples: int,
        batch_size: int,
        verbose: bool = False,
    ) -> Dict:
        dataloader = DataLoader(
            copy.deepcopy(self.dataset_train_val if self.dataset_train_val is not None else self.dataset),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
        )

        # inference
        ret_dict = {}
        varlen_log_scale_list = [0] if self.varlen_max_log_scale is None else list(range(self.varlen_max_log_scale + 1))
        target_images = []
        preds = {vl: [] for vl in varlen_log_scale_list}
        preds['ref'] = []
        if self.lambda_learned_fps > 0.0:
            preds['learned_fps'] = []
        conds = []
        for i in range(0, num_samples, batch_size):
            batch = min(batch_size, num_samples - i)
            data = next(iter(dataloader))
            args = {k: v[:batch] for k, v in data.items()}
            args = recursive_to_device(args, self.device, non_blocking=True)
            target_images.append(args['target_images'])
            cond = args['cond'] if 'cond' in args else None
            conds.append(cond)
            
            cond = self.get_cond(cond)
            z = self.models["encoder"](x=None, cond=cond)
            if self.lambda_learned_fps > 0.0:
                _, query_points, _, _ = self.models["encoder"](x=None, cond=cond, sample_posterior=False, return_raw=True, return_fps=True)
                query_points = self.pcd_to_representation(query_points)
                preds['learned_fps'].extend(query_points)
            for vl in varlen_log_scale_list:
                num_points = self._sample_tgt_pcd_num(vl)
                points = OctreeProbabilityFixedlenDecoder.sample(self.models['decoder'], z, num_points=num_points, level=self.max_voxel_level, temperature=1.0, algo=self.sample_algo)['points']
                pred = self.pcd_to_representation(points)
                preds[vl].extend(pred)
            target_points = args['x_0']['points']
            target_points = self.pcd_to_representation(target_points)
            preds['ref'].extend(target_points)
        target_images = tensordict.cat(target_images, dim=0)
        resolution = target_images["images"].shape[-1]
        gt_images = target_images["images"] * target_images["alphas"][..., None, :, :] 
        gt_images = einops.rearrange(gt_images, 'b nv c h w -> c (b h) (nv w)')[None]
        ret_dict.update({f'gt_image': {'value': gt_images, 'type': 'image'}})
        
        if len(conds) > 0:
            cond_vis = self.vis_cond(tensordict.cat(conds, dim=0))
            ret_dict.update(cond_vis)

        for vl in preds:
            # render single view
            self.renderer.rendering_options.bg_color = (0, 0, 0)
            self.renderer.rendering_options.resolution = resolution
            render_results = self._render_batch(preds[vl], target_images["extrinsics"], target_images["intrinsics"])
            rec_images = einops.rearrange(render_results['color'], "b nv c h w -> c (b h) (nv w)")[None]
            if vl == 'ref':
                ret_dict.update({f'rec_image_ref': {'value': rec_images, 'type': 'image'}})
            elif vl == 'learned_fps':
                ret_dict.update({f'rec_image_learned_fps': {'value': rec_images, 'type': 'image'}})
            else:
                ret_dict.update({f'rec_image_vl{vl}': {'value': rec_images, 'type': 'image'}})

            # render multiview
            self.renderer.rendering_options.resolution = 512
            ## Build camera
            yaws = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
            yaws_offset = np.random.uniform(-np.pi / 4, np.pi / 4)
            yaws = [y + yaws_offset for y in yaws]
            pitch = [np.random.uniform(-np.pi / 4, np.pi / 4) for _ in range(4)]

            ## render each view
            miltiview_images = []
            for yaw, pitch in zip(yaws, pitch):
                orig = torch.tensor([
                    np.sin(yaw) * np.cos(pitch),
                    np.cos(yaw) * np.cos(pitch),
                    np.sin(pitch),
                ]).float().cuda() * 2
                fov = torch.deg2rad(torch.tensor(30)).cuda()
                extrinsics = utils3d.torch.extrinsics_look_at(orig, torch.tensor([0, 0, 0]).float().cuda(), torch.tensor([0, 0, 1]).float().cuda())
                intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
                extrinsics = extrinsics.unsqueeze(0).expand(num_samples, -1, -1)
                intrinsics = intrinsics.unsqueeze(0).expand(num_samples, -1, -1)
                render_results = self._render_batch(preds[vl], extrinsics[:,None], intrinsics[:,None])[:,0]
                miltiview_images.append(render_results['color'])

            ## Concatenate views
            miltiview_images = torch.cat([
                torch.cat(miltiview_images[:2], dim=-2),
                torch.cat(miltiview_images[2:], dim=-2),
            ], dim=-1)
            if vl == 'ref':
                ret_dict.update({f'miltiview_image_ref': {'value': miltiview_images, 'type': 'image'}})
            elif vl == 'learned_fps':
                ret_dict.update({f'miltiview_image_learned_fps': {'value': miltiview_images, 'type': 'image'}})
            else:
                ret_dict.update({f'miltiview_image_vl{vl}': {'value': miltiview_images, 'type': 'image'}})

        self.renderer.rendering_options.bg_color = 'random'
                                    
        return ret_dict
    
    @torch.no_grad()
    def run_train_val_step(self, init=False):
        """
        Run a training step.
        """
        losses = []
        # for fast start, truncate the dataset
        varlen_log_scale_list = [0] if self.varlen_max_log_scale is None else list(range(self.varlen_max_log_scale + 1))
        latent_token_varlen_log_scale_list = [0] if self.latent_token_varlen_max_log_scale is None else list(range(self.latent_token_varlen_max_log_scale + 1))
        for data in tqdm(self.dataloader_train_val, desc="Validating", disable=not self.is_master):

            data = recursive_to_device(data, self.device)
            val_loss_terms, _ = self.training_losses(**data)
            val_loss_terms = {f"train_val/{k}": v for k, v in val_loss_terms.items() if "loss" in k}
            
            
            cond = data['cond']
            cond = self.get_cond(cond)
            for latent_vl in latent_token_varlen_log_scale_list:
                latent_token_length = self._sample_tgt_latent_token_length(self.models["encoder"], latent_token_varlen_log_scale=latent_vl)
                z = self.models["encoder"](x=None, cond=cond, q_token_length=latent_token_length)
                for vl in varlen_log_scale_list:
                    # skip some combinations to reduce computation
                    if latent_vl > 0 and vl > 0:
                        continue
                    num_points = self._sample_tgt_pcd_num(vl)
                    points = OctreeProbabilityFixedlenDecoder.sample(self.models['decoder'], z, num_points=num_points, level=self.max_voxel_level, temperature=1.0, algo=self.sample_algo)['points']
                    
                    target_points = data['x_0']['points']
                    val_loss_chamfer_sqrt = chamfer_distance(points, target_points)
                    val_loss_chamfer = chamfer_distance(points, target_points, training=True)
                    val_loss_terms.update({
                        f"train_val/chamfer_sqrt_vl{vl}_latent{latent_token_length}": val_loss_chamfer_sqrt,
                        f"train_val/chamfer_vl{vl}_latent{latent_token_length}": val_loss_chamfer,
                    })
            
            losses.append(val_loss_terms)   
        
        losses = dict_reduce(losses, lambda x: torch.stack(x).mean())
        losses = dict_flatten(losses, sep='/')
        losses = dict_foreach(losses, lambda x: {"value": x, "reduce": "mean"})
        
        if self.debug:
            print("Validation losses:", losses)
        
        return losses
        
class GaussianOctreeFixlenVaeTrainer(OctreeFixlenVaeTrainer):
    """
    
    trained models:
        encoder: shared fixlen encoder
        decoder: octree fixlen decoder
        decoder_gs: gs attribute decoder
    """
    def __init__(
        self, 
        *args, 
        use_render_grad: bool = False, 
        render_logits_bn: bool = False,
        render_logits_clamp0: bool = False,
        lambda_render_logits: float = 0.0,
        lambda_opacity_reg_logits: float = 0.0,
        render_logits_lambda_floating_reg: float = 0.0,
        critic_warmup_steps: int = 0,
        lambda_offset_norm: float = 0.0,
        lambda_actor_critic: float = 0.0,
        l1c_clamp: bool = True,
        l1c_mean: bool = True,
        l1c_std: bool = False,
        l1c_normalize01: bool = False,
        l1c_clamp_percentile: float = 0.0,
        render_grad_potential_terms: List[str] = None,
        lambda_render_grad_potential: float = 1.0,
        **kwargs
    ):
        self.use_render_grad = use_render_grad
        self.render_logits_bn = render_logits_bn
        self.render_logits_clamp0 = render_logits_clamp0
        self.lambda_render_logits = lambda_render_logits
        self.lambda_opacity_reg_logits = lambda_opacity_reg_logits
        self.lambda_offset_norm = lambda_offset_norm
        self.lambda_actor_critic = lambda_actor_critic
        self.render_logits_lambda_floating_reg = render_logits_lambda_floating_reg
        self.critic_warmup_steps = critic_warmup_steps
        self.l1c_normalize01 = l1c_normalize01
        self.l1c_clamp = l1c_clamp
        self.l1c_mean = l1c_mean
        self.l1c_std = l1c_std
        self.l1c_clamp_percentile = l1c_clamp_percentile
        self.render_grad_potential_terms = render_grad_potential_terms
        self.lambda_render_grad_potential = lambda_render_grad_potential
        super().__init__(*args, **kwargs)
        
    def _init_renderer(self):
        rendering_options = {"near" : 0.8,
                             "far" : 1.6,
                             "bg_color" : 'random'}
        self.renderer = GaussianRenderer(rendering_options)
        if self.use_render_loss:
            self.renderer.pipe.kernel_size = self.models['decoder_gs'].rep_config['2d_filter_kernel_size']
        else:
            self.renderer.pipe.kernel_size = 0.1
            
    def get_opacity_reg_logits_terms(self, reps, pred_points):
        """
        Args:
            pred_mean_opacity: The [B x P] tensor of predicted mean opacity.
            pred_points:
                log_probs: The [B x P] tensor of log probabilities of the points.
        """
        terms = edict(loss = 0.0)
        # get mean opacity
        with torch.no_grad():
            B = len(reps)
            opacity_list = []
            for rep in reps:
                opacity_list.append(rep.get_opacity)
            opacity = torch.stack(opacity_list, dim=0) # [B x PC]
            pred_mean_opacity = opacity.view(B, -1, self.models['decoder_gs'].rep_config['num_gaussians']).mean(dim=-1)
            
            opacity = pred_mean_opacity - pred_mean_opacity.mean(dim=-1, keepdim=True) # [B x P]
        loss_opacity_reg_logits = -opacity * pred_points['log_probs'] # [B x P]
        terms['loss_opacity_reg_logits'] = loss_opacity_reg_logits.mean() # mean: consistent with reg loss
        terms['loss'] = terms['loss'] + self.lambda_opacity_reg_logits * terms['loss_opacity_reg_logits']
        return terms

    def _collect_gaussian_params(self, reps, force_grad=False):
        param_tensors = []
        indices = [] 
        for i, g in enumerate(reps):
            for name in ['xyz', 'features_dc', 'features_rest', 'opacity', 'scaling', 'rotation']:
                val = getattr(g, name)
                if val is not None:
                    if force_grad and not val.requires_grad:
                        val.requires_grad_(True)
                    
                    if val.requires_grad:
                        param_tensors.append(val)
                        indices.append((i, name))
        return param_tensors, indices

    def _compute_potential_stats(self, reps, grads, indices):
        stats = {}
        grads_by_rep = [{} for _ in range(len(reps))]
        for idx, (rep_idx, name) in enumerate(indices):
            grads_by_rep[rep_idx][name] = grads[idx]
        
        all_term_pos = []
        all_term_color = []
        all_term_opacity = []
        
        potential_list = []
        
        for i, g in enumerate(reps):
            g_potential = 0.0
            g_grads = grads_by_rep[i]
            if 'pos' in self.render_grad_potential_terms and 'xyz' in g_grads:
                grad_xyz = g_grads['xyz']
                scaling = g.scaling
                term_pos = -(grad_xyz.norm(dim=-1) * scaling.max(dim=-1).values)
                g_potential += term_pos
                all_term_pos.append(term_pos)
                
            if 'color' in self.render_grad_potential_terms and 'features_dc' in g_grads:
                grad_dc = g_grads['features_dc']
                term_color = -grad_dc.norm(dim=-1).squeeze(-1)
                g_potential += term_color
                all_term_color.append(term_color)
                
            if 'opacity' in self.render_grad_potential_terms and 'opacity' in g_grads:
                grad_opacity = g_grads['opacity']
                opacity = g.opacity
                term_opacity = -grad_opacity.abs().squeeze(-1) * opacity.squeeze(-1)
                g_potential += term_opacity
                all_term_opacity.append(term_opacity)
            
            potential_list.append(g_potential)
            
        if all_term_pos:
            stats['render_grad_potential_term_pos_mean'] = torch.cat(all_term_pos).mean()
        if all_term_color:
            stats['render_grad_potential_term_color_mean'] = torch.cat(all_term_color).mean()
        if all_term_opacity:
            stats['render_grad_potential_term_opacity_mean'] = torch.cat(all_term_opacity).mean()
            
        return potential_list, stats

    def _get_v4_1_surrogate_loss(self, target_loss, reps, l1_token):
        terms = edict()
        
        # Collect parameters for gradient computation
        param_tensors, indices = self._collect_gaussian_params(reps)
        
        # Compute gradients w.r.t parameters and l1_token
        # create_graph=False because we don't need second derivatives of the rendering process
        grads = torch.autograd.grad(target_loss, param_tensors + [l1_token], create_graph=False)
        grads_reps = grads[:len(param_tensors)]
        grad_l1_token = grads[len(param_tensors)]
        
        # Compute Potential Terms if requested
        potential = torch.zeros_like(l1_token)
        if self.render_grad_potential_terms:
            potential_list, stats = self._compute_potential_stats(reps, grads_reps, indices)
            terms.update(stats)
            
            for i, g_potential in enumerate(potential_list):
                if isinstance(g_potential, torch.Tensor):
                    potential[i] = g_potential

            terms['render_grad_potential_mean'] = potential.mean()
            terms['render_grad_potential_max'] = potential.max()
        
        # Construct Surrogate Loss
        surrogate_loss = 0.0
        
        # 1. Parameter gradients (RecLoss)
        for g_grad, p in zip(grads_reps, param_tensors):
            surrogate_loss += (g_grad.detach() * p).sum()
            
        # 2. L1 Token gradients (L1 Contribution)
        surrogate_loss += (grad_l1_token.detach() * l1_token).sum()
        
        # 3. Potential Term (Reward Shaping)
        if self.render_grad_potential_terms:
            render_grad_potential_loss = (potential.detach() * l1_token).sum()
            terms['render_grad_potential_loss'] = render_grad_potential_loss
            surrogate_loss += self.lambda_render_grad_potential * render_grad_potential_loss
            
        return surrogate_loss, terms
          
    def get_render_logits_loss_terms(self, pred_gs: TensorDict, pred_points: TensorDict, d_loss: torch.Tensor, points_gt: Optional[torch.Tensor] = None) -> Dict:
        """
        Args:
            pred_gs:
                features: The [B x P x C] tensor of point features.
            pred_points:
                log_probs: The [B x P] tensor of log probabilities of the points.
            d_loss: The [B x P] tensor of estimated loss improvement brought by each point.
            points_gt: Optional. The [B x P x 3] tensor of ground truth points.
        """
        terms = edict(loss = 0.0)
        log_probs = pred_points['log_probs'] # [B x P]
        B, num_gs = log_probs.shape

        assert num_gs > 1, "num_gs should be greater than 1 to compute render logits loss"
        bad_rate = (d_loss > 0.0).float().mean(dim=-1, keepdim=True) # [B x 1]
        if self.render_logits_clamp0:
            # all points do not increase the loss
            d_loss = d_loss.clamp(max=0.0)

        # floating reg
        if self.render_logits_lambda_floating_reg > 0.0:
            pcds = pred_points['points'] # [B x P x 3]
            gt_pcds = points_gt # [B x P x 3]
            min_dist = torch.cdist(pcds, gt_pcds, p=2.0, compute_mode='donot_use_mm_for_euclid_dist').min(dim=-1)[0] # [B x P]
            quantile = torch.quantile(min_dist, q=0.9, dim=-1, keepdim=True) # [B x 1]
            # FIXME: 10 factor
            floating_d_loss = ((min_dist - quantile * 10) > 0).float() / (B * num_gs) 
            d_loss = d_loss + self.render_logits_lambda_floating_reg * floating_d_loss
            
        # get d loss mean for advantage
        d_loss_mean = d_loss.mean(dim=-1, keepdim=True)
        d_loss_advantage = num_gs / (num_gs - 1) * (d_loss - d_loss_mean) # [B x P], compute the advantage, scale by n/(n-1) to make it unbiased
        if self.render_logits_bn:
            eps = 1e-10
            d_loss_advantage_norm = torch.linalg.vector_norm(d_loss_advantage, dim=-1, keepdim=True)
            d_loss_advantage = d_loss_advantage / (d_loss_advantage_norm + eps)
        render_logits_loss = (d_loss_advantage.detach() * log_probs).sum()
        terms['render_logits_loss'] = render_logits_loss
        terms['render_logits_d_loss_mean'] = d_loss_mean.mean()
        terms['render_loss_improvement_estimated'] = d_loss.sum() # estimiated loss improvement bringed by the points 
        terms['render_logits_d_loss_min'] = d_loss.min()
        terms['render_logits_d_loss_max'] = d_loss.max()
        terms['render_logits_bad_rate'] = bad_rate.mean()
        lambda_render_logits = self.lambda_render_logits if self.geo_warmup_steps < self.step else 0.0
        terms['loss'] = terms['loss'] + lambda_render_logits * terms['render_logits_loss']
        return terms

    @torch.no_grad()
    def _get_pred_points_status(self, points_pred: TensorDict) -> Dict:
        status = edict()
        if 'log_probs' in points_pred:
            status['pred_points_log_probs_mean'] = points_pred['log_probs'].mean()
            status['pred_points_log_probs_min'] = points_pred['log_probs'].min()
            status['pred_points_log_probs_max'] = points_pred['log_probs'].max()
        return status
    
    
    def _render_batch_with_l1c(
        self, 
        reps: List[Gaussian], 
        extrinsics: torch.Tensor, 
        intrinsics: torch.Tensor, 
        gt_image=None, 
        gt_alpha=None,
        l1_token=None, 
        return_opacity: bool = False, 
        return_depth: bool = False,
    ) -> torch.Tensor:
        """
        Render a batch of representations.

        Args:
            reps: The dictionary of lists of representations.
            extrinsics: The [B x Nv x 4 x 4] tensor of extrinsics.
            intrinsics: The [B x Nv x 3 x 3] tensor of intrinsics.
            
            gt_image: The [B x Nv x 3 x H x W] tensor of ground truth images.
            gt_alpha: The [B x Nv x H x W] tensor of ground truth alpha masks.
            l1_token: The [B x N_gs] tensor of l1 tokens.
        """
        B, Nv, _, _ = extrinsics.shape
        ret = []
        for i, representation in enumerate(reps):
            ret_nv = []
            for nv in range(Nv):
                render_pack = self.renderer.render(
                    representation, 
                    extrinsics[i, nv], 
                    intrinsics[i, nv], 
                    gt_image=gt_image[i, nv], 
                    gt_alpha=gt_alpha[i, nv],
                    l1_token=l1_token[i], 
                    return_opacity=return_opacity, 
                    return_depth=return_depth, 
                )
                render_pack['bg_color'] = self.renderer.bg_color
                ret_nv.append(TensorDict(render_pack))
            ret.append(tensordict.stack(ret_nv, dim=0))
        ret = tensordict.stack(ret, dim=0)  # [B x Nv x ...]
        return ret

    def _get_rendered_image_and_gt_image_with_l1c(self, reps, target_images, l1_token):
        # reps = pcds (input data) -> gaussians
        self.renderer.rendering_options.resolution = target_images["images"].shape[-1]
        render_results = self._render_batch_with_l1c(
            reps, target_images["extrinsics"], target_images["intrinsics"], 
            gt_image=target_images["images"], 
            gt_alpha=target_images["alphas"],
            l1_token=l1_token, 
            return_depth=self.lambda_depth > 0, return_opacity=self.lambda_alpha > 0)     
        
        rec_image = render_results['color']
        gt_image = target_images["images"] * target_images["alphas"][..., None, :, :] + (1 - target_images["alphas"][..., None, :, :]) * render_results['bg_color'][..., None, None]
        
        return render_results, rec_image, gt_image

    def get_render_loss_terms_with_l1c(self, reps, target_images, l1_token) -> Dict:
        terms = edict(loss = 0.0)
        
        render_results, rec_image, gt_image = self._get_rendered_image_and_gt_image_with_l1c(reps, target_images, l1_token)
        rec_loss_terms = self.get_rec_loss_terms(rec_image, gt_image)
        other_rec_loss_terms = self.get_other_rec_loss_terms(reps, render_results, target_images)
        terms = terms_merge(terms, rec_loss_terms, other_rec_loss_terms)
        # get l1 map (B Nv C H W)
        l1_map = render_results['l1_map']
        loss_render_logit = l1_map.mean()
        terms['render_logits_loss'] = loss_render_logit
        terms['loss'] = terms['loss'] + self.lambda_render_logits * terms['render_logits_loss']
        
        return terms
        
    def training_losses(
        self,
        x_0: TensorDict,
        target_images: TensorDict,
        cond: Optional[TensorDict] = None,
        **kwargs
    ) -> Tuple[Dict, Dict]:
        """
        Compute training losses.

        Args:
            x_0: empty now
            cond:
                - points: The [B x P x 3] tensor of point clouds.
                - feats: The [B x P x C] tensor of point features.
                - cond_hidden_states_dino: The [B x Lv x C] tensor of image features.
                - ray_embeddings: The [B x Lv x 6] tensor of ray embeddings.
            target_images:
                image: The [B x Nv x 3 x H x W] tensor of images.
                alphas: The [B x Nv x H x W] tensor of alphas channels.
                extrinsics: The [B x Nv x 4 x 4] tensor of extrinsics.
                intrinsics: The [B x Nv x 3 x 3] tensor of intrinsics.
            return_aux: Whether to return auxiliary information.

        Returns:
            a dict with the key "loss" containing a scalar tensor.
            may also contain other keys for different terms.
        """
        cond = self.get_cond(cond)
        latent_token_length = self._sample_tgt_latent_token_length(self.models["encoder"])
        z, mean, logvar = self.training_models["encoder"](x=None, cond=cond, sample_posterior=True, return_raw=True, q_token_length=latent_token_length)
        num_points = self._sample_tgt_pcd_num()
        
        terms = edict(loss = 0.0)
        status = edict()
       
        if self.use_render_loss:
            if self.use_render_grad and torch.is_grad_enabled(): # avoid bugs in no grad run (e.g. validation step)
                # l1c version
                points_pred = OctreeProbabilityFixedlenDecoder.sample(self.training_models['decoder'], z, num_points=num_points, level=self.max_voxel_level, temperature=1.0, algo=self.sample_algo)
                preds = self.training_models['decoder_gs'](x=points_pred, cond=z)
                reps, status_gs = self.models['decoder_gs'].to_representation(points_pred, preds, return_status=True)
                
                # prepare l1_token by expand k times
                log_probs = GradNormalization.apply(points_pred['log_probs'], self.l1c_clamp, self.l1c_mean, self.l1c_std, self.l1c_normalize01, self.l1c_clamp_percentile)
                l1_token = log_probs.repeat_interleave(self.models['decoder_gs'].rep_config['num_gaussians'], dim=1)
                terms_render = self.get_render_loss_terms_with_l1c(reps, target_images, l1_token)
                terms_opacity_reg_logits = self.get_opacity_reg_logits_terms(reps, points_pred)
                terms = terms_merge(terms, terms_render, terms_opacity_reg_logits)
            else:
                with torch.no_grad():
                    points_pred = OctreeProbabilityFixedlenDecoder.sample(self.training_models['decoder'], z, num_points=num_points, level=self.max_voxel_level, temperature=1.0, algo=self.sample_algo)
                preds = self.training_models['decoder_gs'](x=points_pred, cond=z)
                reps, status_gs = self.models['decoder_gs'].to_representation(points_pred, preds, return_status=True)
                terms_render = self.get_render_loss_terms(reps, target_images)
                terms = terms_merge(terms, terms_render)
            
            offset = self.models['decoder_gs']._get_offset(preds['features'])
            offset_reg_loss, reg_terms = self.get_offset_reg(points_pred['points'], offset)
            terms['loss'] = terms['loss'] + offset_reg_loss
            terms.update(reg_terms)
            status.update(status_gs)
            status.update(self._get_status(reps))
        status.update(self._get_pred_points_status(points_pred))
        status.update(self._get_latent_status(z, mean=mean, logvar=logvar))
        
        terms_prob = self.get_geo_prob_loss_terms(x_0, z)
        for k, v in terms_prob.items():
            if k in terms:
                terms[k] = terms[k] + v
            else:
                terms[k] = v

        # kl
        terms["loss_kl"] = 0.5 * torch.mean(mean.pow(2) + logvar.exp() - logvar - 1)
        terms["loss"] = terms["loss"] + self.lambda_kl * terms["loss_kl"]

        return terms, status
    
    @torch.no_grad()
    def run_train_val_step(self, init=False):
        """
        Run a training step.
        """
        losses = []
        # for fast start, truncate the dataset
        varlen_log_scale_list = [0] if self.varlen_max_log_scale is None else list(range(self.varlen_max_log_scale + 1))
        latent_token_varlen_log_scale_list = [0] if self.latent_token_varlen_max_log_scale is None else list(range(self.latent_token_varlen_max_log_scale + 1))
        for data in tqdm(self.dataloader_train_val, desc="Validating", disable=not self.is_master):
            data = recursive_to_device(data, self.device)
            if self.debug:
                val_loss_terms = {}
            else:
                val_loss_terms, _ = self.training_losses(**data)
                val_loss_terms = {f"train_val/{k}": v for k, v in val_loss_terms.items() if "loss" in k}
            
            cond = self.get_cond(data['cond'])
            for latent_vl in latent_token_varlen_log_scale_list:
                latent_token_length = self._sample_tgt_latent_token_length(self.models["encoder"], latent_token_varlen_log_scale=latent_vl)
                z = self.models["encoder"](x=None, cond=cond, q_token_length=latent_token_length)
                for vl in varlen_log_scale_list:
                    # skip some combinations to reduce computation
                    if latent_vl > 0 and vl > 0:
                        continue
                    num_points = self._sample_tgt_pcd_num(vl)
                    points_pred = OctreeProbabilityFixedlenDecoder.sample(self.models['decoder'], z, num_points=num_points, level=self.max_voxel_level, temperature=1.0, algo=self.sample_algo)
                    pred = self.models['decoder_gs'](x=points_pred, cond=z)
                    reps = self.models["decoder_gs"].to_representation(points_pred, pred)
                    
                    target_points = data['x_0']['points']
                    val_loss_chamfer_sqrt = chamfer_distance(points_pred['points'], target_points)
                    val_loss_chamfer = chamfer_distance(points_pred['points'], target_points, training=True)
                    val_loss_terms.update({
                        f"train_val/chamfer_sqrt_vl{vl}_latent{latent_token_length}": val_loss_chamfer_sqrt,
                        f"train_val/chamfer_vl{vl}_latent{latent_token_length}": val_loss_chamfer,
                    })
                    
                    target_images = data['target_images']
                    extrinsics = target_images['extrinsics']
                    intrinsics = target_images['intrinsics']
                    
                    # render
                    self.renderer.rendering_options.bg_color = (0, 0, 0)
                    self.renderer.rendering_options.resolution = target_images["images"].shape[-1]

                    ret = self._render_batch(reps, extrinsics, intrinsics)
                    
                    # compute image loss
                    ret_colors = ret['color'].flatten(0, 1)  # [B*Nv, H, W, 3]
                    tgt_colors = target_images["images"].flatten(0, 1)  # [B*Nv, H, W, 3]
                    tgt_alphas = target_images["alphas"].flatten(0, 1)  # [B*Nv, H, W, 1]
                    bg_color = ret['bg_color'].flatten(0, 1)  # [B*Nv, 3]
                    
                    tgt_colors = tgt_colors * tgt_alphas[:, None] + bg_color[...,None,None] * (1 - tgt_alphas[:, None])
                    
                    val_loss_psnr = psnr(ret_colors, tgt_colors)
                    val_loss_ssim = ssim(ret_colors, tgt_colors)
                    val_loss_lpips = lpips(ret_colors, tgt_colors)
                    
                    val_loss_terms.update({
                        f"train_val/psnr_vl{vl}_latent{latent_token_length}": val_loss_psnr,
                        f"train_val/ssim_vl{vl}_latent{latent_token_length}": val_loss_ssim,
                        f"train_val/lpips_vl{vl}_latent{latent_token_length}": val_loss_lpips,
                    })
                    
                    if self.use_render_grad:
                        with torch.enable_grad():
                            log_probs = points_pred['log_probs'].detach()
                            log_probs.requires_grad_(True)
                            l1_token = log_probs.repeat_interleave(self.models['decoder_gs'].rep_config['num_gaussians'], dim=1)
                            
                            self.renderer.rendering_options.bg_color = (0, 0, 0)
                            self.renderer.rendering_options.resolution = target_images["images"].shape[-1]
                            render_results = self._render_batch_with_l1c(
                                reps, target_images["extrinsics"], target_images["intrinsics"], 
                                gt_image=target_images["images"], 
                                gt_alpha=target_images["alphas"],
                                l1_token=l1_token, 
                            )
                            l1_map = render_results['l1_map']
                            loss_render_logit = l1_map.mean()
                            loss_render_logit.backward()
                            dloss = log_probs.grad
                            bad_rate = (dloss > 0.0).float().mean()
                            val_loss_terms[f"train_val/render_logits_bad_rate_vl{vl}_latent{latent_token_length}"] = bad_rate.detach()

                            if latent_vl == 0 and vl == 0:
                                # record more statistic of dloss
                                val_loss_terms.update({
                                    f"train_val/dloss_mean_vl{vl}_latent{latent_token_length}": dloss.mean(),
                                    f"train_val/dloss_max_vl{vl}_latent{latent_token_length}": dloss.max(),
                                    f"train_val/dloss_min_vl{vl}_latent{latent_token_length}": dloss.min(),
                                })

            losses.append(val_loss_terms)   
        
        losses = dict_reduce(losses, lambda x: torch.stack(x).mean())
        losses = dict_flatten(losses, sep='/')
        losses = dict_foreach(losses, lambda x: {"value": x, "reduce": "mean"})
        if self.debug:
            print("Validation losses:", losses)
        
        return losses
        
    @torch.no_grad()
    def run_snapshot(
        self,
        num_samples: int,
        batch_size: int,
        verbose: bool = False,
    ) -> Dict:
        dataloader = iter(DataLoader(
            copy.deepcopy(self.dataset_train_val if self.dataset_train_val is not None else self.dataset),
            batch_size=batch_size,
            shuffle=False if self.debug else True,
            num_workers=0,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
        ))
        # inference
        ret_dict = {}
        varlen_log_scale_list = [0] if self.varlen_max_log_scale is None else list(range(self.varlen_max_log_scale + 1))
        target_images = []
        reps = {vl: [] for vl in varlen_log_scale_list}
        pcd_reps = {vl: [] for vl in varlen_log_scale_list}
        additional_keys = ['ref', 'log_p']
        graybg_keys = []
        if self.use_render_grad:
            log_probs = []
            dloss_points = []
            additional_keys.extend(['dloss', 'dloss_geq_0', 'dloss_leq_0', 'dloss_top10pct', 'dloss_bottom10pct', 'rec_no_geq_0', 'rec_no_leq_0'])
            graybg_keys.extend(['dloss',])
            # additional_keys.extend(['dloss'])

        pcd_reps.update({k: [] for k in additional_keys})
        conds = []
        for i in range(0, num_samples, batch_size):
            batch = min(batch_size, num_samples - i)
            data = next(dataloader)
            args = {k: v[:batch] for k, v in data.items()}
            args = recursive_to_device(args, self.device, non_blocking=True)
            target_images.append(args['target_images'])
            cond = args['cond'] if 'cond' in args else None
            if cond is not None:
                conds.append(cond)
            
            cond = self.get_cond(cond)
            z = self.models["encoder"](x=None, cond=cond)
            for vl in varlen_log_scale_list:
                num_points = self._sample_tgt_pcd_num(vl)
                points_pred = OctreeProbabilityFixedlenDecoder.sample(self.models['decoder'], z, num_points=num_points, level=self.max_voxel_level, temperature=1.0, algo=self.sample_algo)
                pred = self.training_models['decoder_gs'](x=points_pred, cond=z)
                reps[vl].extend(self.models["decoder_gs"].to_representation(points_pred, pred))
                pcd_reps[vl].extend(self.pcd_to_representation(points_pred['points']))
                if vl == 0:
                    log_p_normalized = points_pred['log_probs'][..., None]
                    log_p_normalized = (log_p_normalized - log_p_normalized.min()) / (log_p_normalized.max() - log_p_normalized.min())
                    pcd_reps['log_p'].extend(self.pcd_to_representation(points_pred['points'], log_p_normalized * 2 - 1, render_mode='custom'))
                    log_probs.append(points_pred['log_probs'])
                    dloss_points.append(points_pred['points'])

            target_points = args['x_0']['points']
            pcd_reps['ref'].extend(self.pcd_to_representation(target_points))

        target_images = tensordict.cat(target_images, dim=0)
        resolution = target_images["images"].shape[-1]
        gt_images = target_images["images"] * target_images["alphas"][..., None, :, :] 
        gt_images = einops.rearrange(gt_images, 'b nv c h w -> c (b h) (nv w)')[None]
        ret_dict.update({f'rec_image_gt': {'value': gt_images, 'type': 'image'}})
        
        if self.use_render_grad:
            with torch.enable_grad():
                log_probs = torch.cat(log_probs, dim=0)
                dloss_points = torch.cat(dloss_points, dim=0)
                log_probs.requires_grad_(True)
                l1_token = log_probs.repeat_interleave(self.models['decoder_gs'].rep_config['num_gaussians'], dim=1)

                # backward rendering
                self.renderer.rendering_options.bg_color = (0, 0, 0)
                self.renderer.rendering_options.resolution = resolution
                render_results = self._render_batch_with_l1c(
                    reps[0], target_images["extrinsics"], target_images["intrinsics"], 
                    gt_image=target_images["images"], 
                    gt_alpha=target_images["alphas"],
                    l1_token=l1_token, 
                )
                l1_map = render_results['l1_map']
                loss_render_logit = l1_map.mean()
                loss_render_logit.backward()
                dloss = log_probs.grad.clone()

            # normalize for vis
            for b in range(dloss.shape[0]):
                dloss_b = dloss[b]
                points_b = dloss_points[b]
                mask_geq_0 = (dloss_b >= 0)
                mask_leq_0 = (dloss_b <= 0)
                pcd_reps['dloss_geq_0'].extend(self.pcd_to_representation(points_b[mask_geq_0][None]))
                pcd_reps['dloss_leq_0'].extend(self.pcd_to_representation(points_b[mask_leq_0][None]))
                # top 10 pct
                k_top = max(1, int(0.1 * points_b.shape[0]))
                k_bottom = max(1, int(0.1 * points_b.shape[0]))
                topk_values, topk_indices = torch.topk(dloss_b, k=k_top, largest=True)
                bottomk_values, bottomk_indices = torch.topk(dloss_b, k=k_bottom, largest=False)
                pcd_reps['dloss_top10pct'].extend(self.pcd_to_representation(points_b[topk_indices][None]))
                pcd_reps['dloss_bottom10pct'].extend(self.pcd_to_representation(points_b[bottomk_indices][None]))

                # visualize
                gs_base = reps[0][b]
                from copy import deepcopy
                # geq 0
                gs_vis = deepcopy(gs_base)
                org_opacity = gs_vis.get_opacity
                org_opacity[mask_geq_0.repeat_interleave(self.models['decoder_gs'].rep_config['num_gaussians'])] = 0.0
                gs_vis.set_opacity(org_opacity)
                pcd_reps['rec_no_geq_0'].append(gs_vis)
                # leq 0
                gs_vis = deepcopy(gs_base)
                org_opacity = gs_vis.get_opacity
                org_opacity[mask_leq_0.repeat_interleave(self.models['decoder_gs'].rep_config['num_gaussians'])] = 0.0
                gs_vis.set_opacity(org_opacity)
                pcd_reps['rec_no_leq_0'].append(gs_vis)
                
            
            dloss = dloss / dloss.abs().mean()
            dloss_reps = self.pcd_to_representation(dloss_points, dloss, render_mode='custom', scale=0.002)
            pcd_reps['dloss'].extend(dloss_reps)
            
        if len(conds) > 0:
            cond_vis = self.vis_cond(tensordict.cat(conds, dim=0))
            ret_dict.update(cond_vis)

        for vl in varlen_log_scale_list:
            # render single view
            self.renderer.rendering_options.bg_color = (0, 0, 0)
            self.renderer.rendering_options.resolution = resolution
            render_results = self._render_batch(reps[vl], target_images["extrinsics"], target_images["intrinsics"])
            rec_images = einops.rearrange(render_results['color'], "b nv c h w -> c (b h) (nv w)")[None]
            ret_dict.update({f'rec_image_vl{vl}': {'value': rec_images, 'type': 'image'}})
            
        for vl in pcd_reps:
            # render single view
            if vl in graybg_keys:
                self.renderer.rendering_options.bg_color = (0.5, 0.5, 0.5)
            else:
                self.renderer.rendering_options.bg_color = (0, 0, 0)
            self.renderer.rendering_options.resolution = resolution
            render_results = self._render_batch(pcd_reps[vl], target_images["extrinsics"], target_images["intrinsics"])
            rec_images = einops.rearrange(render_results['color'], "b nv c h w -> c (b h) (nv w)")[None]
            if vl in additional_keys:
                ret_dict.update({f'rec_pcd_image_{vl}': {'value': rec_images, 'type': 'image'}})
            else:
                ret_dict.update({f'rec_pcd_image_vl{vl}': {'value': rec_images, 'type': 'image'}})

        self.renderer.rendering_options.bg_color = 'random'

        return ret_dict

