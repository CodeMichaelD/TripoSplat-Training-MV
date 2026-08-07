from typing import Optional
import os
import tempfile

import numpy as np
import torch
import yaml
from tensordict import TensorDict

from .base import Pipeline
from .samplers import FlowEulerCfgSampler
from .. import models
from ..models import OctreeProbabilityFixedlenDecoder
from ..trainers.flow_matching.mixins.image_conditioned import ImageConditionedMixin
from ..datasets.latent_seq import GSOctreeLatentVisMixin
from ..utils.general_utils import get_noise_like
from ..utils.hf_utils import DEFAULT_HF_DENOISER_FILE, load_state_dict_file, select_model_state_dict


class DEGImageTo3DPipeline(ImageConditionedMixin, GSOctreeLatentVisMixin, Pipeline):
    """
    Image-to-3D generation pipeline using flow matching over Gaussian latents.

    Wraps the denoiser model, image encoder, and VAE decoder into a single
    callable object.  Checkpoint paths that were baked into training configs
    can be overridden at construction time via ``decoder_path`` /
    ``decoder_ckpt`` so callers never need to edit yaml files.

    Args:
        config_path:   Path to a generation config yaml
                       (e.g. ``configs/generation/latent1k-...yaml``).
        ckpt_path:     Path to a denoiser checkpoint (``.pt`` file).
        decoder_path:  Override ``dataset.args.decoder_path`` in the config.
        decoder_ckpt:  Override ``dataset.args.decoder_ckpt`` in the config
                       (e.g. ``"step0400000"``).
    """

    def __init__(
        self,
        config_path: str,
        ckpt_path: str,
        decoder_path: Optional[str] = None,
        decoder_ckpt: Optional[str] = None,
        decoder_config_path: Optional[str] = None,
    ):
        self._tmp_config = None

        # Patch decoder paths into config if overrides are provided
        if decoder_path is not None or decoder_ckpt is not None or decoder_config_path is not None:
            with open(config_path, 'r') as f:
                cfg = yaml.safe_load(f)
            for section in ('dataset', 'train_val_dataset'):
                if section in cfg and 'args' in cfg[section]:
                    if decoder_path is not None:
                        cfg[section]['args']['decoder_path'] = decoder_path
                        if decoder_ckpt is None and not os.path.exists(os.path.join(decoder_path, 'config.json')):
                            cfg[section]['args']['decoder_ckpt'] = None
                    if decoder_ckpt is not None:
                        cfg[section]['args']['decoder_ckpt'] = decoder_ckpt
                    if decoder_config_path is not None:
                        cfg[section]['args']['decoder_config_path'] = decoder_config_path
            tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
            yaml.dump(cfg, tmp)
            tmp.close()
            self._tmp_config = tmp.name
            config_path = self._tmp_config

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.normalization = None

        # ImageConditionedMixin
        image_cond_model_name = (
            self.config.get('trainer', {})
                       .get('args', {})
                       .get('image_cond_model', 'dinov2_vitl14_reg')
        )
        ImageConditionedMixin.__init__(self, image_cond_model=image_cond_model_name)

        # GSOctreeLatentVisMixin
        dataset_args = self.config.get('dataset', {}).get('args', {})
        self.image_size = dataset_args.get('image_size', 1024)
        mixin_kwargs = {k: v for k, v in {
            'decode_mode':       dataset_args.get('decode_mode', 'pcd'),
            'pretrained_decoder': dataset_args.get('pretrained_decoder'),
            'decoder_path':      dataset_args.get('decoder_path'),
            'decoder_ckpt':      dataset_args.get('decoder_ckpt'),
            'decoder_config_path': dataset_args.get('decoder_config_path'),
            'max_voxel_level':   dataset_args.get('max_voxel_level', 8),
            'max_sampled_points': dataset_args.get('max_sampled_points', 8192),
        }.items() if v is not None}
        GSOctreeLatentVisMixin.__init__(self, **mixin_kwargs)

        # Denoiser model
        denoiser_cfg = self.config['models']['denoiser']
        denoiser = getattr(models, denoiser_cfg['name'])(**denoiser_cfg['args'])

        # Pipeline base (registers self.models dict)
        Pipeline.__init__(self, models={'denoiser': denoiser})
        self.model = self.models['denoiser']

        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self._device).eval()

        if ckpt_path:
            self._validate_checkpoint_config(ckpt_path)
            self._load_checkpoint(ckpt_path)

        self.sigma_min = self.config['trainer']['args'].get('sigma_min', 1e-5)

    def __del__(self):
        if self._tmp_config and os.path.exists(self._tmp_config):
            os.unlink(self._tmp_config)

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------

    def _validate_checkpoint_config(self, ckpt_path: str) -> None:
        if DEFAULT_HF_DENOISER_FILE not in ckpt_path:
            return
        q_token_length = self.config['models']['denoiser']['args'].get('q_token_length')
        latent_token_length = self.config.get('dataset', {}).get('args', {}).get('latent_token_length')
        if q_token_length != 8192 or latent_token_length != 8192:
            raise ValueError(
                "The released VAST-AI/TripoSplat denoiser expects the 8k latent config "
                "(q_token_length=8192, latent_token_length=8192). Use "
                "configs/dit/latent8k-latentseq_flow_img_s3dit-L.yaml or omit --config in inference_gs.py."
            )

    def _load_checkpoint(self, ckpt_path: str) -> None:
        print(f"Loading checkpoint: {ckpt_path}")
        state_dict = load_state_dict_file(ckpt_path, map_location='cpu', weights_only=True)
        cleaned = select_model_state_dict(
            state_dict,
            model_name='denoiser',
            valid_keys=self.model.state_dict().keys(),
        )
        model_state = self.model.state_dict()
        if 'pos_pe' not in cleaned and 'pos_pe' in model_state:
            cleaned['pos_pe'] = model_state['pos_pe']
        missing, unexpected = self.model.load_state_dict(cleaned, strict=True)
        print(f"Checkpoint loaded — missing: {len(missing)}, unexpected: {len(unexpected)}")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def run(
        self,
        image,
        seed: int = 42,
        steps: int = 50,
        cfg_strength: float = 7.0,
        rescale_t: float = 3.0,
    ):
        """
        Generate a 3D Gaussian from a single image.

        Args:
            image: PIL Image or path string.
            seed:  Random seed.
            steps: Number of diffusion steps.
            cfg_strength: Classifier-free guidance strength.
            rescale_t:    Time-step rescaling factor.

        Returns:
            gaussian:     Gaussian splatting representation.
            pcd_gaussian: Point-cloud Gaussian (geometry only).
            camera:       Predicted camera latent tensor, or None.
        """
        if isinstance(image, str):
            from ..utils.render_utils import load_condition_image_tensor
            image = load_condition_image_tensor(image, image_size=self.image_size).unsqueeze(0).to(self._device)

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Encode image condition
        cond_input = image if isinstance(image, torch.Tensor) else [image]
        cond_out = self.get_inference_cond(cond_input)
        cond     = cond_out['cond']
        neg_cond = cond_out['neg_cond']

        # Build noise
        noise_latent = torch.randn(
            1, self.model.q_token_length, self.model.in_channels,
            device=self._device,
        )
        noise = TensorDict({'latent': noise_latent}, batch_size=1, device=self._device)
        if self.model.cam_channels is not None:
            noise['camera'] = torch.randn(1, 1, self.model.cam_channels, device=self._device)

        # Sample
        sampler = FlowEulerCfgSampler(sigma_min=self.sigma_min)
        out = sampler.sample(
            self.model, noise,
            cond=cond, neg_cond=neg_cond,
            steps=steps, cfg_strength=cfg_strength,
            rescale_t=rescale_t, verbose=False,
        )

        samples = out.samples['latent']
        camera  = out.samples.get('camera', None)

        # Decode GS
        gaussian = self._decode_latent(samples, mode='gs')[0]

        # Decode PCD
        pcd_gaussian = self._decode_latent(samples, mode='pcd')[0]

        return gaussian, pcd_gaussian, camera

    @torch.no_grad()
    def _decode_latent(self, z, mode: str = 'gs', batch_size: int = 4):
        """Decode latent tokens into representations."""
        original_mode = self.decode_mode
        self.decode_mode = mode
        self._loading_decoder()

        reps = []
        if self.normalization is not None:
            z = z * self.std.to(z.device) + self.mean.to(z.device)

        for i in range(0, z.shape[0], batch_size):
            z_batch = z[i:i + batch_size]
            points_pred = OctreeProbabilityFixedlenDecoder.sample(
                self.decoder, z_batch,
                num_points=self.max_sampled_points,
                level=self.max_voxel_level,
                temperature=1.0, algo='systematic',
            )
            if mode == 'gs':
                pred = self.decoder_gs(x=points_pred, cond=z_batch)
                reps.extend(self.decoder_gs.to_representation(points_pred, pred))
            else:
                reps.extend(self.pcd_to_representation(points_pred['points']))

        self.decode_mode = original_mode
        return reps
