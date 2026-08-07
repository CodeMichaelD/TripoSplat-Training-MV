from typing import *
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from PIL import Image
import os
from tensordict import TensorDict

from ....utils import dist_utils
from ....utils.general_utils import get_zeros_like
from ....utils.hf_utils import DEFAULT_FLUX2_VAE_PATH, DEFAULT_DINOV3_PATH, hf_local_files_only


class ImageConditionedMixin:
    """
    Mixin for image-conditioned models.
    
    Args:
        image_cond_model: The image conditioning model.
    """
    def __init__(self, *args, image_cond_model: str = 'dinov2_vitl14_reg', **kwargs):
        """
        Initialize the image conditioned mixin.
        
        Args:
            image_cond_model: The image conditioning model. Defaults to 'dinov2_vitl14_reg'. 
                Supported models: 'dinov2_vitl14_reg', 'dinov3_vith16plus', 'flux2_dev_vae'.
                Supported mixture mode: 'dinov3_flux2_concat_512', 'dinov3_flux2_concat_1024'.
        """
        super().__init__(*args, **kwargs)
        self.image_cond_model_name = image_cond_model
        self.image_cond_model = None     # the model is init lazily
        
    @staticmethod
    def prepare_for_training(image_cond_model: str, **kwargs):
        """
        Prepare for training.
        """
        if hasattr(super(ImageConditionedMixin, ImageConditionedMixin), 'prepare_for_training'):
            super(ImageConditionedMixin, ImageConditionedMixin).prepare_for_training(**kwargs)

        models_to_load = []
        if 'dinov3_flux2_concat' in image_cond_model:
            models_to_load = ['dinov3_vith16plus', 'flux2_dev_vae']
        else:
            models_to_load = [image_cond_model]

        for model_name in models_to_load:
            if model_name == 'dinov3_vith16plus':
                from transformers import AutoModel
                pretrained_model_name = os.environ.get(
                    "DINO_V3_PATH",
                    DEFAULT_DINOV3_PATH,
                )
                AutoModel.from_pretrained(pretrained_model_name, local_files_only=hf_local_files_only())
            elif model_name == 'flux2_dev_vae':
                from diffusers.models import AutoencoderKLFlux2
                AutoencoderKLFlux2.from_pretrained(
                    DEFAULT_FLUX2_VAE_PATH, subfolder="vae", torch_dtype=torch.bfloat16
                )
            else:
                torch.hub.load('facebookresearch/dinov2', model_name, pretrained=True)
        
    def _load_single_model(self, model_name):
        if model_name == 'dinov3_vith16plus':
            from transformers import AutoModel

            pretrained_model_name = os.environ.get(
                "DINO_V3_PATH",
                DEFAULT_DINOV3_PATH,
            )
            dinov3_model = AutoModel.from_pretrained(
                pretrained_model_name,
                local_files_only=hf_local_files_only(),
            ).eval().cuda()
            dinov3_model.requires_grad_(False)
            transform = transforms.Compose([
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            return {
                'type': 'dinov3',
                'model': dinov3_model,
                'transform': transform,
                'resolution': (512, 512),
            }
        elif model_name == 'flux2_dev_vae':
            from diffusers.models import AutoencoderKLFlux2

            flux2_vae = AutoencoderKLFlux2.from_pretrained(
                DEFAULT_FLUX2_VAE_PATH, subfolder="vae", torch_dtype=torch.bfloat16
            ).eval().cuda()
            flux2_vae.requires_grad_(False)
            return {
                'type': 'flux2_vae',
                'model': flux2_vae,
                'resolution': (512, 512),
            }
        else:
            try:
                hub_dir = torch.hub.get_dir()
                repo_dir = os.path.join(hub_dir, "facebookresearch_dinov2_main")
                dinov2_model = torch.hub.load(
                    repo_or_dir=repo_dir,
                    model=model_name,
                    source="local",
                    pretrained=True
                )
            except:
                dinov2_model = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=True)
            dinov2_model.eval().cuda()
            dinov2_model.requires_grad_(False)
            transform = transforms.Compose([
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            return {
                'type': 'dinov2',
                'model': dinov2_model,
                'transform': transform,
                'resolution': (518, 518),
            }

    def _init_image_cond_model(self):
        """
        Initialize the image conditioning model.
        """
        if 'dinov3_flux2_concat' in self.image_cond_model_name:
            model1 = self._load_single_model('dinov3_vith16plus')
            model2 = self._load_single_model('flux2_dev_vae')
            
            if self.image_cond_model_name == 'dinov3_flux2_concat_1024':
                model1['resolution'] = (1024, 1024)
                model2['resolution'] = (1024, 1024)
            if self.image_cond_model_name == 'dinov3_flux2_concat_512':
                model1['resolution'] = (512, 512)
                model2['resolution'] = (512, 512)
            self.image_cond_model = {
                'type': 'mixture',
                'models': [model1, model2]
            }
        else:
            self.image_cond_model = self._load_single_model(self.image_cond_model_name)
    
    def _encode_single_image(self, image, model_dict):
        model_type = model_dict.get('type', 'dinov2')
        resolution = model_dict.get('resolution', (518, 518))

        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
            images = image
            if images.shape[-2:] != resolution:
                images = F.interpolate(images, size=resolution, mode='bilinear', align_corners=False)
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            images = [i.resize(resolution, Image.LANCZOS) for i in image]
            images = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in images]
            images = [torch.from_numpy(i).permute(2, 0, 1).float() for i in images]
            images = torch.stack(images).cuda()
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")

        if model_type in ('dinov2', 'dinov3'):
            images = images.to(dtype=torch.float32)
            images = model_dict['transform'](images)
            if model_type == 'dinov3':
                outputs = model_dict['model'](pixel_values=images)
                last_hidden = getattr(outputs, 'last_hidden_state', None)
                if last_hidden is None:
                    last_hidden = outputs[0]
                tokens = last_hidden
                return F.layer_norm(tokens, tokens.shape[-1:])
            else:
                features = model_dict['model'](images, is_training=True)['x_prenorm']
                tokens = features
                return F.layer_norm(tokens, tokens.shape[-1:])

        if model_type == 'flux2_vae':
            images = images.to(dtype=torch.bfloat16)
            images = images * 2 - 1
            features = model_dict['model'].encode(images)
            latents = features.latent_dist.sample()

            def _patchify_latents(latents):
                batch_size, num_channels_latents, height, width = latents.shape
                latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
                latents = latents.permute(0, 1, 3, 5, 2, 4)
                latents = latents.reshape(batch_size, num_channels_latents * 4, height // 2, width // 2)
                return latents


            patchtokens = _patchify_latents(latents)
            bn_mean = model_dict['model'].bn.running_mean.view(1, -1, 1, 1).to(
                patchtokens.device, patchtokens.dtype
            )
            bn_std = torch.sqrt(
                model_dict['model'].bn.running_var.view(1, -1, 1, 1)
                + model_dict['model'].config.batch_norm_eps
            ).to(patchtokens.device, patchtokens.dtype)
            patchtokens = (patchtokens - bn_mean) / bn_std
            patchtokens = patchtokens.to(dtype=torch.float32).flatten(2).transpose(1, 2).contiguous()
            if 'dinov3_flux2_concat' in self.image_cond_model_name:
                # padding zeros to match register token in dinov3
                # 1 class + 4 register tokens
                zero_register_tokens = torch.zeros(patchtokens.shape[0], 4 + 1, patchtokens.shape[2], dtype=patchtokens.dtype, device=patchtokens.device)
                patchtokens = torch.cat([zero_register_tokens, patchtokens], dim=1)
            return patchtokens

        raise ValueError(f"Unsupported image conditioning model type: {model_type}")

    @torch.no_grad()
    def encode_image(self, image: Union[torch.Tensor, List[Image.Image]]) -> Union[torch.Tensor, TensorDict]:
        """
        Encode the image.
        """
        if self.image_cond_model is None:
            self._init_image_cond_model()

        model_type = self.image_cond_model['type']
        
        if model_type == 'mixture':
            feat1 = self._encode_single_image(image, self.image_cond_model['models'][0])
            feat2 = self._encode_single_image(image, self.image_cond_model['models'][1])
            return TensorDict({'feature1': feat1, 'feature2': feat2}, batch_size=feat1.shape[0])
        else:
            feat1 = self._encode_single_image(image, self.image_cond_model)
            return TensorDict({'feature1': feat1}, batch_size=feat1.shape[0])
        
    def get_cond(self, cond, **kwargs):
        """
        Get the conditioning data.
        """
        cond = self.encode_image(cond)
        kwargs['neg_cond'] = get_zeros_like(cond)
        cond = super().get_cond(cond, **kwargs)
        return cond
    
    def get_inference_cond(self, cond, **kwargs):
        """
        Get the conditioning data for inference.
        FIXME: use attention mask for neg cond
        """
        cond = self.encode_image(cond)
        kwargs['cond'] = cond
        kwargs['neg_cond'] = get_zeros_like(cond)
        return kwargs

    def vis_cond(self, cond, **kwargs):
        """
        Visualize the conditioning data.
        """
        return {'image': {'value': cond, 'type': 'image'}}

class LatentImageConditionedMixin:
    def get_cond(self, cond, **kwargs):
        """
        Get the conditioning data.
        """
        kwargs['neg_cond'] = get_zeros_like(cond)
        cond = super().get_cond(cond, **kwargs)
        return cond
    
    def get_inference_cond(self, cond, **kwargs):
        """
        Get the conditioning data for inference.
        """
        kwargs['neg_cond'] = get_zeros_like(cond)
        cond = super().get_inference_cond(cond, **kwargs)
        return cond
