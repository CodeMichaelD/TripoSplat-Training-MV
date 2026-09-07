from typing import *
import copy
import torch
from torch.utils.data import DataLoader
import numpy as np
from deg.utils.general_utils import edict
from torchvision import utils
import os
import functools

from ..basic import BasicTrainer
from ...pipelines import samplers 
from ...utils.general_utils import dict_reduce, get_noise_like
from ...utils.loss_utils import general_mse_loss
from ...utils.data_utils import recursive_to_device
from .mixins.classifier_free_guidance import ClassifierFreeGuidanceMixin
from .mixins.image_conditioned import ImageConditionedMixin
from .mixins.latent_seq_val_mixin import LatentSeqValidationMixin
from ...utils.data_utils import cycle, BalancedResumableSampler

class FlowMatchingTrainer(BasicTrainer):
    """
    Trainer for diffusion model with flow matching objective.
    
    Args:
        models (Dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.

        t_schedule (dict): Time schedule for flow matching.
        sigma_min (float): Minimum noise level.
    """
    def __init__(
        self,
        *args,
        t_schedule: dict = {
            'name': 'logitNormal',
            'args': {
                'mean': 0.0,
                'std': 1.0,
            }
        },
        sigma_min: float = 1e-5,
        mse_lambda: Dict = {},
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.t_schedule = t_schedule
        self.sigma_min = sigma_min
        self.mse_lambda = mse_lambda

    def prepare_dataloader(self, sync_data=False, **kwargs):
        """
        Prepare dataloader.
        """
        if hasattr(self.dataset, 'loads'):
            self.data_sampler = BalancedResumableSampler(
                self.dataset,
                shuffle=True,
                batch_size=self.batch_size_per_gpu,
                sync=sync_data,
            )
            self.dataloader = DataLoader(
                self.dataset,
                batch_size=self.batch_size_per_gpu,
                num_workers=int(np.ceil(os.cpu_count() / torch.cuda.device_count())),
                pin_memory=True,
                drop_last=True,
                persistent_workers=True,
                collate_fn=functools.partial(self.dataset.collate_fn, split_size=self.batch_split),
                sampler=self.data_sampler,
            )
            self.data_iterator = cycle(self.dataloader)
        else:
            super().prepare_dataloader(sync_data=sync_data, **kwargs)
        
    def diffuse(self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Diffuse the data for a given number of diffusion steps.
        In other words, sample from q(x_t | x_0).

        Args:
            x_0: The [N x C x ...] tensor of noiseless inputs.
            t: The [N] tensor of diffusion steps [0-1].
            noise: If specified, use this noise instead of generating new noise.

        Returns:
            x_t, the noisy version of x_0 under timestep t.
        """
        if noise is None:
            noise = get_noise_like(x_0)
        assert noise.shape == x_0.shape, "noise must have same shape as x_0"

        t = t.view(-1, *[1 for _ in range(len(x_0.shape) - 1)])
        x_t = x_0 * (1 - t) + noise * (self.sigma_min + (1 - self.sigma_min) * t)

        return x_t

    def reverse_diffuse(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Get original image from noisy version under timestep t.
        """
        assert noise.shape == x_t.shape, "noise must have same shape as x_t"
        t = t.view(-1, *[1 for _ in range(len(x_t.shape) - 1)])
        x_0 = (x_t - (self.sigma_min + (1 - self.sigma_min) * t) * noise) / (1 - t)
        return x_0

    def get_v(self, x_0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the velocity of the diffusion process at time t.
        """
        return (1 - self.sigma_min) * noise - x_0

    def get_cond(self, **kwargs):
        """
        Get the conditioning data.
        """
        return kwargs
    
    def get_inference_cond(self, **kwargs):
        """
        Get the conditioning data for inference.
        """
        return kwargs

    def get_sampler(self, **kwargs) -> samplers.FlowEulerSampler:
        """
        Get the sampler for the diffusion process.
        """
        return samplers.FlowEulerSampler(self.sigma_min)
    
    def vis_cond(self, cond, slice_batch_size=None, **kwargs):
        """
        Visualize the conditioning data.
        """
        return {}

    def sample_t(self, batch_size: int) -> torch.Tensor:
        """
        Sample timesteps.
        """
        if self.t_schedule['name'] == 'uniform':
            t = torch.rand(batch_size)
        elif self.t_schedule['name'] == 'logitNormal':
            mean = self.t_schedule['args']['mean']
            std = self.t_schedule['args']['std']
            t = torch.sigmoid(torch.randn(batch_size) * std + mean)
        elif self.t_schedule['name'] == 'linear':
            t = torch.sqrt(torch.rand(batch_size))
        else:
            raise ValueError(f"Unknown t_schedule: {self.t_schedule['name']}")
        return t
    
    def encode_x_0(self, x_0):
        return x_0
    
    def decode_x_0(self, x_0, slice_batch_size=None, **kwargs):
        return x_0, {}
    
    def delete_usused_models(self):
        """
        Delete unused models to save memory.
        For example, the unused vae or decoder models for visualizatin
        """
        pass

    def training_losses(
        self,
        x_0: torch.Tensor,
        cond=None,
        **kwargs
    ) -> Tuple[Dict, Dict]:
        """
        Compute training losses for a single timestep.

        Args:
            x_0: The [N x C x ...] tensor of noiseless inputs.
            cond: The [N x ...] tensor of additional conditions.
            kwargs: Additional arguments to pass to the backbone.

        Returns:
            a dict with the key "loss" containing a tensor of shape [N].
            may also contain other keys for different terms.
        """
        x_0 = self.encode_x_0(x_0)
        noise = get_noise_like(x_0)
        t = self.sample_t(x_0.shape[0]).to(x_0.device).float()
        x_t = self.diffuse(x_0, t, noise=noise)
        
        # Encode control image if present
        ctrl_image = kwargs.pop('ctrl_image', None)
        if ctrl_image is not None and hasattr(self, 'encode_image'):
            kwargs['ctrl_tokens'] = self.encode_image(ctrl_image)['feature1']
            
        cond = self.get_cond(cond=cond, **kwargs)
        pred = self.training_models['denoiser'](x_t, t * 1000, cond, **kwargs)
        assert pred.shape == noise.shape == x_0.shape
        target = self.get_v(x_0, noise, t)
        terms = edict()
        terms["mse"], terms["loss_dict"] = general_mse_loss(pred, target, mse_lambda=self.mse_lambda)
        terms["loss"] = terms["mse"]

        # log loss with time bins
        mse_per_instance = np.array([
            general_mse_loss(pred[i], target[i], mse_lambda=self.mse_lambda)[0].item()
            for i in range(x_0.shape[0])
        ])
        time_bin = np.digitize(t.cpu().numpy(), np.linspace(0, 1, 11)) - 1
        for i in range(10):
            if (time_bin == i).sum() != 0:
                terms[f"bin_{i}"] = {"mse": mse_per_instance[time_bin == i].mean()}

        return terms, {}
    
    @torch.no_grad()
    def snapshot_dataset(self, num_samples=None):
        """
        Sample images from the dataset.
        """
        num_samples = self.snapshot_dataset_num_samples if num_samples is None else num_samples
        if self.is_master:
            print(f'\nSampling {num_samples} images from dataset...', end='')
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=num_samples,
            num_workers=0,
            shuffle=True,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
        )
        data = next(iter(dataloader))
        data = recursive_to_device(data, self.device)
        
        x_0 = self.encode_x_0(data.pop('x_0'))
        x_0, x_0_kwargs = self.decode_x_0(x_0, **data)
        
        vis_sample = dict(
            x_0=x_0,
            **x_0_kwargs,
            **data,
        )
        vis = self.visualize_sample(vis_sample)
        if isinstance(vis, dict):
            save_cfg = [(f'dataset_{k}', v) for k, v in vis.items()]
        else:
            save_cfg = [('dataset', vis)]
        for name, image in save_cfg:
            utils.save_image(
                image,
                os.path.join(self.output_dir, 'samples', f'{name}.jpg'),
                nrow=int(np.sqrt(num_samples)),
                normalize=True,
                value_range=self.dataset.value_range,
            )
        self.delete_usused_models()
        if self.is_master:
            print(f' Done.')
    
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
        sampler = self.get_sampler()
        if self.is_master and hasattr(sampler, 'default_cfg_strength'):
            print(f'\nSampling with cfg strength {sampler.default_cfg_strength}...', end='')
        sample_gt = []
        sample = []
        cond_vis = []
        cond_raw_vis = []
        for i in range(0, num_samples, batch_size):
            data = next(iter(dataloader))
            data = recursive_to_device(data, self.device) # SLICE BATCH...
            x_0 = data.pop('x_0')
            
            x_0 = self.encode_x_0(x_0)
            noise = get_noise_like(x_0)

            # ─── NEW: Encode control image if present ───
            ctrl_image = data.pop('ctrl_image', None)
            if ctrl_image is not None and hasattr(self, 'encode_image'):
                data['ctrl_tokens'] = self.encode_image(ctrl_image)['feature1']

            args = self.get_inference_cond(**data)
            res = sampler.sample(
                self.models['denoiser'],
                noise=noise,
                **args,
                steps=50,
                verbose=verbose,
            ).samples
            
            x_0_dec, x_0_kwargs = self.decode_x_0(x_0, **data)
            res_dec, res_kwargs = self.decode_x_0(res, **data)
            x_0_dec = dict(
                x_0=x_0_dec,
                **x_0_kwargs,
                **data,
            )
            res_dec = dict(
                x_0=res_dec,
                **res_kwargs,
                **data,
            )
            x_0_vis = self.visualize_sample(x_0_dec)
            res_vis = self.visualize_sample(res_dec)

            
            batch = min(batch_size, num_samples - i)
            # slice batch
            cond_vis.append(data['cond'][:batch])
            if 'cond_raw' in data:
                cond_raw_vis.append(data['cond_raw'][:batch])
            sample_gt.append({k: v[:batch] for k, v in x_0_vis.items()})
            sample.append({k: v[:batch] for k, v in res_vis.items()})

        sample_dict = {}
        cond_vis = torch.cat(cond_vis, dim=0)
        cond_vis = {'cond': {'value': cond_vis, 'type': 'image'}}
        sample_dict.update(cond_vis)
        if 'cond_raw' in data:
            cond_raw_vis = torch.cat(cond_raw_vis, dim=0)
            cond_raw_vis = {'cond_raw': {'value': cond_raw_vis, 'type': 'image'}}
            sample_dict.update(cond_raw_vis)
        # cat the dict sample and sample gt
        if isinstance(sample[0], dict):
            sample_gt = {f"sample_gt_{k}": {'value': torch.cat([v[k] for v in sample_gt], dim=0), 'type': 'image'} for k in sample_gt[0].keys()}
            sample = {f"sample_{k}": {'value': torch.cat([v[k] for v in sample], dim=0), 'type': 'image'} for k in sample[0].keys()}
            sample_dict.update(sample_gt)
            sample_dict.update(sample)
        else:
            sample_gt = torch.cat(sample_gt, dim=0)
            sample = torch.cat(sample, dim=0)
            sample_dict['sample_gt'] = {'value': sample_gt, 'type': 'image'},
            sample_dict['sample'] = {'value': sample, 'type': 'image'},
        self.delete_usused_models()
        
        return sample_dict

    
class FlowMatchingCFGTrainer(ClassifierFreeGuidanceMixin, FlowMatchingTrainer):
    """
    Trainer for diffusion model with flow matching objective and classifier-free guidance.
    
    Args:
        models (Dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.

        t_schedule (dict): Time schedule for flow matching.
        sigma_min (float): Minimum noise level.
        p_uncond (float): Probability of dropping conditions.
    """
    pass




class ImageConditionedFlowMatchingCFGTrainer(ImageConditionedMixin, FlowMatchingCFGTrainer):
    """
    Trainer for image-conditioned diffusion model with flow matching objective and classifier-free guidance.
    
    Args:
        models (Dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.

        t_schedule (dict): Time schedule for flow matching.
        sigma_min (float): Minimum noise level.
        p_uncond (float): Probability of dropping conditions.
        image_cond_model (str): Image conditioning model.
    """
    pass

class ImageConditionedLatentSeqFlowMatchingCFGTrainer(LatentSeqValidationMixin, ImageConditionedFlowMatchingCFGTrainer):
    pass
