
import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import Dict, Any, Optional, List
import numpy as np
from tensordict import TensorDict

from ....utils.general_utils import dict_reduce, dict_flatten, dict_foreach, get_noise_like
from ....utils.loss_utils import psnr, ssim, lpips
from ....utils.data_utils import recursive_to_device
from transformers import CLIPProcessor, CLIPModel

class LatentSeqValidationMixin:
    """
    Validation mixin for latent sequence flow matching models.
    Evaluates alignment between generated results and ground truth using PSNR, LPIPS, SSIM, and CLIP Score.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clip_model = None
        self.clip_processor = None
        self.clip_model_name = "openai/clip-vit-large-patch14"

    def _load_clip(self):
        if self.clip_model is None:
            print(f"Loading CLIP model: {self.clip_model_name} for validation...")
            try:
                self.clip_model = CLIPModel.from_pretrained(self.clip_model_name).to(self.device).eval()
                self.clip_processor = CLIPProcessor.from_pretrained(self.clip_model_name)
            except Exception as e:
                print(f"Failed to load CLIP model: {e}")
                self.clip_model = None
        
        # Ensure model is on the correct device
        if self.clip_model is not None and self.clip_model.device != self.device:
             self.clip_model = self.clip_model.to(self.device)

    def _offload_clip(self):
        """
        Offload CLIP model to CPU to save GPU memory.
        """
        if self.clip_model is not None:
             self.clip_model = self.clip_model.cpu()
             torch.cuda.empty_cache()

    def _compute_clip_score(self, pred_images: torch.Tensor, gt_images: torch.Tensor) -> torch.Tensor:
        """
        Compute CLIP score (cosine similarity) between predicted and GT images.
        Images should be [B, C, H, W] in range [0, 1].
        Returns a scalar tensor (mean score).
        """
        # Model should be loaded by caller (run_train_val_step) for efficiency
        # But we check just in case
        if self.clip_model is None:
             self._load_clip()
             
        if self.clip_model is None:
            return torch.tensor(0.0, device=self.device)
        
        # Convert to 0-255 uint8 numpy for processor
        # Detach and move to CPU
        pred_np = (pred_images.detach().cpu().permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
        gt_np = (gt_images.detach().cpu().permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
        
        # Process inputs
        try:
            inputs_pred = self.clip_processor(images=list(pred_np), return_tensors="pt", padding=True).to(self.device)
            inputs_gt = self.clip_processor(images=list(gt_np), return_tensors="pt", padding=True).to(self.device)
            
            with torch.no_grad():
                feat_pred = self.clip_model.get_image_features(**inputs_pred)
                feat_gt = self.clip_model.get_image_features(**inputs_gt)
                
                # Normalize features
                feat_pred = feat_pred / (feat_pred.norm(dim=-1, keepdim=True) + 1e-6)
                feat_gt = feat_gt / (feat_gt.norm(dim=-1, keepdim=True) + 1e-6)
                
                # Cosine similarity
                similarity = (feat_pred * feat_gt).sum(dim=-1)
                
            return similarity.mean()
        except Exception as e:
            print(f"Error computing CLIP score: {e}")
            return torch.tensor(0.0, device=self.device)

    @torch.no_grad()
    def run_train_val_step(self, init=False):
        """
        Run a training/validation step.
        Overrides the base method to perform specialized evaluation for latent sequence flow matching.
        """
        # Load CLIP model to GPU at the start of validation
        self._load_clip()
        
        losses = []
        
        # Use existing dataloader for validation
        dataloader = self.dataloader_train_val if hasattr(self, 'dataloader_train_val') and self.dataloader_train_val is not None else self.dataloader
        
        try:
            # Iterate over validation dataset
            for data in tqdm(dataloader, desc="Validating", disable=not self.is_master):
                data = recursive_to_device(data, self.device)
                val_loss_terms = {}
                
                # 1. Extract Ground Truth
                # In FlowMatchingTrainer, x_0 is typically in data['x_0']
                if 'x_0' not in data:
                    continue
                    
                x_0_gt = data.pop('x_0') # TensorDict containing 'latent' and 'camera'
                
                # 2. Predict (Sample from Flow Matching)
                # Generate noise
                noise = get_noise_like(x_0_gt)
                    
                # Get inference conditions
                cond = self.get_inference_cond(**data)
                
                # Sample
                sampler = self.get_sampler()
                x_0_pred = sampler.sample(
                    self.models['denoiser'],
                    noise=noise,
                    **cond,
                    steps=50, # Default steps, could be configurable
                    verbose=False
                )
                if hasattr(x_0_pred, 'samples'):
                    x_0_pred = x_0_pred.samples
                    
                # 3. Visualize (Render) to get images
                # We use the dataset's visualize_sample method which returns 'camera_vis'
                # visualize_sample expects a dict with 'x_0' key
                
                # For Prediction
                vis_sample_pred = dict(x_0=x_0_pred, **data)
                # Call dataset.visualize_sample
                # Note: We assume self.dataset (or self.dataset_train_val) has this method
                dataset = self.dataset_train_val if hasattr(self, 'dataset_train_val') and self.dataset_train_val is not None else self.dataset
                
                if hasattr(dataset, 'visualize_sample'):
                    # For Prediction
                    vis_ret_pred = dataset.visualize_sample(dict(x_0=x_0_pred, **data), camera_pred=True)
                    # For Ground Truth
                    vis_ret_gt = dataset.visualize_sample(dict(x_0=x_0_gt, **data), camera_pred=True)
                    
                    # 4. Compare images
                    if 'camera_vis' in vis_ret_pred and 'camera_vis' in vis_ret_gt:
                        pred_imgs = vis_ret_pred['camera_vis'] # [B, C, H, W]
                        gt_imgs = vis_ret_gt['camera_vis']     # [B, C, H, W]
                        
                        # Ensure shapes match
                        if pred_imgs.shape == gt_imgs.shape:
                            # PSNR
                            val_psnr = psnr(pred_imgs, gt_imgs)
                            val_loss_terms['train_val/gs_gt_psnr'] = val_psnr
                            
                            # SSIM
                            val_ssim = ssim(pred_imgs, gt_imgs)
                            val_loss_terms['train_val/gs_gt_ssim'] = val_ssim
                            
                            # LPIPS
                            val_lpips = lpips(pred_imgs, gt_imgs, value_range=(0, 1))
                            val_loss_terms['train_val/gs_gt_lpips'] = val_lpips
                            
                            # CLIP Score
                            val_clip = self._compute_clip_score(pred_imgs, gt_imgs)
                            val_loss_terms['train_val/gs_gt_clip_score'] = val_clip
                    
                    # 5. Compare predicted camera view with raw condition image
                    if 'camera_vis' in vis_ret_pred and 'cond_raw' in data:
                        pred_imgs = vis_ret_pred['camera_vis']
                        cond_raw = data['cond_raw'] # [B, 3, H, W]
                        
                        # Ensure cond_raw is on the same device and type
                        cond_raw = cond_raw.to(pred_imgs.device, dtype=pred_imgs.dtype)
                        
                        # Ensure resolutions match. Resize cond_raw if needed.
                        if pred_imgs.shape[-2:] != cond_raw.shape[-2:]:
                             cond_raw = F.interpolate(cond_raw, size=pred_imgs.shape[-2:], mode='bilinear', align_corners=False)

                        # PSNR
                        val_psnr = psnr(pred_imgs, cond_raw)
                        val_loss_terms['train_val/cond_raw_psnr'] = val_psnr
                        
                        # SSIM
                        val_ssim = ssim(pred_imgs, cond_raw)
                        val_loss_terms['train_val/cond_raw_ssim'] = val_ssim
                        
                        # LPIPS
                        val_lpips = lpips(pred_imgs, cond_raw, value_range=(0, 1))
                        val_loss_terms['train_val/cond_raw_lpips'] = val_lpips
                        
                        # CLIP Score
                        val_clip = self._compute_clip_score(pred_imgs, cond_raw)
                        val_loss_terms['train_val/cond_raw_clip_score'] = val_clip
                
                if val_loss_terms:
                    losses.append(val_loss_terms)

            # Aggregate losses
            if losses:
                losses = dict_reduce(losses, lambda x: torch.stack(x).mean())
                losses = dict_flatten(losses, sep='/')
                losses = dict_foreach(losses, lambda x: {"value": x, "reduce": "mean"})
            else:
                losses = {}
            
            if self.debug and self.is_master:
                print("Validation losses:", losses)
                
            return losses
        
        finally:
            # Offload CLIP model to CPU after validation
            self._offload_clip()
