import torch
import numpy as np
from ....utils.general_utils import dict_foreach
from ....pipelines import samplers
from tensordict import TensorDict
from typing import *
class ClassifierFreeGuidanceMixin:
    def __init__(self, *args, eval_cfg_strength: Union[List[float], float] = 3.0, eval_step_num: Union[List[int], int] = 20, p_uncond: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.p_uncond = p_uncond
        if isinstance(eval_cfg_strength, float):
            self.eval_cfg_strength = [eval_cfg_strength]
        elif isinstance(eval_cfg_strength, list):
            self.eval_cfg_strength = eval_cfg_strength
        if isinstance(eval_step_num, int):
            self.eval_step_num = [eval_step_num]
        elif isinstance(eval_step_num, list):
            self.eval_step_num = eval_step_num
        

    def get_cond(self, cond, neg_cond=None, **kwargs):
        """
        Get the conditioning data.
        """
        assert neg_cond is not None, "neg_cond must be provided for classifier-free guidance" 

        if self.p_uncond > 0:
            # randomly drop the class label
            def get_batch_size(cond):
                if isinstance(cond, torch.Tensor):
                    return cond.shape[0]
                if isinstance(cond, TensorDict):
                    return cond.shape[0]
                elif isinstance(cond, list):
                    return len(cond)
                else:
                    raise ValueError(f"Unsupported type of cond: {type(cond)}")
                
            ref_cond = cond if not isinstance(cond, dict) else cond[list(cond.keys())[0]]
            B = get_batch_size(ref_cond)
            
            def select(cond, neg_cond, mask):
                if isinstance(cond, torch.Tensor):
                    mask = torch.tensor(mask, device=cond.device).reshape(-1, *[1] * (cond.ndim - 1))
                    return torch.where(mask, neg_cond, cond)
                elif isinstance(cond, TensorDict):
                    mask = torch.tensor(mask, device=cond.device)
                    out = cond.clone()
                    out[mask] = neg_cond[mask]
                    return out
                elif isinstance(cond, list):
                    return [nc if m else c for c, nc, m in zip(cond, neg_cond, mask)]
                else:
                    raise ValueError(f"Unsupported type of cond: {type(cond)}")
            
            mask = list(np.random.rand(B) < self.p_uncond)
            if not isinstance(cond, dict):
                cond = select(cond, neg_cond, mask)
            else:
                cond = dict_foreach([cond, neg_cond], lambda x: select(x[0], x[1], mask))
    
        return cond.contiguous()

    def get_inference_cond(self, cond, neg_cond=None, **kwargs):
        """
        Get the conditioning data for inference.
        """
        assert neg_cond is not None, "neg_cond must be provided for classifier-free guidance"
        return {'cond': cond, 'neg_cond': neg_cond, **kwargs}
    
    def get_sampler(self, **kwargs) -> samplers.FlowEulerCfgSampler:
        """
        Get the sampler for the diffusion process.
        """
        sampler = samplers.FlowEulerCfgSampler(self.sigma_min)
        if len(self.eval_cfg_strength) > 0:
            sampler.default_cfg_strength = self.eval_cfg_strength[0]
        return sampler
    
    def get_cfg_prediction(self, model, x_t, t_1000, cond, neg_cond=None, cfg_strength=0.0, **kwargs):
        """
        Get the prediction for the classifier-free guidance.
        """
        if cfg_strength == 0.0:
            # No need to do classifier-free guidance
            return model(x_t, t_1000, cond, **kwargs)
        else:
            # Get the prediction from the model
            pred = model(x_t, t_1000, cond, **kwargs)
            
            # Get the prediction from the negative conditioning
            neg_pred = model(x_t, t_1000, neg_cond, **kwargs)
            
            # Combine the predictions
            return pred + cfg_strength * (pred - neg_pred)
        