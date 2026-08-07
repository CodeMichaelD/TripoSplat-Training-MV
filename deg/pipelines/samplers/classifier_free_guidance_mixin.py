from typing import *
from tensordict import TensorDict

class ClassifierFreeGuidanceSamplerMixin:
    """
    A mixin class for samplers that apply classifier-free guidance.
    """
    default_cfg_strength = 3.0
    # def _inference_model(self, model, x_t, t, cond, neg_cond, cfg_strength, **kwargs):
    #     if cfg_strength == 0:
    #         return super()._inference_model(model, x_t, t, cond, **kwargs)
    #     else:
    #         pred = super()._inference_model(model, x_t, t, cond, **kwargs)
    #         neg_pred = super()._inference_model(model, x_t, t, neg_cond, **kwargs)
    #         return (1 + cfg_strength) * pred - cfg_strength * neg_pred

    def _get_model_prediction(self, model, x_t, t, cond, neg_cond, cfg_strength, **kwargs):
        if cfg_strength is None:
            cfg_strength = self.default_cfg_strength
        pred_v = self._inference_model(model, x_t, t, cond, **kwargs)
        if isinstance(cfg_strength, dict):
            assert isinstance(pred_v, TensorDict)
            neg_pred_v = self._inference_model(model, x_t, t, neg_cond, **kwargs)
            for key in pred_v.keys():
                if key in cfg_strength:
                    pred_v[key] = (1 + cfg_strength[key]) * pred_v[key] - cfg_strength[key] * neg_pred_v[key]
        elif cfg_strength > 0:
            neg_pred_v = self._inference_model(model, x_t, t, neg_cond, **kwargs)
            pred_v = (1 + cfg_strength) * pred_v - cfg_strength * neg_pred_v
        pred_x_0, pred_eps = self._v_to_xstart_eps(x_t=x_t, t=t, v=pred_v)
        return pred_x_0, pred_eps, pred_v