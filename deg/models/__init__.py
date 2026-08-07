import importlib

__attributes = {
    'FixedlenEncoder': 'gs_seqence_vae.gs_fixlen_vae',
    'ElasticFixedlenEncoder': 'gs_seqence_vae.gs_fixlen_vae',
    'FixedlenDecoder': 'gs_seqence_vae.gs_fixlen_vae',
    'GaussianFixedlenDecoder': 'gs_seqence_vae.gs_fixlen_vae',
    'ElasticGaussianFixedlenDecoder': 'gs_seqence_vae.gs_fixlen_vae',
    'OctreeProbabilityFixedlenDecoder': 'gs_seqence_vae.gs_fixlen_vae',

    'LatentSeqMMFlowModel': 'latent_seq_flow',
}

__submodules = []

__all__ = list(__attributes.keys()) + __submodules

def __getattr__(name):
    if name not in globals():
        if name in __attributes:
            module_name = __attributes[name]
            module = importlib.import_module(f".{module_name}", __name__)
            globals()[name] = getattr(module, name)
        elif name in __submodules:
            module = importlib.import_module(f".{name}", __name__)
            globals()[name] = module
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")
    return globals()[name]


def from_pretrained(path: str, **kwargs):
    """
    Load a model from a pretrained checkpoint.

    Args:
        path: The path to the checkpoint. Can be either local path or a Hugging Face model name.
              NOTE: config file and model file should take the name f'{path}.json' and f'{path}.safetensors' respectively.
        **kwargs: Additional arguments for the model constructor.
    """
    import json
    from deg.utils.hf_utils import load_model_state_dict, resolve_model_file

    strict = kwargs.pop("strict", True)
    config_file = resolve_model_file(f"{path}.json")
    model_file = resolve_model_file(f"{path}.safetensors")
    with open(config_file, 'r') as f:
        config = json.load(f)
    model = __getattr__(config['name'])(**config['args'], **kwargs)
    load_model_state_dict(model, model_file, strict=strict)

    return model
