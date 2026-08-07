import json
import os
import posixpath
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

DEFAULT_HF_REPO_ID = "VAST-AI/TripoSplat"
DEFAULT_HF_DENOISER_FILE = "diffusion_models/triposplat_fp16.safetensors"
DEFAULT_HF_VAE_ENCODER_FILE = "vae/triposplat_vae_encoder_fp16.safetensors"
DEFAULT_HF_VAE_DECODER_FILE = "vae/triposplat_vae_decoder_fp16.safetensors"
DEFAULT_HF_CONFIG_FILE = "configs/dit/latent8k-cond1k-latentseq_flow_img_s3dit-L.yaml"
DEFAULT_HF_DENOISER_PATH = f"hf://{DEFAULT_HF_REPO_ID}/{DEFAULT_HF_DENOISER_FILE}"
DEFAULT_HF_VAE_ENCODER_PATH = f"hf://{DEFAULT_HF_REPO_ID}/{DEFAULT_HF_VAE_ENCODER_FILE}"
DEFAULT_DINOV3_PATH = "facebook/dinov3-vith16plus-pretrain-lvd1689m"
DEFAULT_FLUX2_VAE_PATH = "black-forest-labs/FLUX.2-dev"

def env_flag_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def hf_local_files_only() -> bool:
    return any(
        env_flag_enabled(name)
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "DIFFUSERS_OFFLINE")
    )


def _strip_hf_scheme(path: str) -> str:
    return path[len("hf://"):] if path.startswith("hf://") else path


def split_hf_path(path: str, filename: Optional[str] = None) -> Tuple[str, str]:
    path = _strip_hf_scheme(path).strip("/")
    parts = path.split("/")
    if len(parts) < 2:
        raise ValueError(
            f"Invalid Hugging Face reference {path!r}. Expected 'org/repo[/path]' or 'hf://org/repo[/path]'."
        )
    repo_id = "/".join(parts[:2])
    repo_path = "/".join(parts[2:])
    if filename:
        repo_path = posixpath.join(repo_path, filename) if repo_path else filename
    if not repo_path:
        raise ValueError(f"Hugging Face reference {path!r} does not include a file path.")
    return repo_id, repo_path


def is_local_path(path: str, filename: Optional[str] = None) -> bool:
    if filename is None:
        return os.path.exists(path)
    return os.path.exists(os.path.join(path, filename))


def resolve_model_file(path: str, filename: Optional[str] = None) -> str:
    """
    Resolve a local path or a Hugging Face reference to a local file path.

    Hugging Face references use either 'org/repo/path/to/file' or the explicit
    'hf://org/repo/path/to/file' form. If filename is provided, path may be a
    local directory, a repo id, or a repo subdirectory.
    """
    if filename is None:
        if os.path.exists(path):
            return path
    else:
        local_candidate = os.path.join(path, filename)
        if os.path.exists(local_candidate):
            return local_candidate

    from huggingface_hub import hf_hub_download

    repo_id, repo_path = split_hf_path(path, filename)
    return hf_hub_download(repo_id=repo_id, filename=repo_path)


def list_model_files(path: str, suffixes: Sequence[str]) -> List[Tuple[str, str]]:
    """
    List model files from a local directory or Hugging Face repo/subdirectory.

    Returns (display_name, resolvable_path) tuples. The resolvable path can be
    passed back to resolve_model_file/load_state_dict_file.
    """
    suffixes = tuple(suffixes)
    if os.path.isdir(path):
        return [
            (name, os.path.join(path, name))
            for name in sorted(os.listdir(path))
            if name.endswith(suffixes)
        ]

    from huggingface_hub import list_repo_files

    clean = _strip_hf_scheme(path).strip("/")
    parts = clean.split("/")
    if len(parts) < 2:
        raise ValueError(f"{path!r} is neither a local directory nor a Hugging Face repo reference.")
    repo_id = "/".join(parts[:2])
    prefix = "/".join(parts[2:])
    files = list_repo_files(repo_id=repo_id)
    entries = []
    for file_name in sorted(files):
        if prefix and not file_name.startswith(prefix.rstrip("/") + "/"):
            continue
        if file_name.endswith(suffixes):
            entries.append((os.path.basename(file_name), f"hf://{repo_id}/{file_name}"))
    return entries


def load_state_dict_file(
    path: str,
    *,
    map_location="cpu",
    weights_only: bool = True,
    device: Optional[str] = None,
):
    local_path = resolve_model_file(path)
    if local_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(local_path, device=device or str(map_location))

    return torch.load(local_path, map_location=map_location, weights_only=weights_only)


def unwrap_state_dict(state_dict):
    if isinstance(state_dict, dict):
        for key in ("state_dict", "module"):
            if key in state_dict and isinstance(state_dict[key], dict):
                return state_dict[key]
    return state_dict


def strip_state_dict_prefixes(state_dict: Dict[str, torch.Tensor], prefixes: Iterable[str]) -> Dict[str, torch.Tensor]:
    for prefix in prefixes:
        if any(k.startswith(prefix) for k in state_dict):
            return {
                (k[len(prefix):] if k.startswith(prefix) else k): v
                for k, v in state_dict.items()
                if k.startswith(prefix)
            }
    return state_dict


def select_model_state_dict(
    state_dict: Dict[str, torch.Tensor],
    model_name: Optional[str] = None,
    valid_keys: Optional[Iterable[str]] = None,
) -> Dict[str, torch.Tensor]:
    state_dict = unwrap_state_dict(state_dict)
    if model_name:
        aliases = {
            "decoder": ("octree",),
            "decoder_gs": ("gs",),
        }
        names = (model_name, *aliases.get(model_name, ()))
        prefixes = tuple(prefix for name in names for prefix in (f"models.{name}.", f"{name}."))
        state_dict = strip_state_dict_prefixes(state_dict, prefixes=prefixes)
    if valid_keys is not None:
        valid_keys = set(valid_keys)
        if any(k not in valid_keys for k in state_dict):
            filtered = {k: v for k, v in state_dict.items() if k in valid_keys}
            if filtered:
                return filtered
    return state_dict


def load_model_state_dict(
    model,
    path: str,
    *,
    model_name: Optional[str] = None,
    strict: bool = True,
    map_location="cpu",
    weights_only: bool = True,
    device: Optional[str] = None,
):
    state_dict = load_state_dict_file(
        path,
        map_location=map_location,
        weights_only=weights_only,
        device=device,
    )
    state_dict = select_model_state_dict(
        state_dict,
        model_name=model_name,
        valid_keys=model.state_dict().keys(),
    )
    return model.load_state_dict(state_dict, strict=strict)


def load_config(path: str, filename: Optional[str] = None):
    config_file = resolve_model_file(path, filename)
    with open(config_file, "r") as f:
        if config_file.endswith((".yaml", ".yml")):
            import re
            import yaml
            yaml.FullLoader.add_implicit_resolver(
                u'tag:yaml.org,2002:float',
                re.compile(r'''^(?:[-+]?(?:[0-9][0-9_]*)?\.[0-9_]*(?:[eE][-+]?[0-9]+)?
                            |[-+]?[0-9][0-9_]*(?:[eE][-+]?[0-9]+)
                            |[-+]?\.[0-9_]+(?:[eE][-+]?[0-9]+)?)$''', re.X),
                list('-+0123456789.')
            )
            return yaml.load(f, Loader=yaml.FullLoader)
        return json.load(f)


def default_vae_config_path() -> str:
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "configs", "vae", "stage3-gs_fixlen_vae-XL-fps.yaml")
    )


def default_hf_config_path() -> str:
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", DEFAULT_HF_CONFIG_FILE)
    )
