import os
import copy
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import json
import importlib
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import utils3d
from tqdm import tqdm
from easydict import EasyDict as edict
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from torchvision import transforms
from PIL import Image
import trimesh
from deg.utils.hf_utils import DEFAULT_DINOV3_PATH, hf_local_files_only

torch.set_grad_enabled(False)


def get_data(frames, sha256, resolution=(518, 518)):
    with ThreadPoolExecutor(max_workers=16) as executor:
        def worker(view):
            image_path = os.path.join(opt.output_dir, 'renders', sha256, view['file_path'])
            try:
                image = Image.open(image_path)
            except:
                print(f"Error loading image {image_path}")
                return None
            image = image.resize(resolution, Image.Resampling.LANCZOS)
            image = np.array(image).astype(np.float32) / 255
            image = image[:, :, :3] * image[:, :, 3:]
            image = torch.from_numpy(image).permute(2, 0, 1).float()

            c2w = torch.tensor(view['transform_matrix'])
            c2w[:3, 1:3] *= -1
            extrinsics = torch.inverse(c2w)
            fov = view['camera_angle_x']
            intrinsics = utils3d.torch.intrinsics_from_fov_xy(torch.tensor(fov), torch.tensor(fov))

            return {
                'image': image,
                'extrinsics': extrinsics,
                'intrinsics': intrinsics
            }
        
        datas = executor.map(worker, frames)
        for data in datas:
            if data is not None:
                yield data
                
def mesh2pcd(mesh, num_pcds):
    points, _ = trimesh.sample.sample_surface(mesh, count=num_pcds)
    return points
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the metadata')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--model', type=str, default='dinov2_vitl14_reg',
                        help='Feature extraction model')
    parser.add_argument('--instances', type=str, default=None,
                        help='Instances to process')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--num_pcds', type=int, default=16384)
    parser.add_argument('--points_chunk', type=int, default=10_000,
                    help='Number of points per grid_sample chunk')

    opt = parser.parse_args()
    opt = edict(vars(opt))

    feature_name = opt.model
    output_folder = "verbose/pcd_features" if opt.verbose else "pcd_features"
    os.makedirs(os.path.join(opt.output_dir, output_folder, feature_name), exist_ok=True)

    # load model
    if opt.model == "flux1_dev_vae":
        from diffusers.models import AutoencoderKL
        encoder_model = AutoencoderKL.from_pretrained(
            "unsloth/FLUX.1-dev", subfolder="vae", torch_dtype=torch.bfloat16
        ).to("cuda")
        encoder_model.requires_grad_(False)
        encoder_model.eval()
        encoder_type = 'flux_vae'
    elif opt.model == "flux2_dev_vae":
        from diffusers.models import AutoencoderKLFlux2
        # Read from environment variable, fallback to official repo if not set
        flux2_path = os.environ.get("FLUX2_VAE_PATH", "black-forest-labs/FLUX.2-dev")
        encoder_model = AutoencoderKLFlux2.from_pretrained(
            flux2_path, subfolder="vae", torch_dtype=torch.bfloat16
        ).to("cuda")
        encoder_model.requires_grad_(False)
        encoder_model.eval()
        encoder_type = 'flux2_vae'
    elif opt.model == "dinov3_vith16plus":
        from transformers import AutoModel
        pretrained_model_name = os.environ.get(
            "DINO_V3_PATH",
            DEFAULT_DINOV3_PATH,
        )
        dinov3_model = AutoModel.from_pretrained(
            pretrained_model_name,
            local_files_only=hf_local_files_only(),
        ).eval().cuda()
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        encoder_type = 'dinov3'
    else:
        try:
            hub_dir = torch.hub.get_dir()
            repo_dir = os.path.join(hub_dir, "facebookresearch_dinov2_main")
            dinov2_model = torch.hub.load(
                repo_or_dir=repo_dir,
                model=opt.model,
                source="local",
                pretrained=True
            )
        except:
            dinov2_model = torch.hub.load('facebookresearch/dinov2', opt.model)
        dinov2_model.eval().cuda()
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        n_patch = 518 // 14
        encoder_type = 'dinov2'

    # get file list
    if os.path.exists(os.path.join(opt.output_dir, 'metadata.csv')):
        metadata = pd.read_csv(os.path.join(opt.output_dir, 'metadata.csv'))
    else:
        raise ValueError('metadata.csv not found')
    if opt.instances is not None:
        with open(opt.instances, 'r') as f:
            instances = f.read().splitlines()
        metadata = metadata[metadata['sha256'].isin(instances)]
    else:
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
        if f'pcd_feature_{feature_name}' in metadata.columns and not opt.verbose:
            metadata = metadata[metadata[f'pcd_feature_{feature_name}'] == False]
        metadata = metadata[metadata['rendered'] == True]

    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]
    records = []

    # filter out objects that are already processed
    if opt.verbose:
        np.random.seed(42)
        # metadata = metadata[metadata['sha256']=="7d05890d-e2b8-4bf7-adb8-169357d26e4d"]
        metadata = metadata.sample(n=4)
        sha256s = list(metadata['sha256'].values)
    else:
        sha256s = list(metadata['sha256'].values)
        for sha256 in copy.copy(sha256s):
            if os.path.exists(os.path.join(opt.output_dir, output_folder, feature_name, f'{sha256}.npz')):
                records.append({'sha256': sha256, f'pcd_feature_{feature_name}' : True})
                sha256s.remove(sha256)

    # extract features
    load_queue = Queue(maxsize=4)
    try:
        with ThreadPoolExecutor(max_workers=8) as loader_executor, \
            ThreadPoolExecutor(max_workers=8) as saver_executor:
            def loader(sha256):
                try:
                    with open(os.path.join(opt.output_dir, 'renders', sha256, 'transforms.json'), 'r') as f:
                        metadata = json.load(f)
                    frames = metadata['frames']
                    data = []
                    if encoder_type == 'dinov2':
                        resolution = (518, 518)
                        for datum in get_data(frames, sha256, resolution=resolution):
                            datum['image'] = transform(datum['image'])
                            data.append(datum)
                    elif encoder_type == 'dinov3':
                        resolution = (1024, 1024)
                        for datum in get_data(frames, sha256, resolution=resolution):
                            datum['image'] = transform(datum['image'])
                            data.append(datum)
                    elif encoder_type == 'flux_vae' or encoder_type == 'flux2_vae':
                        resolution = (512, 512)
                        for datum in get_data(frames, sha256, resolution=resolution):
                            datum['image'] = datum['image'].to(torch.bfloat16) * 2 - 1
                            data.append(datum)
                    try:
                        mesh = trimesh.load(os.path.join(opt.output_dir, 'renders', sha256, 'mesh.stl'))
                    except:
                        mesh = trimesh.load(os.path.join(opt.output_dir, 'renders', sha256, 'mesh.ply'))
                        
                    pcds = mesh2pcd(mesh, opt.num_pcds)
                    load_queue.put((sha256, data, pcds))
                except Exception as e:
                    load_queue.put((None, None, None))
                    print(f"Error loading data for {sha256}: {e}")

            loader_executor.map(loader, sha256s)
            
            def saver(sha256, pack):
                save_path = os.path.join(opt.output_dir, output_folder, feature_name, f'{sha256}.npz')
                np.savez_compressed(save_path, **pack)
                records.append({'sha256': sha256, f'pcd_feature_{feature_name}' : True})
                
            for _ in tqdm(range(len(sha256s)), desc="Extracting pcd features"):
                sha256, data, positions = load_queue.get()
                if sha256 is None:
                    print(f"Error loading data for {sha256}")
                    continue
                positions = torch.from_numpy(positions).float().cuda()
                n_views = len(data)
                N = positions.shape[0]
                pack = {
                    "pcds": positions.cpu().numpy().astype(np.float16),
                }
                patchtokens_lst = []
                uv_lst = []
                for i in range(0, n_views, opt.batch_size):
                    batch_data = data[i:i+opt.batch_size]
                    bs = len(batch_data)
                    batch_images = torch.stack([d['image'] for d in batch_data]).cuda()
                    batch_extrinsics = torch.stack([d['extrinsics'] for d in batch_data]).cuda()
                    batch_intrinsics = torch.stack([d['intrinsics'] for d in batch_data]).cuda()
                    uv = utils3d.torch.project_cv(positions, batch_extrinsics, batch_intrinsics)[0] * 2 - 1
                    if encoder_type == 'flux_vae':
                        features = encoder_model.encode(batch_images)
                        patchtokens = features.latent_dist.sample()  # [B, C, H, W]
                        patchtokens = (patchtokens - encoder_model.config.shift_factor) * encoder_model.config.scaling_factor 
                        patchtokens = patchtokens.to(torch.float32)
                    elif encoder_type == 'flux2_vae':
                        features = encoder_model.encode(batch_images)
                        patchtokens = features.latent_dist.sample()  # [B, C, H, W]

                        def _patchify_latents(latents):
                            batch_size, num_channels_latents, height, width = latents.shape
                            latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
                            latents = latents.permute(0, 1, 3, 5, 2, 4)
                            latents = latents.reshape(batch_size, num_channels_latents * 4, height // 2, width // 2)
                            return latents

                        def _unpatchify_latents(latents):
                            batch_size, num_channels_latents, height, width = latents.shape
                            latents = latents.reshape(batch_size, num_channels_latents // (2 * 2), 2, 2, height, width)
                            latents = latents.permute(0, 1, 4, 2, 5, 3)
                            latents = latents.reshape(batch_size, num_channels_latents // (2 * 2), height * 2, width * 2)
                            return latents
                        
                        patchtokens = _patchify_latents(patchtokens)
                        bn_mean = encoder_model.bn.running_mean.view(1, -1, 1, 1).to(
                            patchtokens.device, patchtokens.dtype
                        )
                        bn_std = torch.sqrt(
                            encoder_model.bn.running_var.view(1, -1, 1, 1) + encoder_model.config.batch_norm_eps
                        ).to(patchtokens.device, patchtokens.dtype)
                        patchtokens = (patchtokens - bn_mean) / bn_std
                        patchtokens = patchtokens.to(torch.float32)
                        patchtokens = _unpatchify_latents(patchtokens)
                    elif encoder_type == 'dinov3':
                        outputs = dinov3_model(batch_images)
                        last_hidden = outputs.last_hidden_state
                        tokens_flat = last_hidden[:, 1+4:, :]  # drop class + 4 register tokens
                        Hp, Wp = batch_images.shape[-2] // 16, batch_images.shape[-1] // 16
                        patchtokens = tokens_flat.permute(0, 2, 1).reshape(bs, last_hidden.shape[-1], Hp, Wp)
                    else:
                        features = dinov2_model(batch_images, is_training=True)
                        patchtokens = features['x_prenorm'][:, dinov2_model.num_register_tokens + 1:].permute(0, 2, 1).reshape(bs, 1024, n_patch, n_patch)
                    patchtokens_lst.append(patchtokens)
                    uv_lst.append(uv)
                # patchtokens = torch.cat(patchtokens_lst, dim=0)
                # uv = torch.cat(uv_lst, dim=0)
                # pack['patchtokens'] = F.grid_sample(
                #     patchtokens,
                #     uv.unsqueeze(1),
                #     mode='bilinear',
                #     align_corners=False,
                # ).squeeze(2).permute(0, 2, 1).mean(dim=0).cpu().numpy().astype(np.float16)
                patchtokens = torch.cat(patchtokens_lst, dim=0)  # [V, 1024, H, W]
                uv = torch.cat(uv_lst, dim=0)                    # [V, N, 2]

                V, N = uv.shape[0], uv.shape[1]
                if encoder_type == 'flux_vae':
                    accum = torch.zeros(N, 16, dtype=torch.float32, device=patchtokens.device)  # accumulate mean over views
                elif encoder_type == 'flux2_vae':
                    accum = torch.zeros(N, 32, dtype=torch.float32, device=patchtokens.device)
                elif encoder_type == 'dinov2':
                    accum = torch.zeros(N, 1024, dtype=torch.float32, device=patchtokens.device)  # accumulate mean over views
                elif encoder_type == 'dinov3':
                    accum = torch.zeros(N, 1280, dtype=torch.float32, device=patchtokens.device)  # accumulate mean over views

                for start in range(0, N, opt.points_chunk):
                    end = min(start + opt.points_chunk, N)
                    # grid: [V, 1, n_chunk, 2] in [-1, 1]
                    grid = uv[:, start:end, :].unsqueeze(1)  # contiguous not strictly required here

                    # Sample: [V, 1024, 1, n_chunk] -> [V, n_chunk, 1024]
                    sampled = F.grid_sample(
                        patchtokens, grid, mode='bilinear', align_corners=False
                    ).squeeze(2).permute(0, 2, 1)

                    # Mean over views -> [n_chunk, 1024], then accumulate at positions [start:end]
                    accum[start:end] = sampled.mean(dim=0)

                # Move to CPU + fp16 for saving (as before)
                pack['patchtokens'] = accum.to('cpu', torch.float16).numpy()
                # save features
                saver_executor.submit(saver, sha256, pack)
                
            saver_executor.shutdown(wait=True)
    except Exception as e:
        print(f"Error happened during processing: {e}.")
        
    if not opt.verbose:
        records = pd.DataFrame.from_records(records)
        records.to_csv(os.path.join(opt.output_dir, f'pcd_feature_{feature_name}_{opt.rank}.csv'), index=False)
