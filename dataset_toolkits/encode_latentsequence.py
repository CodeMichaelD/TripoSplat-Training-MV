import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import copy
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from easydict import EasyDict as edict
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import ot # POT

import deg.models as models
import deg.datasets as datasets
from deg.utils.hf_utils import (
    DEFAULT_HF_VAE_ENCODER_PATH,
    default_vae_config_path,
    load_config,
    load_model_state_dict,
    load_state_dict_file,
    resolve_model_file,
)


torch.set_grad_enabled(False)
def ot_solve_perm(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Compute the OT permutation using POT's unified solve API.
    Maps Y to X (Y[perm] ~ X).
    """
    sol = ot.solve_sample(X, Y, metric='sqeuclidean')
    P = sol.plan  # np to_dense()
    indices = np.argmax(P, axis=1)
    return indices


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the metadata')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=4.0,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--enc_pretrained', type=str, default=DEFAULT_HF_VAE_ENCODER_PATH,
                        help='Pretrained encoder checkpoint or legacy model prefix')
    parser.add_argument('--encoder_config', type=str, default=None,
                        help='Encoder architecture config; defaults to the bundled stage-3 VAE config')
    parser.add_argument('--model_root', type=str, default='results',
                        help='Experiment dir of the model')
    parser.add_argument('--enc_model', type=str, default=None,
                        help='Encoder model tag. if specified, use this model instead of pretrained model')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='Checkpoint to load')
    parser.add_argument('--latent_length', type=int, nargs='+', default=[1024, 2048, 4096],
                        help='Latent length list')
    parser.add_argument('--instances', type=str, default=None,
                        help='Instances to process')
    parser.add_argument('--fgw_alpha',type=float, nargs='+', default=[],
                        help='Fused Gromov-Wasserstein alpha')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--rot90_axis', type=int, default=2,
                        help='Axis to rotate 90 degrees. Set 2 to rotate around z-axis (turn around axis).') 
    opt = parser.parse_args()
    opt = edict(vars(opt))

    if opt.enc_model is None:
        latent_name = os.path.splitext(os.path.basename(opt.enc_pretrained))[0]
        cfg = edict(load_config(opt.encoder_config or default_vae_config_path()))
        if opt.enc_pretrained.endswith(('.pt', '.safetensors')):
            encoder = getattr(models, cfg.models.encoder.name)(**cfg.models.encoder.args).cuda()
            load_model_state_dict(
                encoder,
                opt.enc_pretrained,
                model_name='encoder',
                strict=True,
                map_location='cpu',
            )
            encoder.eval()
            print(f'Loaded pretrained encoder from {opt.enc_pretrained}')
        else:
            encoder = models.from_pretrained(opt.enc_pretrained).eval().cuda()
    else:
        latent_name = f'{opt.enc_model}_{opt.ckpt}'
        cfg = edict(load_config(opt.model_root, 'config.json'))
        encoder = getattr(models, cfg.models.encoder.name)(**cfg.models.encoder.args).cuda()
        ckpt_name = opt.ckpt if opt.ckpt.endswith(('.pt', '.safetensors')) else f'ckpts/encoder_{opt.ckpt}.pt'
        ckpt_path = resolve_model_file(opt.model_root, ckpt_name)
        encoder.load_state_dict(load_state_dict_file(ckpt_path), strict=False)
        encoder.eval()
        print(f'Loaded model from {ckpt_path}')

    dataset = getattr(datasets, cfg.dataset.name)(roots=opt.output_dir, **cfg.dataset.args)
    print(f'Loaded dataset class from {opt.output_dir}')
    
    os.makedirs(os.path.join(opt.output_dir, 'latents', latent_name), exist_ok=True)

    # get file list
    if os.path.exists(os.path.join(opt.output_dir, 'metadata.csv')):
        metadata = pd.read_csv(os.path.join(opt.output_dir, 'metadata.csv'))
    else:
        raise ValueError('metadata.csv not found')
    print(f"Original metadata num: {len(metadata)}")
    if opt.instances is not None:
        with open(opt.instances, 'r') as f:
            sha256s = [line.strip() for line in f]
        metadata = metadata[metadata['sha256'].isin(sha256s)]
    else:
        print(f"Filter low aesthetic score: {opt.filter_low_aesthetic_score}")
        if opt.filter_low_aesthetic_score is not None:
            print(f"nan num: {metadata['aesthetic_score'].isna().sum()}")
            print(f"low aesthetic score num: {(metadata['aesthetic_score'] < opt.filter_low_aesthetic_score).sum()}")
            metadata = metadata[metadata['aesthetic_score'].isna() | (metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score)]
        print(f"Filtered metadata num: {len(metadata)}")
        if f'latent_{latent_name}' in metadata.columns:
            metadata = metadata[metadata[f'latent_{latent_name}'] == False]
        print(f"Filtered metadata num: {len(metadata)}")

    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]
    dataset.set_metadata(roots=[opt.output_dir], metadatas=[metadata])
    records = []
    
    # filter out objects that are already processed
    sha256s = list(metadata['sha256'].values)
    for sha256 in copy.copy(sha256s):
        if os.path.exists(os.path.join(opt.output_dir, 'latents', latent_name, f'{sha256}.npz')):
            records.append({'sha256': sha256, f'latent_{latent_name}': True})
            sha256s.remove(sha256)
    print(f"Remaining metadata num: {len(sha256s)}")
    print(f"NOTE: There is a slow background OT computing process after encoding,")
    print(f"NOTE: so it is normal for the program to appear stuck for a (long) while even after the tqdm progress bar reaches 100%.")

    # encode latents
    load_queue = Queue(maxsize=4)
    try:
        with ThreadPoolExecutor(max_workers=32) as loader_executor, \
            ThreadPoolExecutor(max_workers=16) as saver_executor:
            def loader(sha256):
                try:
                    feats = dataset.get_instance(opt.output_dir, sha256)
                    load_queue.put((sha256, feats))
                except Exception as e:
                    load_queue.put(None)
                    # --- PATCH: print full traceback ---
                    import traceback
                    traceback.print_exc()
                    print(f"Error loading features for {sha256}: {e}")
                    # ------------------------------------
            loader_executor.map(loader, sha256s)
            
            def saver(sha256, pack):
                try:
                    for l, latent_list in pack.items():
                        for item in latent_list:
                            sobol_seq_3d = item.pop('sobol_seq_3d')
                            query_points = item.pop('query_points')
                            item['3d_ot_permutation'] = ot_solve_perm(sobol_seq_3d, query_points)
                    
                    save_path = os.path.join(opt.output_dir, 'latents', latent_name, f'{sha256}.npz')
                    np.savez_compressed(save_path, **pack)
                    records.append({'sha256': sha256, f'latent_{latent_name}': True})
                except Exception as e:
                    print(f"Error saving latents for {sha256}: {e}")
                
            for _ in tqdm(range(len(sha256s)), desc="Extracting latents"):
                load_get = load_queue.get()
                if load_get is None:
                    continue
                sha256, feats = load_get

                def encode_latent():
                    """
                    Encode latents to a dictionary of numpy arrays.
                    Output pack format:
                        - latent length 1
                            - list of 4 rotations
                                - 'latent': (L, D)
                                - '3d_ot_permutation': (L,) permute latent to it 3d sobol ot order
                                - '3d_fgw_{alpha}_permutation': (L,) permute latent to it 3d sobol fgw order
                                - 'feature_ot_permutation': (L,) permute latent to it feature space sobol ot order
                                - 'points': (L, 3)
                        - latent length 2
                            - ...
                    """
                    pack = {str(l): [] for l in opt.latent_length}
                    
                    def rotate_points(points, axis, k):
                        if k == 0: return points
                        points = points - 0.5
                        for _ in range(k):
                            x, y, z = points[:, 0], points[:, 1], points[:, 2]
                            if axis == 0:
                                points = torch.stack([x, -z, y], dim=1)
                            elif axis == 1:
                                points = torch.stack([z, y, -x], dim=1)
                            elif axis == 2:
                                points = torch.stack([-y, x, z], dim=1)
                        points = points + 0.5
                        return points

                    for rot_idx in range(4):
                        cond = feats['cond'].clone()
                        cond['points'] = rotate_points(cond['points'], opt.rot90_axis, rot_idx)
                        if 'points2' in cond:
                            cond['points2'] = rotate_points(cond['points2'], opt.rot90_axis, rot_idx)
                        
                        cond_cuda = cond[None].cuda()

                        for latent_token_length in opt.latent_length:
                            latent, query_points = encoder(x=None, cond=cond_cuda, sample_posterior=True, return_raw=False, return_fps=True, q_token_length=latent_token_length)
                            latent, query_points = latent[0], query_points[0]
                            L, D = latent.shape
                            # NOTE: make sure seed is the same
                            sobol_seq_3d = torch.quasirandom.SobolEngine(dimension=3, scramble=True, seed=123).draw(L).cuda()
                            # sobol_seq_feat = torch.quasirandom.SobolEngine(dimension=D, scramble=True, seed=123).draw(L).cuda()
                            
                            item = {}
                            item[f'latent'] = latent.detach().cpu().numpy()
                            # item[f'3d_ot_permutation'] = ot_solve_perm(sobol_seq_3d, query_points).detach().cpu().numpy()
                            # item[f'feature_ot_permutation'] = ot_solve_perm(sobol_seq_feat, latent).detach().cpu().numpy()
                            item[f'points'] = query_points.detach().cpu().numpy()
                            
                            item['sobol_seq_3d'] = sobol_seq_3d.detach().cpu().numpy()
                            item['query_points'] = query_points.detach().cpu().numpy()
                            
                            pack[str(latent_token_length)].append(item)
                    return pack
                pack = encode_latent()

                saver_executor.submit(saver, sha256, pack)
                
            saver_executor.shutdown(wait=True)
    except Exception as e:
        # --- PATCH: print full traceback ---
        import traceback
        traceback.print_exc()
        print(f"Error happened during processing: {e}")
        # ------------------------------------
        
    records = pd.DataFrame.from_records(records)
    records.to_csv(os.path.join(opt.output_dir, f'latent_{latent_name}_{opt.rank}.csv'), index=False)
