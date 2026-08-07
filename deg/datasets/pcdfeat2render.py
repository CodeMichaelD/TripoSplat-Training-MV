import os
from PIL import Image
import json
import numpy as np
import torch
import open3d as o3d
from .components import StandardDatasetBase
import utils3d
from tensordict import TensorDict
import einops
from typing import *
from ..utils.postprocessing_utils import octree_from_points
import hashlib

class PcdFeat2Render(StandardDatasetBase):
    """
    Dataset for Structured Latent and rendered images.
    
    Args:
        roots (str): paths to the dataset
        image_size (int): size of the image
        latent_model (str): latent model name
        min_aesthetic_score (float): minimum aesthetic score
        max_num_voxels (int): maximum number of voxels
    """
    def __init__(
        self,
        roots: str,
        image_size: int,
        target_image_num: int = 8,
        cond_image_num: int = None,
        cond_mv_latent_num: int = None,
        min_aesthetic_score: float = 5.0,
        max_num_points: int = 8192,
        max_num_points_cond: int = 16384,
        model: str = 'dinov2_vitl14_reg',
        max_num_points_cond2: int = 16384,
        model2: Optional[str] = None,
        output_mode: Literal["pcd_x0", "pcd_cond"] = "pcd_x0",
        read_depth: bool = False,
        fitone: int = None, # DEBUG
        fitoner: int = 1, # DEBUG
        seed: Optional[int] = None,
    ):
        self.image_size = image_size
        self.target_image_num = target_image_num
        self.cond_image_num = cond_image_num
        self.cond_mv_latent_num = cond_mv_latent_num
        self.min_aesthetic_score = min_aesthetic_score
        self.value_range = (0, 1)
        self.max_num_points = max_num_points
        self.max_num_points_cond = max_num_points_cond
        self.max_num_points_cond2 = max_num_points_cond2
        self.model = model
        self.model2 = model2
        self.output_mode = output_mode
        self.read_depth = read_depth
        self.fitone = fitone
        self.fitoner = fitoner
        self._seed = seed
        self._epoch = 0
        
        super().__init__(roots)
        
    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)

    def _rng(self, instance: str, stream: str):
        base = 0 if self._seed is None else int(self._seed)
        key = f"{instance}|{self._epoch}|{base}|{stream}"
        h = hashlib.sha256(key.encode()).digest()
        seed64 = int.from_bytes(h[:8], byteorder="big", signed=False)
        return np.random.default_rng(seed64)

    def filter_metadata(self, metadata):
        stats = {}
        metadata = metadata[metadata[f'pcd_feature_{self.model}']]
        if self.model2 is not None:
            metadata = metadata[metadata[f'pcd_feature_{self.model2}']]
        stats['With pcds'] = len(metadata)
        metadata = metadata[metadata['aesthetic_score'].isna() | (metadata['aesthetic_score'] >= self.min_aesthetic_score)]
        stats[f'Aesthetic score >= {self.min_aesthetic_score}'] = len(metadata)
        if self.fitone is not None:
            R = self.fitoner # repeat 16 times
            metadata = metadata[:self.fitone]
            metadata = metadata.loc[metadata.index.repeat(R)].reset_index(drop=True)
            stats[f'Fit one'] = len(metadata)

        return metadata, stats
    
    def _get_mv_latent(self, root, instance, mv_latent_num, stream: str = "mv_latent"):
        mv_latent_path = os.path.join(root, 'mv_latents', self.model2, f'{instance}.npz')
        mv_latent = np.load(mv_latent_path)
        """
        expected mv latent keys:
            patchtokens shape: (B, C, H, W)
            extrinsics shape: (B, 4, 4)
            intrinsics shape: (B, 3, 3)
        """
        patchtokens = mv_latent['patchtokens']
        extrinsics = mv_latent['extrinsics']
        intrinsics = mv_latent['intrinsics']
        latent_view_num = patchtokens.shape[0]
        rng = self._rng(instance, stream)
        if mv_latent_num < latent_view_num:
            mv_latent_idx = rng.choice(latent_view_num, mv_latent_num, replace=False)
        else:
            mv_latent_idx = rng.choice(latent_view_num, mv_latent_num, replace=True)
        mv_latent = {
            'mv_features': torch.from_numpy(patchtokens[mv_latent_idx]).float(),
            'mv_extrinsics': torch.from_numpy(extrinsics[mv_latent_idx]).float(),
            'mv_intrinsics': torch.from_numpy(intrinsics[mv_latent_idx]).float(),
        }
        return mv_latent

    def _get_images(self, root, instance, image_num, stream: str = "images"):
        with open(os.path.join(root, 'renders', instance, 'transforms.json')) as f:
            metadata = json.load(f)
        n_views = len(metadata['frames'])
        rng = self._rng(instance, stream)
        if image_num < n_views:
            view_idx = rng.choice(n_views, image_num, replace=False)
        else:
            view_idx = rng.choice(n_views, image_num, replace=True)
        images = []
        depths = []
        alphas = []
        extrinsics = []
        intrinsics = []
        for view in view_idx:
            metadata_i = metadata['frames'][view]
            fov = metadata_i['camera_angle_x']
            intrinsic = utils3d.torch.intrinsics_from_fov_xy(torch.tensor(fov), torch.tensor(fov))
            c2w = torch.tensor(metadata_i['transform_matrix'])
            c2w[:3, 1:3] *= -1
            extrinsic = torch.inverse(c2w)

            image_path = os.path.join(root, 'renders', instance, metadata_i['file_path'])
            image = Image.open(image_path)
            alpha = image.getchannel(3)
            image = image.convert('RGB')
            image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
            alpha = alpha.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
            image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
            alpha = torch.tensor(np.array(alpha)).float() / 255.0
            
            if self.read_depth:
                depth_min = metadata_i['depth']['min']
                depth_max = metadata_i['depth']['max']
                metadata_depth_i = metadata['depth_frames'][view]
                depth_path = os.path.join(root, 'renders', instance, metadata_depth_i['file_path'])
                depth_i = Image.open(depth_path)
                depth_i = depth_i.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
                depth_np = np.array(depth_i)
                depth_dtype = depth_np.dtype
                if depth_dtype == np.uint16:
                    depth = torch.tensor(depth_np).float() / 65535.0 * (depth_max - depth_min) + depth_min
                elif depth_dtype == np.uint8:
                    depth = torch.tensor(depth_np).float() / 255.0 * (depth_max - depth_min) + depth_min
                else:
                    raise NotImplementedError(f"Depth dtype {depth_dtype} not supported")
                depths.append(depth)
            images.append(image)
            alphas.append(alpha)
            extrinsics.append(extrinsic)
            intrinsics.append(intrinsic)
        images = torch.stack(images)
        alphas = torch.stack(alphas)
        extrinsics = torch.stack(extrinsics)
        intrinsics = torch.stack(intrinsics)
        
        ret = {
            'images': images,
            'alphas': alphas,
            'extrinsics': extrinsics,
            'intrinsics': intrinsics,
        }
        if self.read_depth:
            depths = torch.stack(depths)
            ret['depths'] = depths
        return TensorDict(ret)
    
    def _get_pcds_raw(self, root, instance):
        data = np.load(os.path.join(root, 'pcd_features', self.model, instance + '.npz'))
        points_q = torch.tensor(data["pcds"]).float()
        features = torch.tensor(data["patchtokens"]).float()
        if self.model2 is not None:
            data2 = np.load(os.path.join(root, 'pcd_features', self.model2, instance + '.npz'))
            points_q2 = torch.tensor(data2["pcds"]).float()
            features2 = torch.tensor(data2["patchtokens"]).float()
            return TensorDict({
                'points': points_q,
                'features': features,
                'points2': points_q2,
                'features2': features2,
            })
        return TensorDict({
            'points': points_q,
            'features': features,
        })
        
    def _get_pcds(self, pcds_dict, max_num_points, *, instance: str, stream: str = "pcd"):
        points_q = pcds_dict['points']
        features = pcds_dict['features']
        
        # sample max_num_points points
        assert points_q.shape[0] >= max_num_points, f"Point cloud has less points ({points_q.shape[0]}) than max_num_points ({max_num_points})"
        if points_q.shape[0] > max_num_points:
            rng = self._rng(instance, stream)
            choice = rng.choice(points_q.shape[0], max_num_points, replace=False)
            points_q = points_q[choice]
            features = features[choice]
            
        # convert to [0, 1]
        points_q = points_q + 0.5
        points_q = points_q.clamp(0.0, 1.0)
        
        if self.model2 is not None:
            points_q2 = pcds_dict['points2']
            features2 = pcds_dict['features2']
            # sample max_num_points_cond2 points
            assert points_q2.shape[0] >= self.max_num_points_cond2, f"Point cloud has less points ({points_q2.shape[0]}) than max_num_points_cond2 ({self.max_num_points_cond2})"
            if points_q2.shape[0] > self.max_num_points_cond2:
                rng2 = self._rng(instance, stream + "_2")
                choice2 = rng2.choice(points_q2.shape[0], self.max_num_points_cond2, replace=False)
                points_q2 = points_q2[choice2]
                features2 = features2[choice2]
            # convert to [0, 1]
            points_q2 = points_q2 + 0.5
            points_q2 = points_q2.clamp(0.0, 1.0)
            return TensorDict({
                'points': points_q,
                'features': features,
                'points2': points_q2,
                'features2': features2,
            })
        return TensorDict({
            'points': points_q,
            'features': features,
        })
        


    @torch.no_grad()
    def visualize_sample(self, sample: dict):
        return einops.rearrange(sample['target_images']['images'], "b nv c h w -> c (b h) (nv w)")[None]
    
    def get_instance(self, root, instance):
        pcds_raw = self._get_pcds_raw(root, instance)
        pcds = self._get_pcds(pcds_raw, self.max_num_points, instance=instance, stream="pcd")
        target_images = self._get_images(root, instance, self.target_image_num, stream="images")
        if self.output_mode == "pcd_cond":
            cond = self._get_pcds(pcds_raw, self.max_num_points_cond, instance=instance, stream="pcd_cond")
            if self.cond_image_num is not None:
                # merge two tensordicts
                cond_images = self._get_images(root, instance, self.cond_image_num, stream="images_cond")
                cond.update(cond_images)
            if self.cond_mv_latent_num is not None:
                cond_mv_latent = self._get_mv_latent(root, instance, self.cond_mv_latent_num, stream="mv_latent_cond")
                cond.update(cond_mv_latent)
            x_0 = pcds
            return {
                "x_0": x_0,
                "cond": cond,
                "target_images": target_images,
            }
        elif self.output_mode == "pcd_x0":
            ret = {
                "x_0": pcds,
                "target_images": target_images,
            }
            if self.cond_image_num is not None:
                cond_images = self._get_images(root, instance, self.cond_image_num, stream="images_cond")
                ret.update({'cond': cond_images})
        return ret
        
    @staticmethod
    def collate_fn(batch, split_size=None):
        # specially handle TensorDict
        tdicts = {key: [] for key in batch[0].keys() if isinstance(batch[0][key], TensorDict)}
        # collate each tensor dict
        
        for i, b in enumerate(batch):
            for key in tdicts.keys():
                tdicts[key].append(b.pop(key))
        # collate with default collate_fn
        pack = torch.utils.data.default_collate(batch)
        for key in tdicts.keys():
            pack[key] = TensorDict({
                in_key: torch.stack([t[in_key] for t in tdicts[key]])  # Stack tensors for each key
                for in_key in tdicts[key][0].keys()
            }, batch_size=[len(batch)] + list(tdicts[key][0].shape))
        
        return pack
