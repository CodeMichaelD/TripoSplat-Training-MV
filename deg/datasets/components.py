from typing import *
from abc import abstractmethod
import os
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from tensordict import TensorDict

class StandardDatasetBase(Dataset):
    """
    Base class for standard datasets.

    Args:
        roots (str): paths to the dataset
    """

    def __init__(self,
        roots: str,
    ):
        super().__init__()
        self.roots = roots.split(',')
        self.instances = []
        self.metadata = pd.DataFrame()
        
        self._stats = {}
        for root in self.roots:
            key = os.path.basename(root)
            self._stats[key] = {}
            metadata = pd.read_csv(os.path.join(root, 'metadata.csv'))
            self._stats[key]['Total'] = len(metadata)
            metadata, stats = self.filter_metadata(metadata)
            self._stats[key].update(stats)
            self.instances.extend([(root, sha256) for sha256 in metadata['sha256'].values])
            metadata.set_index('sha256', inplace=True)
            self.metadata = pd.concat([self.metadata, metadata])
            
    def set_metadata(self, roots: List[str], metadatas: List[pd.DataFrame]):
        self.metadata = pd.concat(metadatas)
        self.instances = []
        for root, metadata in zip(roots, metadatas):
            self.instances.extend([(root, sha256) for sha256 in metadata['sha256'].values])
            
    @abstractmethod
    def filter_metadata(self, metadata: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        pass
    
    @abstractmethod
    def get_instance(self, root: str, instance: str) -> Dict[str, Any]:
        pass
        
    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index) -> Dict[str, Any]:
        try:
            root, instance = self.instances[index]
            return self.get_instance(root, instance)
        except Exception as e:
            print(e)
            return self.__getitem__(np.random.randint(0, len(self)))
        
    def __str__(self):
        lines = []
        lines.append(self.__class__.__name__)
        lines.append(f'  - Total instances: {len(self)}')
        lines.append(f'  - Sources:')
        for key, stats in self._stats.items():
            lines.append(f'    - {key}:')
            for k, v in stats.items():
                lines.append(f'      - {k}: {v}')
        return '\n'.join(lines)


class TextConditionedMixin:
    def __init__(self, roots, **kwargs):
        super().__init__(roots, **kwargs)
        self.captions = {}
        for instance in self.instances:
            sha256 = instance[1]
            self.captions[sha256] = json.loads(self.metadata.loc[sha256]['captions'])
    
    def filter_metadata(self, metadata):
        metadata, stats = super().filter_metadata(metadata)
        metadata = metadata[metadata['captions'].notna()]
        stats['With captions'] = len(metadata)
        return metadata, stats
    
    def get_instance(self, root, instance):
        pack = super().get_instance(root, instance)
        text = np.random.choice(self.captions[instance])
        pack['cond'] = text
        return pack
    
    
class ImageConditionedMixin:
    def __init__(self, roots, *, image_size=518, use_crop_aug=True, **kwargs):
        self.image_size = image_size
        self.use_crop_aug = use_crop_aug
        super().__init__(roots, **kwargs)
    
    def filter_metadata(self, metadata):
        metadata, stats = super().filter_metadata(metadata)
        metadata = metadata[metadata[f'cond_rendered']]
        stats['Cond rendered'] = len(metadata)
        return metadata, stats
    
    def get_instance(self, root, instance):
        pack = super().get_instance(root, instance)
       
        image_root = os.path.join(root, 'renders_cond', instance)
        with open(os.path.join(image_root, 'transforms.json')) as f:
            metadata = json.load(f)
        n_views = len(metadata['frames'])
        view = np.random.randint(n_views)
        metadata = metadata['frames'][view]

        image_path = os.path.join(image_root, metadata['file_path'])
        image = Image.open(image_path)

        if self.use_crop_aug:
            alpha = np.array(image.getchannel(3))
            bbox = np.array(alpha).nonzero()
            bbox = [bbox[1].min(), bbox[0].min(), bbox[1].max(), bbox[0].max()]
            center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
            hsize = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
            aug_size_ratio = 1.2
            aug_hsize = hsize * aug_size_ratio
            aug_center_offset = [0, 0]
            aug_center = [center[0] + aug_center_offset[0], center[1] + aug_center_offset[1]]
            aug_bbox = [int(aug_center[0] - aug_hsize), int(aug_center[1] - aug_hsize), int(aug_center[0] + aug_hsize), int(aug_center[1] + aug_hsize)]
            image = image.crop(aug_bbox)

        image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        alpha = image.getchannel(3)
        image = image.convert('RGB')
        image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
        alpha = torch.tensor(np.array(alpha)).float() / 255.0
        image = image * alpha.unsqueeze(0)
        pack['cond'] = image
       
        return pack
    
class ImageConditionedMixinv2:
    def __init__(self, roots, *, cond_resolution=512, cond_resolution_dino=518, load_dino_cond=True, load_mv_image_latents_cond=False, use_crop_aug=True, **kwargs):
        self.cond_resolution = cond_resolution
        self.cond_image_size = (cond_resolution, cond_resolution)
        self.load_dino_cond = load_dino_cond
        self.load_mv_image_latents_cond = load_mv_image_latents_cond
        self.cond_resolution_dino = cond_resolution_dino
        self.cond_dino_image_size = (cond_resolution_dino, cond_resolution_dino)
        self.use_crop_aug = use_crop_aug
        super().__init__(roots=roots, **kwargs)

    def filter_metadata(self, metadata):
        metadata, stats = super().filter_metadata(metadata)
        metadata = metadata[metadata[f'cond_rendered']]
        stats['Cond rendered'] = len(metadata)
        return metadata, stats
    
    def get_instance(self, root, instance):
        pack = super().get_instance(root, instance)
        out_dict = {}

        transforms_meta_path = os.path.join(root, 'renders_cond', instance, 'transforms.json')
        transforms_meta = json.load(open(transforms_meta_path, 'r'))
        n_views = len(transforms_meta['frames'])
        view = np.random.randint(n_views)
        metadata = transforms_meta['frames'][view]
        
        if self.load_mv_image_latents_cond:
            assert not (self.load_mv_image_latents or self.load_mv_image), "mv image cond conflict with mv image latents or mv image"
            mv_latents = np.load(os.path.join(root, 'mv_latents', self.mv_latent_model, f'{instance}.npz'))
            z = torch.tensor(mv_latents['mean']).float() # shape (N, C, H, W)
            view_id_list = [int(vid) for vid in self.views_map.keys()]
            z = z[view_id_list]
            out_dict.update({'cond_mv_hidden_states': z})

        if self.load_mv_image or self.load_dino_cond:
            image_path = os.path.join(root, "renders_cond", instance, metadata['file_path'])
            image = Image.open(image_path)

            if self.use_crop_aug:
                # cond image resize
                alpha = np.array(image.getchannel(3))
                bbox = np.array(alpha).nonzero()
                bbox = [bbox[1].min(), bbox[0].min(), bbox[1].max(), bbox[0].max()]
                center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
                hsize = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
                aug_size_ratio = 1.2
                aug_hsize = hsize * aug_size_ratio
                aug_center_offset = [0, 0]
                aug_center = [center[0] + aug_center_offset[0], center[1] + aug_center_offset[1]]
                aug_bbox = [int(aug_center[0] - aug_hsize), int(aug_center[1] - aug_hsize), int(aug_center[0] + aug_hsize), int(aug_center[1] + aug_hsize)]
                image = image.crop(aug_bbox)

        if self.load_mv_image:
            image_mv = image.resize(self.cond_image_size, Image.Resampling.LANCZOS)
            alpha_mv = image_mv.getchannel(3)
            image_mv = image_mv.convert('RGB')
            image_mv = torch.tensor(np.array(image_mv)).permute(2, 0, 1).float() / 255.0
            alpha_mv = torch.tensor(np.array(alpha_mv)).float() / 255.0
            image_mv = image_mv * alpha_mv.unsqueeze(0) + self.bg_color[:,None,None] * (1 - alpha_mv.unsqueeze(0))
            out_dict.update({'cond_hidden_states': image_mv})

        elif self.load_mv_image_latents or self.load_mv_image_latents_cond:
            cond_latents = np.load(os.path.join(root, 'cond_latents', self.mv_latent_model, f'{instance}.npz'))
            z = torch.tensor(cond_latents['mean']).float()
            z = z[view]
            out_dict.update({'cond_hidden_states': z})
            
        if self.load_dino_cond:
            image_dino = image.resize(self.cond_dino_image_size, Image.Resampling.LANCZOS)
            alpha_dino = image_dino.getchannel(3)
            image_dino = image_dino.convert('RGB')
            image_dino = torch.tensor(np.array(image_dino)).permute(2, 0, 1).float() / 255.0
            alpha_dino = torch.tensor(np.array(alpha_dino)).float() / 255.0
            image_dino = image_dino * alpha_dino.unsqueeze(0) + self.bg_color[:,None,None] * (1 - alpha_dino.unsqueeze(0))
            out_dict.update({'cond_hidden_states_dino': image_dino})
        pack['cond'] = TensorDict(out_dict)
       
        return pack
    