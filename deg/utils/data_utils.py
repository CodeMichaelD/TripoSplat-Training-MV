from typing import *
import math
import torch
import numpy as np
from torch.utils.data import Sampler, Dataset, DataLoader, DistributedSampler
import torch.distributed as dist
from tensordict import TensorDict
from PIL import Image, ImageFilter, ImageEnhance
import random
import io
import cv2

def jitter_image(image: Image.Image, prob: float = 0.0, method: str = None) -> Image.Image:
    """
    Apply various image jittering augmentations with a given probability.
    Jittering includes: blur, jpeg compression, color jitter, distortion (perspective), noise.
    """
    if prob <= 0.0:
        return image

    if method is None:
        if random.random() >= prob:
            return image
        methods = ['blur', 'jpeg', 'color', 'distortion', 'noise', 'crop']
        method = random.choice(methods)
    
    if method == 'blur':
        radius = random.uniform(0.5, 2.0)
        return image.filter(ImageFilter.GaussianBlur(radius=radius))
        
    elif method == 'jpeg':
        # Split alpha if exists
        if image.mode == 'RGBA':
            rgb = image.convert('RGB')
            alpha = image.split()[-1]
            
            buffer = io.BytesIO()
            rgb.save(buffer, format="JPEG", quality=random.randint(50, 95))
            buffer.seek(0)
            rgb_jittered = Image.open(buffer)
            
            return Image.merge('RGBA', (*rgb_jittered.split(), alpha))
        else:
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=random.randint(50, 95))
            buffer.seek(0)
            return Image.open(buffer)
        
    elif method == 'color':
        # Apply to RGB only, preserve alpha
        if image.mode == 'RGBA':
            rgb = image.convert('RGB')
            alpha = image.split()[-1]
        else:
            rgb = image
            alpha = None
            
        enhancers = [ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color]
        random.shuffle(enhancers)
        for enhancer in enhancers:
            factor = random.uniform(0.8, 1.2)
            rgb = enhancer(rgb).enhance(factor)
            
        if alpha:
            return Image.merge('RGBA', (*rgb.split(), alpha))
        else:
            return rgb
        
    elif method == 'distortion':
        # Perspective distortion using cv2
        # Ensure image has alpha channel for transparent background
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
            
        img_np = np.array(image)
        h, w = img_np.shape[:2]
        
        # Define source points (corners of the original image)
        src_pts = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        
        # Define destination points with small random shifts (inside the image)
        # This shrinks the image, leaving empty space around
        offset_x = w * 0.1
        offset_y = h * 0.1
        
        dst_pts = np.float32([
            [random.uniform(0, offset_x), random.uniform(0, offset_y)], 
            [w - random.uniform(0, offset_x), random.uniform(0, offset_y)],
            [random.uniform(0, offset_x), h - random.uniform(0, offset_y)],
            [w - random.uniform(0, offset_x), h - random.uniform(0, offset_y)]
        ])
        
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        # Use BORDER_CONSTANT with transparent color (0,0,0,0)
        img_distorted = cv2.warpPerspective(img_np, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        return Image.fromarray(img_distorted)
        
    elif method == 'noise':
        img_np = np.array(image)
        # Apply to RGB only
        if image.mode == 'RGBA':
            rgb = img_np[:, :, :3]
            alpha = img_np[:, :, 3]
        else:
            rgb = img_np
            alpha = None

        # Add Gaussian noise
        noise = np.random.normal(0, 15, rgb.shape).astype(np.int16)
        rgb = rgb.astype(np.int16) + noise
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        
        if alpha is not None:
            return Image.fromarray(np.dstack((rgb, alpha)))
        else:
            return Image.fromarray(rgb)
        
    return image

def slice_batch_and_cuda(data, batch_slice):
    new_batch = {}
    for k in data:
        if isinstance(data[k], list):
            new_batch[k] = data[k][batch_slice]
        elif isinstance(data[k], torch.Tensor) or isinstance(data[k], TensorDict):
            new_batch[k] = data[k][batch_slice].cuda()
        elif isinstance(data[k], dict):
            new_batch[k] = slice_batch_and_cuda(data[k], batch_slice)
        else:
            new_batch[k] = data[k]
    return new_batch

def recursive_to_device(
    data: Any,
    device: torch.device,
    non_blocking: bool = False,
) -> Any:
    """
    Recursively move all tensors in a data structure to a device.
    """
    if hasattr(data, "to"):
        return data.to(device, non_blocking=non_blocking)
    elif isinstance(data, (list, tuple)):
        return type(data)(recursive_to_device(d, device, non_blocking) for d in data)
    elif isinstance(data, dict):
        return {k: recursive_to_device(v, device, non_blocking) for k, v in data.items()}
    else:
        return data


def load_balanced_group_indices(
    load: List[int],
    num_groups: int,
    equal_size: bool = False,
) -> List[List[int]]:
    """
    Split indices into groups with balanced load.
    """
    if equal_size:
        group_size = len(load) // num_groups
    indices = np.argsort(load)[::-1]
    groups = [[] for _ in range(num_groups)]
    group_load = np.zeros(num_groups)
    for idx in indices:
        min_group_idx = np.argmin(group_load)
        groups[min_group_idx].append(idx)
        if equal_size and len(groups[min_group_idx]) == group_size:
            group_load[min_group_idx] = float('inf')
        else:
            group_load[min_group_idx] += load[idx]
    return groups


def cycle(data_loader: DataLoader) -> Iterator:
    while True:
        for data in data_loader:
            if isinstance(data_loader.sampler, ResumableSampler):
                data_loader.sampler.idx += data_loader.batch_size   # type: ignore[attr-defined]
            yield data
        if isinstance(data_loader.sampler, DistributedSampler):
            data_loader.sampler.epoch += 1
        if isinstance(data_loader.sampler, ResumableSampler):
            data_loader.sampler.epoch += 1
            data_loader.sampler.idx = 0
        dataset = getattr(data_loader, 'dataset', None)
        if hasattr(dataset, 'set_epoch'):
            dataset.set_epoch(getattr(data_loader.sampler, 'epoch', 0))
        

class ResumableSampler(Sampler):
    """
    Distributed sampler that is resumable.

    Args:
        dataset: Dataset used for sampling.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.
    """

    def __init__(
        self,
        dataset: Dataset,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        sync: bool = False,
    ) -> None:
        self.dataset = dataset
        self.epoch = 0
        self.idx = 0
        self.drop_last = drop_last
        self.sync = sync
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        
        if self.sync:
            self.num_samples = len(self.dataset)
            self.total_size = len(self.dataset)
        else:
            # If the dataset length is evenly divisible by # of replicas, then there
            # is no need to drop any data, since the dataset will be split equally.
            if self.drop_last and len(self.dataset) % self.world_size != 0:  # type: ignore[arg-type]
                # Split to nearest available length that is evenly divisible.
                # This is to ensure each rank receives the same amount of data when
                # using this Sampler.
                self.num_samples = math.ceil(
                    (len(self.dataset) - self.world_size) / self.world_size  # type: ignore[arg-type]
                )
            else:
                self.num_samples = math.ceil(len(self.dataset) / self.world_size)  # type: ignore[arg-type]
            self.total_size = self.num_samples * self.world_size
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.sync:
            if not self.drop_last:
                # add extra samples to make it evenly divisible
                padding_size = self.total_size - len(indices)
                if padding_size <= len(indices):
                    indices += indices[:padding_size]
                else:
                    indices += (indices * math.ceil(padding_size / len(indices)))[
                        :padding_size
                    ]
            else:
                # remove tail of data to make it evenly divisible.
                indices = indices[: self.total_size]
            assert len(indices) == self.total_size

            # subsample
            indices = indices[self.rank : self.total_size : self.world_size]
        
        # resume from previous state
        indices = indices[self.idx:]

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def state_dict(self) -> Dict[str, int]:
        return {
            'epoch': self.epoch,
            'idx': self.idx,
        }
        
    def load_state_dict(self, state_dict):
        self.epoch = state_dict['epoch']
        self.idx = state_dict['idx']
        

class BalancedResumableSampler(ResumableSampler):
    """
    Distributed sampler that is resumable and balances the load among the processes.

    Args:
        dataset: Dataset used for sampling.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.
    """

    def __init__(
        self,
        dataset: Dataset,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        batch_size: int = 1,
        sync: bool = False,
    ) -> None:
        assert hasattr(dataset, 'loads'), 'Dataset must have "loads" attribute to use BalancedResumableSampler'
        super().__init__(dataset, shuffle, seed, drop_last, sync=sync)
        self.batch_size = batch_size
        self.loads = dataset.loads
        
    def __iter__(self) -> Iterator:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # balance load among processes
        if not self.sync:
            num_batches = len(indices) // (self.batch_size * self.world_size)
            balanced_indices = []
            for i in range(num_batches):
                start_idx = i * self.batch_size * self.world_size
                end_idx = (i + 1) * self.batch_size * self.world_size
                batch_indices = indices[start_idx:end_idx]
                batch_loads = [self.loads[idx] for idx in batch_indices]
                groups = load_balanced_group_indices(batch_loads, self.world_size, equal_size=True)
                balanced_indices.extend([batch_indices[j] for j in groups[self.rank]])
            indices = balanced_indices
        
        # resume from previous state
        indices = indices[self.idx:]

        return iter(indices)
