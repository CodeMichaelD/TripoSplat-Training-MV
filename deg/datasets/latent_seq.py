import os
import json
from typing import *
import numpy as np
import torch
import utils3d
from ..renderers import GaussianRenderer
from ..representations import Gaussian
from .components import StandardDatasetBase, TextConditionedMixin, ImageConditionedMixin, ImageConditionedMixinv2
from .. import models
from ..utils.render_utils import get_renderer, encode_latent_camera, decode_latent_camera
from ..utils.data_utils import jitter_image
from ..utils.hf_utils import (
    DEFAULT_HF_REPO_ID,
    DEFAULT_HF_VAE_DECODER_FILE,
    default_vae_config_path,
    load_config,
    load_model_state_dict,
    resolve_model_file,
)
from ..models import OctreeProbabilityFixedlenDecoder
from PIL import Image
from tensordict import TensorDict

class GSOctreeLatentVisMixin:
    def __init__(
        self,
        *args,
        decode_mode: Literal['pcd', 'gs'] = 'pcd',
        pretrained_decoder: str = None,
        decoder_path: Optional[str] = None,
        decoder_ckpt: Optional[str] = None,
        decoder_config_path: Optional[str] = None,
        max_voxel_level: int = 8,
        max_sampled_points: int = 8192,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.decode_mode = decode_mode
        self.decoder = None
        self.decoder_gs = None
        self.pretrained_decoder = pretrained_decoder
        self.decoder_path = decoder_path
        self.decoder_ckpt = decoder_ckpt
        self.decoder_config_path = decoder_config_path
        self.max_voxel_level = max_voxel_level
        self.max_sampled_points = max_sampled_points
        
        # Build camera
        yaws = [np.pi, 0, 3 * np.pi / 2, np.pi / 2] + [np.pi * (1+1/6), np.pi * (1-1/6)]
        pitches = [0.0 for _ in range(4)] + [np.pi / 6 for _ in range(2)]
        self.vis_yaws = yaws
        self.vis_pitches = pitches

    def _resolve_pretrained_decoder_weight(self):
        if self.pretrained_decoder is not None:
            return self.pretrained_decoder
        if self.decoder_path is None:
            return f"hf://{DEFAULT_HF_REPO_ID}/{DEFAULT_HF_VAE_DECODER_FILE}"

        weight_name = self.decoder_ckpt
        if weight_name is None:
            path_parts = self.decoder_path.replace("hf://", "").strip("/").split("/")
            weight_name = (
                os.path.basename(DEFAULT_HF_VAE_DECODER_FILE)
                if len(path_parts) > 2 or os.path.isdir(self.decoder_path)
                else DEFAULT_HF_VAE_DECODER_FILE
            )
        elif not weight_name.endswith((".pt", ".safetensors")):
            weight_name = f"{weight_name}.safetensors"

        return resolve_model_file(self.decoder_path, weight_name)
        
    def _loading_decoder(self):
        if self.decode_mode == 'pcd' and self.decoder is not None:
            return
        
        if self.decode_mode == 'gs' and self.decoder_gs is not None:
            return
        
        local_decoder_config = (
            self.decoder_path is not None
            and os.path.exists(os.path.join(self.decoder_path, 'config.json'))
        )
        if local_decoder_config:
            cfg = json.load(open(os.path.join(self.decoder_path, 'config.json'), 'r'))
            
            # Load Geometry Decoder
            self.decoder = getattr(models, cfg['models']['decoder']['name'])(**cfg['models']['decoder']['args'])
            ckpt_path = os.path.join(self.decoder_path, 'ckpts', f'decoder_{self.decoder_ckpt}.pt')
            load_model_state_dict(self.decoder, ckpt_path, model_name='decoder', strict=True, map_location='cpu')
            self.decoder = self.decoder.cuda().eval()
            
            if self.decode_mode == 'gs':
                # Load GS Attribute Decoder
                self.decoder_gs = getattr(models, cfg['models']['decoder_gs']['name'])(**cfg['models']['decoder_gs']['args'])
                ckpt_path_gs = os.path.join(self.decoder_path, 'ckpts', f'decoder_gs_{self.decoder_ckpt}.pt')
                load_model_state_dict(self.decoder_gs, ckpt_path_gs, model_name='decoder_gs', strict=False, map_location='cpu') # FIXME
                self.decoder_gs = self.decoder_gs.cuda().eval()
            
        else:
            cfg = load_config(self.decoder_config_path or default_vae_config_path())
            weight_ref = self._resolve_pretrained_decoder_weight()

            # Load Geometry Decoder
            self.decoder = getattr(models, cfg['models']['decoder']['name'])(**cfg['models']['decoder']['args'])
            load_model_state_dict(self.decoder, weight_ref, model_name='decoder', strict=True, map_location='cpu')
            self.decoder = self.decoder.cuda().eval()

            if self.decode_mode == 'gs':
                # Load GS Attribute Decoder
                self.decoder_gs = getattr(models, cfg['models']['decoder_gs']['name'])(**cfg['models']['decoder_gs']['args'])
                load_model_state_dict(self.decoder_gs, weight_ref, model_name='decoder_gs', strict=False, map_location='cpu')
                self.decoder_gs = self.decoder_gs.cuda().eval()

    def _delete_decoder(self):
        del self.decoder
        del self.decoder_gs
        self.decoder = None
        self.decoder_gs = None

    def pcd_to_representation(self, pcds, scale=0.004):
        """
        copied from gs fit trainer
        pcds in range [0, 1]
        """
        reps = []
        for i in range(pcds.shape[0]):
            pcd = pcds[i]
            representation = Gaussian(
                sh_degree=0,
                aabb=[-0.5, -0.5, -0.5, 1.0, 1.0, 1.0],
                scaling_bias=scale,
            )
            setattr(representation, '_xyz', pcd)
            setattr(representation, '_features_dc', pcd[...,None,:] * 2 -1)
            setattr(representation, '_scaling', torch.zeros_like(pcd))
            setattr(representation, '_rotation', torch.zeros(pcd.shape[0], 4).to(pcd.device))
            setattr(representation, '_opacity', torch.ones(pcd.shape[0], 1).to(pcd.device) * 10.0)
            reps.append(representation)
        return reps

    @torch.no_grad()
    def _decode_gs_latent(self, z, batch_size=4):
        self._loading_decoder()
        reps = []
        if self.normalization is not None:
            z = z * self.std.to(z.device) + self.mean.to(z.device)
            
        for i in range(0, z.shape[0], batch_size):
            z_batch = z[i:i+batch_size]
            
            # Sample points
            points_pred = OctreeProbabilityFixedlenDecoder.sample(
                self.decoder, 
                z_batch, 
                num_points=self.max_sampled_points,
                level=self.max_voxel_level, 
                temperature=1.0, 
                algo='systematic'
            )
            
            if self.decode_mode == 'gs':
                # Predict GS attributes
                pred = self.decoder_gs(x=points_pred, cond=z_batch)
                
                # Convert to representation
                batch_reps = self.decoder_gs.to_representation(points_pred, pred)
            else:
                batch_reps = self.pcd_to_representation(points_pred['points'])
            
            reps.extend(batch_reps)
            
        self._delete_decoder()
        return reps

    @torch.no_grad()
    def visualize_sample(self, x_0: Union[torch.Tensor, dict], camera_pred: bool = False):
        if self.decoder_path is None and self.pretrained_decoder is None:
            return {}
        x_0 = x_0 if isinstance(x_0, torch.Tensor) else x_0['x_0']
        z = x_0['latent']
        reps = self._decode_gs_latent(z.cuda())
        
        renderer = get_renderer(reps[0])
        ret = {}

        if not camera_pred:
            exts = []
            ints = []
            for yaw, pitch in zip(self.vis_yaws, self.vis_pitches):
                orig = torch.tensor([
                    np.sin(yaw) * np.cos(pitch),
                    np.cos(yaw) * np.cos(pitch),
                    np.sin(pitch),
                ]).float().cuda() * 2
                fov = torch.deg2rad(torch.tensor(30)).cuda()
                extrinsics = utils3d.torch.extrinsics_look_at(orig, torch.tensor([0, 0, 0]).float().cuda(), torch.tensor([0, 0, 1]).float().cuda())
                intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
                exts.append(extrinsics)
                ints.append(intrinsics)

            images = []
            
            for i, rep in enumerate(reps):
                image = torch.zeros(3, 512*3, 512*2).cuda()
                tile = [3, 2]
                for j, (ext, intr) in enumerate(zip(exts, ints)):
                    res = renderer.render(rep, ext, intr)
                    image[:, 512 * (j // tile[1]):512 * (j // tile[1] + 1), 512 * (j % tile[1]):512 * (j % tile[1] + 1)] = res['color']
                images.append(image)
            
            ret["gs_latent_vis"] = torch.stack(images)

        if "camera" in x_0:
            camera_images = []
            
            camera = x_0["camera"]
            for i, rep in enumerate(reps):
                camera_param = decode_latent_camera(camera[i,0])
                ext = camera_param['extrinsics']
                intr = camera_param['intrinsics']
                res = renderer.render(rep, ext, intr)
                camera_images.append(res['color'])
            ret["camera_vis"] = torch.stack(camera_images)
            
        return ret


class GSOctreeLatent(GSOctreeLatentVisMixin, StandardDatasetBase):
    """
    GS Octree Latent Dataset
    
    Args:
        roots (str): path to the dataset
        latent_model (str): name of the latent model
        latent_token_length (int): token length of the latent to load
        min_aesthetic_score (float): minimum aesthetic score
        normalization (dict): normalization stats
        pretrained_decoder (str): name of the pretrained decoder
        decoder_path (str): path to the decoder, if given, will override the pretrained_decoder
        decoder_ckpt (str): name of the decoder checkpoint
    """
    def __init__(self,
        roots: str,
        *,
        latent_model: str,
        latent_token_length: int = 1024,
        min_aesthetic_score: float = 5.0,
        normalization: Optional[dict] = None,
        pretrained_decoder: str = None,
        decoder_path: Optional[str] = None,
        decoder_ckpt: Optional[str] = None,
        decoder_config_path: Optional[str] = None,
        max_voxel_level: int = 8,
        max_sampled_points: int = 8192,
        decode_mode: Literal['pcd', 'gs'] = 'pcd',
        ot_mode: Literal['none', '3d_ot', 'feature_ot'] = '3d_ot'
    ):
        self.latent_model = latent_model
        self.latent_token_length = str(latent_token_length)
        self.min_aesthetic_score = min_aesthetic_score
        self.normalization = normalization
        self.value_range = (0, 1) # Not strictly true for latents but kept for compatibility
        self.ot_mode = ot_mode
        
        super().__init__(
            roots,
            pretrained_decoder=pretrained_decoder,
            decoder_path=decoder_path,
            decoder_ckpt=decoder_ckpt,
            decoder_config_path=decoder_config_path,
            max_voxel_level=max_voxel_level,
            max_sampled_points=max_sampled_points,
            decode_mode=decode_mode,
        )
        
        if self.normalization is not None:
            self.mean = torch.tensor(self.normalization['mean']).reshape(-1, 1, 1, 1) # Adjust shape as needed
            self.std = torch.tensor(self.normalization['std']).reshape(-1, 1, 1, 1)
  
    def filter_metadata(self, metadata):
        stats = {}
        # Check if column exists, if not, might be handled differently or assume all processed if using separate file check
        if f'latent_{self.latent_model}' in metadata.columns:
            metadata = metadata[metadata[f'latent_{self.latent_model}']]
        stats['With gs latents'] = len(metadata)
        
        if 'aesthetic_score' in metadata.columns:
            metadata = metadata[metadata['aesthetic_score'].isna() | (metadata['aesthetic_score'] >= self.min_aesthetic_score)]
            stats[f'Aesthetic score >= {self.min_aesthetic_score}'] = len(metadata)
        return metadata, stats
                
    def get_instance(self, root, instance, rot_idx=None, **kwargs):
        # Load npz file
        path = os.path.join(root, 'latents', self.latent_model, f'{instance}.npz')
        try:
            data = np.load(path, allow_pickle=True)
            if rot_idx is None:
                latent_data = data[self.latent_token_length].item()
            else:
                latent_data = data[self.latent_token_length][rot_idx]
            z = torch.tensor(latent_data['latent']).float()
            
            if torch.isnan(z).any() or torch.isinf(z).any() or torch.abs(z).max() > 100:
                raise ValueError(f"Latent data contains NaNs or Infs or extreme values > 100")
                
            use_points = 'points' in latent_data
            points = torch.tensor(latent_data['points']).float() if use_points else None
            
            if self.ot_mode == '3d_ot':
                perm = torch.tensor(latent_data['3d_ot_permutation']).long()
                z = z[perm]
                points = points[perm] if use_points else None
            elif self.ot_mode == 'feature_ot':
                perm = torch.tensor(latent_data['feature_ot_permutation']).long()
                z = z[perm]
                points = points[perm] if use_points else None
            
            if self.normalization is not None:
                z = (z - self.mean) / self.std

            pack = {
                'x_0': TensorDict({'latent': z}),
            }
            if use_points:
                pack['points'] = points
            return pack
        except Exception as e:
            print(f"Error loading {path}: {e}")
            raise e

    @staticmethod
    def collate_fn(batch):
        # specially handle TensorDict
        tdicts = {key: [] for key in batch[0].keys() if isinstance(batch[0][key], TensorDict)}
        
        if not tdicts:
            return torch.utils.data.default_collate(batch)
            
        # collate each tensor dict
        for i, b in enumerate(batch):
            for key in tdicts.keys():
                tdicts[key].append(b.pop(key))
                
        # collate with default collate_fn
        pack = torch.utils.data.default_collate(batch)
        
        # stack tensordicts
        for key in tdicts.keys():
            pack[key] = torch.stack(tdicts[key])
            
        return pack
    

class TextConditionedGSOctreeLatent(TextConditionedMixin, GSOctreeLatent):
    """
    Text-conditioned GS octree dataset
    """
    pass


class ImageConditionedGSOctreeLatent(ImageConditionedMixin, GSOctreeLatent):
    """
    Image-conditioned GS octree dataset
    """
    pass


class SmartImageConditionedMixin(ImageConditionedMixin):
    def __init__(
        self,
        *args,
        predict_cam_token: bool = False,
        cond_jitter_prob: float = 0.0,
        control_image_dir: Optional[str] = None,
        **kwargs,
    ):
        """
        Smart image-conditioned GS octree dataset. Automatically rotate the latent to facing the condition image camera.
        Args:
            predict_cam_token: Whether to predict the camera token. MMDiT is trained to predict camera if set true.
            cond_jitter_prob: Probability of applying jittering to the condition image.
        """
        super().__init__(
            *args,
            **kwargs,
        )
        self.predict_cam_token = predict_cam_token
        self.cond_jitter_prob = cond_jitter_prob
        self.control_image_dir = control_image_dir
    
    def get_instance(self, root, instance):
        # We skip ImageConditionedMixin.get_instance to control the order and view selection
        # Call GSOctreeLatent.get_instance directly (via super of parent)
        
        image_root = os.path.join(root, 'renders_cond', instance)
        with open(os.path.join(image_root, 'transforms.json')) as f:
            metadata_full = json.load(f)
        n_views = len(metadata_full['frames'])
        view_idx = np.random.randint(n_views)
        metadata = metadata_full['frames'][view_idx]

        # Calculate rot_idx
        c2w = np.array(metadata['transform_matrix'])
        x, y, z = c2w[:3, 3]
        yaw = np.arctan2(y, x)
        yaw_deg = np.rad2deg(yaw)
        k = int(round(-yaw_deg / 90.0)) % 4

        # Call GSOctreeLatent.get_instance
        # We need to bypass ImageConditionedMixin.get_instance
        pack = super(ImageConditionedMixin, self).get_instance(root, instance, rot_idx=k)
        
        # Add a 4 dim camera key to x_0
        if self.predict_cam_token:
            # convert camera to a 5 dim latent vector
            params = {
                "camera_angle_x": metadata["camera_angle_x"],
                "camera_ortho_scale": metadata["camera_ortho_scale"],
                "camera_orthographic": metadata["camera_orthographic"], 
                "transform_matrix": torch.tensor(metadata["transform_matrix"])
            }
            latent_cam = encode_latent_camera(params, rot_idx=k)
            pack['x_0']['camera'] = latent_cam.unsqueeze(0) # 1 token length
        # Load and process image (copied from ImageConditionedMixin)
        image_path = os.path.join(image_root, metadata['file_path'])
        image = Image.open(image_path)
        image_raw = image.copy()

        if self.cond_jitter_prob > 0:
            image = jitter_image(image, self.cond_jitter_prob)

        def unified_crop(image: Union[Image.Image, torch.Tensor], bbox: Optional[List[int]] = None, aug_ratio: float = 1.2) -> Union[Tuple[Image.Image, List[int]], Tuple[torch.Tensor, List[int]]]:
            if bbox is None:
                try:
                    alpha = np.array(image.getchannel(3))
                    nonzero = np.array(alpha).nonzero()
                    if len(nonzero[0]) == 0:
                        bbox = [0, 0, image.width, image.height]
                    else:
                        bbox = [nonzero[1].min(), nonzero[0].min(), nonzero[1].max(), nonzero[0].max()]
                except ValueError:
                    raise ValueError("Image must have an alpha channel for unified cropping if bbox is None")
            
            center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
            hsize = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
            aug_hsize = hsize * aug_ratio
            aug_bbox = [int(center[0] - aug_hsize), int(center[1] - aug_hsize), int(center[0] + aug_hsize), int(center[1] + aug_hsize)]
            image = image.crop(aug_bbox)
            return image
        
        if self.use_crop_aug:
            image = unified_crop(image)

        def image2tensor(image, image_size: List[int]):
            image = image.resize((image_size[0], image_size[1]), Image.Resampling.LANCZOS)
            alpha = image.getchannel(3)
            image = image.convert('RGB')
            image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
            alpha = torch.tensor(np.array(alpha)).float() / 255.0
            image = image * alpha.unsqueeze(0)
            return image
        
        image = image2tensor(image, [self.image_size, self.image_size])
        image_raw = image2tensor(image_raw, [self.image_size, self.image_size])

        pack['cond'] = image
        pack['cond_raw'] = image_raw
        
        # Load Control Image if configured
        if self.control_image_dir is not None:
            ctrl_root = os.path.join(root, self.control_image_dir, instance)
            ctrl_meta_path = os.path.join(ctrl_root, 'transforms.json')
            if os.path.exists(ctrl_meta_path):
                with open(ctrl_meta_path) as f:
                    ctrl_meta = json.load(f)
                view_idx = np.random.randint(len(ctrl_meta['frames']))
                ctrl_frame = ctrl_meta['frames'][view_idx]
                ctrl_image_path = os.path.join(ctrl_root, ctrl_frame['file_path'])
                ctrl_image = Image.open(ctrl_image_path)
                ctrl_image = unified_crop(ctrl_image)
                ctrl_image = image2tensor(ctrl_image, [self.image_size, self.image_size])
                pack['ctrl_image'] = ctrl_image
            else:
                pack['ctrl_image'] = torch.zeros_like(image)
                
        return pack


class SmartImageConditionedGSOctreeLatent(SmartImageConditionedMixin, GSOctreeLatent):
    """
    Smart Image-conditioned GS octree dataset
    """
    pass
