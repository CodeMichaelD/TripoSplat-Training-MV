import torch
import numpy as np
from tqdm import tqdm
import utils3d
from ..renderers import GaussianRenderer
from ..representations import Gaussian
from .random_utils import sphere_hammersley_sequence

# Threshold for Ortho vs Perspective
ORTHO_THRESHOLD = 1e-3
# The camera distance used for approximating the orthographic projection with perspective camera
ORTHO_CAMERA_DIST = 1000.0


def load_image(path):
    from PIL import Image

    image = Image.open(path)
    if image.mode == 'RGBA':
        alpha = np.array(image.getchannel(3))
        bbox = np.array(alpha).nonzero()
        if len(bbox[0]) > 0:
            bbox = [bbox[1].min(), bbox[0].min(), bbox[1].max(), bbox[0].max()]
            aug_size_ratio = 1.2
        else:
            bbox = [0, 0, image.width, image.height]
            aug_size_ratio = 1.0
    else:
        bbox = [0, 0, image.width, image.height]
        aug_size_ratio = 1.0
    center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    hsize = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
    aug_hsize = hsize * aug_size_ratio
    aug_center = [center[0], center[1]]
    aug_bbox = [
        int(aug_center[0] - aug_hsize), int(aug_center[1] - aug_hsize),
        int(aug_center[0] + aug_hsize), int(aug_center[1] + aug_hsize),
    ]
    image = image.crop(aug_bbox)
    if image.mode == 'RGBA':
        bg = Image.new('RGB', image.size, (0, 0, 0))
        bg.paste(image, mask=image.split()[3])
        return bg
    return image.convert('RGB')


def load_condition_image_tensor(path, image_size=1024):
    from PIL import Image

    image = Image.open(path)
    if image.mode not in ('RGBA', 'LA'):
        image = image.convert('RGBA')

    alpha_arr = np.array(image.getchannel(3))
    nonzero = alpha_arr.nonzero()
    if len(nonzero[0]) > 0:
        bbox = [nonzero[1].min(), nonzero[0].min(), nonzero[1].max(), nonzero[0].max()]
        aug_ratio = 1.2
    else:
        bbox = [0, 0, image.width, image.height]
        aug_ratio = 1.0

    center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    hsize = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
    aug_hsize = hsize * aug_ratio
    aug_bbox = [
        int(center[0] - aug_hsize), int(center[1] - aug_hsize),
        int(center[0] + aug_hsize), int(center[1] + aug_hsize),
    ]
    image = image.crop(aug_bbox)
    image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)

    alpha = torch.tensor(np.array(image.getchannel(3))).float() / 255.0
    image = image.convert('RGB')
    image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
    return image * alpha.unsqueeze(0)

def yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, rs, fovs):
    is_list = isinstance(yaws, list)
    if not is_list:
        yaws = [yaws]
        pitchs = [pitchs]
    if not isinstance(rs, list):
        rs = [rs] * len(yaws)
    if not isinstance(fovs, list):
        fovs = [fovs] * len(yaws)
    extrinsics = []
    intrinsics = []
    for yaw, pitch, r, fov in zip(yaws, pitchs, rs, fovs):
        fov = torch.deg2rad(torch.tensor(float(fov))).cuda()
        yaw = torch.tensor(float(yaw)).cuda()
        pitch = torch.tensor(float(pitch)).cuda()
        orig = torch.tensor([
            torch.sin(yaw) * torch.cos(pitch),
            torch.cos(yaw) * torch.cos(pitch),
            torch.sin(pitch),
        ]).cuda() * r
        extr = utils3d.torch.extrinsics_look_at(orig, torch.tensor([0, 0, 0]).float().cuda(), torch.tensor([0, 0, 1]).float().cuda())
        intr = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
        extrinsics.append(extr)
        intrinsics.append(intr)
    if not is_list:
        extrinsics = extrinsics[0]
        intrinsics = intrinsics[0]
    return extrinsics, intrinsics


def get_renderer(sample, **kwargs):
    if isinstance(sample, Gaussian):
        renderer = GaussianRenderer()
        renderer.rendering_options.resolution = kwargs.get('resolution', 512)
        renderer.rendering_options.near = kwargs.get('near', 0.8)
        renderer.rendering_options.far = kwargs.get('far', 1.6)
        renderer.rendering_options.bg_color = kwargs.get('bg_color', (0, 0, 0))
        renderer.rendering_options.ssaa = kwargs.get('ssaa', 1)
        renderer.pipe.kernel_size = kwargs.get('kernel_size', 0.1)
        renderer.pipe.use_mip_gaussian = True
    else:
        raise ValueError(f'Unsupported sample type: {type(sample)}')
    return renderer


def render_frames(sample, extrinsics, intrinsics, options={}, colors_overwrite=None, verbose=True, **kwargs):
    renderer = get_renderer(sample, **options)
    rets = {}
    for j, (extr, intr) in tqdm(enumerate(zip(extrinsics, intrinsics)), desc='Rendering', disable=not verbose):
        res = renderer.render(sample, extr, intr, colors_overwrite=colors_overwrite)
        if 'color' not in rets: rets['color'] = []
        if 'depth' not in rets: rets['depth'] = []
        rets['color'].append(np.clip(res['color'].detach().cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8))
        if 'percent_depth' in res:
            rets['depth'].append(res['percent_depth'].detach().cpu().numpy())
        elif 'depth' in res:
            rets['depth'].append(res['depth'].detach().cpu().numpy())
        else:
            rets['depth'].append(None)
    return rets


def render_video(sample, resolution=512, bg_color=(0, 0, 0), num_frames=300, r=2, fov=40, yaws_base=0.0, pitch_base=0.25, pitch_rate=0.5, **kwargs):
    yaws = torch.linspace(0, 2 * 3.1415, num_frames) + yaws_base
    pitch = pitch_base + pitch_rate * torch.sin(torch.linspace(0, 2 * 3.1415, num_frames))
    pitch = pitch.clamp(-0.25, 1.0)
    yaws = yaws.tolist()
    pitch = pitch.tolist()
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitch, r, fov)
    return render_frames(sample, extrinsics, intrinsics, {'resolution': resolution, 'bg_color': bg_color}, **kwargs)


def render_multiview(sample, resolution=512, nviews=30):
    r = 2
    fov = 40
    cams = [sphere_hammersley_sequence(i, nviews) for i in range(nviews)]
    yaws = [cam[0] for cam in cams]
    pitchs = [cam[1] for cam in cams]
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, r, fov)
    res = render_frames(sample, extrinsics, intrinsics, {'resolution': resolution, 'bg_color': (0, 0, 0)})
    return res['color'], extrinsics, intrinsics


def render_snapshot(samples, resolution=512, bg_color=(0, 0, 0), offset=(-16 / 180 * np.pi, 20 / 180 * np.pi), r=10, fov=8, **kwargs):
    yaw = [0, np.pi/2, np.pi, 3*np.pi/2]
    yaw_offset = offset[0]
    yaw = [y + yaw_offset for y in yaw]
    pitch = [offset[1] for _ in range(4)]
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaw, pitch, r, fov)
    return render_frames(samples, extrinsics, intrinsics, {'resolution': resolution, 'bg_color': bg_color}, **kwargs)


def encode_latent_camera(camera_params, rot_idx=0):
    """
    Encodes camera parameters to a unified 5D latent vector.
    
    Representation: [dx, dy, dz, k, w]
    - [dx, dy, dz]: Direction vector (normalized) in World Coordinate System.
                    Assumes Z-up World Coordinate System.
                    Direction points from World Origin to Camera Center.
    - k: Inverse Distance (0 = Ortho, >0 = Perspective).
    - w: Frustum Width at Origin.
    
    Args:
        camera_params (dict): Dictionary containing camera parameters.
            Must contain 'transform_matrix' (4x4) or 'position' (3,).
            Optional: 'camera_angle_x' (fov), 'camera_orthographic', 'camera_ortho_scale'.
            
    Returns:
        torch.Tensor: Latent vector of shape (5,).
                    Values are roughly in [-1, 1] (except direction is unit vector).
    """
    # Extract position
    if 'transform_matrix' in camera_params:
        c2w = camera_params['transform_matrix']
        pos = c2w[:3, 3]
    elif 'position' in camera_params:
        pos = camera_params['position']
    else:
        raise ValueError("Camera parameters must contain 'transform_matrix' or 'position'.")
        
    # Calculate Direction and Distance
    dist = torch.norm(pos)
    direction = pos / dist
    
    if rot_idx !=0:
        # rotate 90
        rot_idx = rot_idx % 4
        for _ in range(rot_idx):
            # rotate on z axis
            direction = torch.tensor([-direction[1], direction[0], direction[2]], device=pos.device, dtype=pos.dtype)
        
    # Check projection type
    is_ortho = camera_params.get('camera_orthographic', False)
    
    if is_ortho:
        # Orthographic
        k = torch.tensor(0.0, device=pos.device, dtype=pos.dtype)
        scale = camera_params.get('camera_ortho_scale')

        
        w = torch.tensor(scale / 2.0, device=pos.device, dtype=pos.dtype)
    else:
        # Perspective
        k = 1.0 / dist
        fov = camera_params.get('camera_angle_x')
        fov = torch.tensor(fov, device=pos.device, dtype=pos.dtype)
        w = dist * torch.tan(fov / 2.0)
        
    return torch.cat([direction, k.view(1), w.view(1)])

def decode_latent_camera(latent_vector):
    """
    Decodes a unified 5D latent vector to camera parameters.
    
    Args:
        latent_vector (torch.Tensor): Shape (5,). [dx, dy, dz, k_norm, w_norm]
        
    Returns:
        dict: Camera parameters dict.
    """
    if len(latent_vector) != 5:
        raise ValueError(f"Latent vector must have 5 elements, got {len(latent_vector)}")
        
    direction_raw = latent_vector[:3]
    k = latent_vector[3]
    w = latent_vector[4]
    
    # Normalize direction (ensure unit length)
    d_norm = torch.norm(direction_raw)
    if d_norm < 1e-6:
        direction = torch.tensor([0, 0, 1.0], device=latent_vector.device, dtype=latent_vector.dtype)
    else:
        direction = direction_raw / d_norm
    
    # Threshold for Ortho vs Perspective
    # If k is very small, treat as Ortho.
    
    if k < ORTHO_THRESHOLD:
        is_ortho = True
        scale = w * 2.0
        fov = 2.0 * torch.atan(w / ORTHO_CAMERA_DIST)
        pos = direction * ORTHO_CAMERA_DIST
        
    else:
        is_ortho = False
        scale = None
        dist = 1.0 / k
        pos = direction * dist
        fov = 2.0 * torch.atan(w * k)
        
    # Reconstruct Rotation (LookAt Origin)
    # Use utils3d for consistency (OpenCV convention: Right, Down, Forward)
    origin = torch.zeros_like(pos)
    up = torch.tensor([0, 0, 1.0], device=latent_vector.device, dtype=latent_vector.dtype)
    extrinsics = utils3d.torch.extrinsics_look_at(pos, origin, up)
    # transform_matrix = torch.inverse(extrinsics)

    if is_ortho:
        intrinsics = torch.eye(3, device=latent_vector.device, dtype=latent_vector.dtype)
        intrinsics[0, 0] = 0.5 * ORTHO_CAMERA_DIST / w
        intrinsics[1, 1] = 0.5 * ORTHO_CAMERA_DIST / w
        intrinsics[0, 2] = 0.5
        intrinsics[1, 2] = 0.5
    else:
        intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
    
    return {
        'extrinsics': extrinsics,
        'intrinsics': intrinsics,
        # 'transform_matrix': transform_matrix,
        'camera_orthographic': is_ortho,
        'camera_angle_x': fov.item() if isinstance(fov, torch.Tensor) else fov,
        'camera_ortho_scale': scale.item() if isinstance(scale, torch.Tensor) else scale
    }


def decode_latent_camera_params(latent_vector):
    """
    Decodes a unified 5D latent vector to camera parameters (yaw_base, fov, radius, pitch).
    
    Args:
        latent_vector (torch.Tensor): Shape (5,). [dx, dy, dz, k, w]
        
    Returns:
        tuple: (yaw_base, fov, radius, pitch)
            - yaw_base: Yaws value in range [-0.5, 0.5].
            - fov: Field of view in degrees.
            - radius: Distance from origin.
            - pitch: Pitch value in radians.
    """
    if len(latent_vector) != 5:
        raise ValueError(f"Latent vector must have 5 elements, got {len(latent_vector)}")
        
    direction_raw = latent_vector[:3]
    k = latent_vector[3]
    w = latent_vector[4]
    
    # Normalize direction (ensure unit length)
    d_norm = torch.norm(direction_raw)
    if d_norm < 1e-6:
        direction = torch.tensor([0, 0, 1.0], device=latent_vector.device, dtype=latent_vector.dtype)
    else:
        direction = direction_raw / d_norm
        
    dx, dy, dz = direction[0], direction[1], direction[2]
    
    # Consistent with decode_yaws_base
    yaw_base = torch.atan2(dx, dy)
    
    # Pitch from direction
    pitch_base = torch.asin(torch.clamp(dz, -1.0, 1.0))
    
    
    if k < ORTHO_THRESHOLD:
        radius = torch.tensor(ORTHO_CAMERA_DIST, device=latent_vector.device, dtype=latent_vector.dtype)
        fov = torch.rad2deg(2.0 * torch.atan(w / ORTHO_CAMERA_DIST))
    else:
        radius = 1.0 / k
        fov = torch.rad2deg(2.0 * torch.atan(w * k))
        
    return yaw_base, fov, radius, pitch_base
