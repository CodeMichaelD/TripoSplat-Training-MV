#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from deg.utils.general_utils import edict
import numpy as np
from ..representations.gaussian import Gaussian
from .sh_utils import eval_sh
import torch.nn.functional as F


def intrinsics_to_projection(
        intrinsics: torch.Tensor,
        near: float,
        far: float,
    ) -> torch.Tensor:
    """
    OpenCV intrinsics to OpenGL perspective matrix

    Args:
        intrinsics (torch.Tensor): [3, 3] OpenCV intrinsics matrix
        near (float): near plane to clip
        far (float): far plane to clip
    Returns:
        (torch.Tensor): [4, 4] OpenGL perspective matrix
    """
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    ret = torch.zeros((4, 4), dtype=intrinsics.dtype, device=intrinsics.device)
    ret[0, 0] = 2 * fx
    ret[1, 1] = 2 * fy
    ret[0, 2] = 2 * cx - 1
    ret[1, 2] = - 2 * cy + 1
    ret[2, 2] = far / (far - near)
    ret[2, 3] = near * far / (near - far)
    ret[3, 2] = 1.
    return ret

def render(viewpoint_camera, pc: Gaussian, pipe, bg_color: torch.Tensor,
           scaling_modifier=1.0, override_color=None,
           return_opacity: bool = False, return_depth: bool = False,
           gt_image=None, l1_token=None
           ):   # NEW
    """
    Render the scene. 
    Background tensor (bg_color) must be on GPU!
    """
    # lazy import
    if 'GaussianRasterizer' not in globals():
        from diff_gaussian_rasterization import GaussianRasterizer, GaussianRasterizationSettings
    
    # screen-space grads target
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # intrinsics
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    kernel_size = pipe.kernel_size
    subpixel_offset = torch.zeros((int(viewpoint_camera.image_height), int(viewpoint_camera.image_width), 2),
                                  dtype=torch.float32, device="cuda")

    # main (RGB) pass settings
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        kernel_size=kernel_size,
        subpixel_offset=subpixel_offset,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # covariance handling
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # color / SH handling
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # ---- Pass 1: normal RGB render ----
    if gt_image is not None and l1_token is not None:
        rendered_image, radii, l1_map = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
            gt_image=gt_image,
            l1_token=l1_token,
        )
    else:
        rendered_image, radii = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp
        )

    out = edict({
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
    })
    if gt_image is not None and l1_token is not None:
        out["l1_map"] = l1_map

    # ---- Pass 2: packed alpha/depth on RGB channels (optional) ----
    if return_opacity or return_depth:
        # black background + use precomputed colors (no SH)             # NEW
        bg_color_depth = torch.tensor([0, 2 + 0.5 * 3**0.5, 0]).to(bg_color)
        opacity_depth_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            kernel_size=kernel_size,
            subpixel_offset=subpixel_offset,
            bg=bg_color_depth,   # BLACK background
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=0,                     # force colors_precomp
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=pipe.debug
        )
        rasterizer_aux = GaussianRasterizer(raster_settings=opacity_depth_settings)

        N = means3D.shape[0]
        onesN = torch.ones(N, 1, device=means3D.device)

        # camera-space z per Gaussian (from world->view transform)      # NEW
        xyz1 = torch.cat([means3D, onesN], dim=1)                      # [N,4]
        cam = (xyz1 @ viewpoint_camera.world_view_transform)         # [N,4]
        z_cam = cam[:, 2:3]                                            # [N,1]

        # colors: [1, z, 0] so R=Σw_i (alpha), G=Σz_i w_i (depth num)   # NEW
        carriers = torch.cat([onesN, z_cam, torch.zeros_like(onesN)], dim=1)

        packed_rgb, _ = rasterizer_aux(
            means3D=means3D,
            means2D=means2D,              # keep for correct grads
            shs=None,
            colors_precomp=carriers,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp
        )

        # opacity from R, expected depth from G / R                      # NEW
        alpha_img = packed_rgb[0, ...].clamp(0, 1)
        if return_opacity:
            out["opacity"] = alpha_img

        if return_depth:
            depth_num = packed_rgb[1, ...]
            out["depth"] = depth_num
    return out


class GaussianRenderer:
    """
    Renderer for the Voxel representation.

    Args:
        rendering_options (dict): Rendering options.
    """

    def __init__(self, rendering_options={}) -> None:
        self.pipe = edict({
            "kernel_size": 0.1,
            "convert_SHs_python": False,
            "compute_cov3D_python": False,
            "scale_modifier": 1.0,
            "debug": False
        })
        self.rendering_options = edict({
            "resolution": None,
            "near": None,
            "far": None,
            "ssaa": 1,
            "bg_color": 'random',
        })
        self.rendering_options.update(rendering_options)
        self.bg_color = None
    
    def render(
            self,
            gausssian: Gaussian,
            extrinsics: torch.Tensor,
            intrinsics: torch.Tensor,
            colors_overwrite: torch.Tensor = None,
            return_opacity: bool = False,
            return_depth: bool = False,
            gt_image=None, 
            gt_alpha=None,
            l1_token=None, 
        ) -> edict:
        """
        Render the gausssian.

        Args:
            gaussian : gaussianmodule
            extrinsics (torch.Tensor): (4, 4) camera extrinsics
            intrinsics (torch.Tensor): (3, 3) camera intrinsics
            colors_overwrite (torch.Tensor): (N, 3) override color
            
            gt_image (torch.Tensor): (3, H, W) ground truth image (for l1 contribution)
            gt_alpha (torch.Tensor): (H, W) ground truth alpha mask (for l1 contribution)
            l1_token (torch.Tensor): (N_gs) l1 token (for l1 contribution)

        Returns:
            edict containing:
                color (torch.Tensor): (3, H, W) rendered color image
        """
        resolution = self.rendering_options["resolution"]
        near = self.rendering_options["near"]
        far = self.rendering_options["far"]
        ssaa = self.rendering_options["ssaa"]
        
        if self.rendering_options["bg_color"] == 'random':
            self.bg_color = torch.zeros(3, dtype=torch.float32, device="cuda")
            if np.random.rand() < 0.5:
                self.bg_color += 1
        else:
            self.bg_color = torch.tensor(self.rendering_options["bg_color"], dtype=torch.float32, device="cuda")

        if gt_image is not None and gt_alpha is not None:
            if ssaa > 1:
                gt_image = F.interpolate(gt_image[None], size=(resolution * ssaa, resolution * ssaa), mode='bilinear', align_corners=False, antialias=True)[0]
                gt_alpha = F.interpolate(gt_alpha[None, None], size=(resolution * ssaa, resolution * ssaa), mode='bilinear', align_corners=False, antialias=True)[0, 0]
            gt_image = gt_image * gt_alpha[None, ...] + (1 - gt_alpha[None, ...]) * self.bg_color[:, None, None]

        view = extrinsics
        perspective = intrinsics_to_projection(intrinsics, near, far)
        camera = torch.inverse(view)[:3, 3]
        focalx = intrinsics[0, 0]
        focaly = intrinsics[1, 1]
        fovx = 2 * torch.atan(0.5 / focalx)
        fovy = 2 * torch.atan(0.5 / focaly)
            
        camera_dict = edict({
            "image_height": resolution * ssaa,
            "image_width": resolution * ssaa,
            "FoVx": fovx,
            "FoVy": fovy,
            "znear": near,
            "zfar": far,
            "world_view_transform": view.T.contiguous(),
            "projection_matrix": perspective.T.contiguous(),
            "full_proj_transform": (perspective @ view).T.contiguous(),
            "camera_center": camera
        })

        # Render
        render_ret = render(
            camera_dict, 
            gausssian, 
            self.pipe, 
            self.bg_color, 
            override_color=colors_overwrite, 
            scaling_modifier=self.pipe.scale_modifier, 
            gt_image=gt_image,
            l1_token=l1_token,
            return_opacity=return_opacity, 
            return_depth=return_depth,
        )

        if ssaa > 1:
            render_ret.render = F.interpolate(render_ret.render[None], size=(resolution, resolution), mode='bilinear', align_corners=False, antialias=True).squeeze()
            if 'opacity' in render_ret:
                render_ret.opacity = F.interpolate(render_ret.opacity[None, None], size=(resolution, resolution), mode='bilinear', align_corners=False, antialias=True).squeeze()
            if 'depth' in render_ret:
                render_ret.depth = F.interpolate(render_ret.depth[None, None], size=(resolution, resolution), mode='bilinear', align_corners=False, antialias=True).squeeze()
            
        ret = edict({
            'color': render_ret['render']
        })
        if 'opacity' in render_ret:
            ret['opacity'] = render_ret['opacity']
        if 'depth' in render_ret:
            ret['depth'] = render_ret['depth']
        if 'l1_map' in render_ret:
            ret['l1_map'] = render_ret['l1_map']
        return ret
