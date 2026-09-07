import os, sys, json, argparse, glob
import numpy as np
import torch
import trimesh
import pyrender
import cv2
from PIL import Image

def load_mesh(file_path, unit_radius=False):
    mesh = trimesh.load(file_path)
    if not isinstance(mesh, trimesh.Trimesh):
        # Handle scenes or multiple geometries
        mesh = trimesh.util.concatenate([g for g in mesh.geometry.values()])
    # Normalize: center and scale to unit cube (or unit radius)
    if unit_radius:
        bbox = mesh.bounds
        center = (bbox[0] + bbox[1]) / 2
        mesh.vertices -= center
        radius = np.linalg.norm(mesh.vertices, axis=1).max()
        mesh.vertices /= radius
    else:
        # Scale to fit in a unit cube (0..1) as original Blender script did
        bbox = mesh.bounds
        scale = 1.0 / (bbox[1] - bbox[0]).max()
        mesh.vertices = (mesh.vertices - bbox[0]) * scale
        mesh.vertices -= 0.5   # center around origin
    return mesh

def render_view(mesh, yaw, pitch, radius, fov, resolution, ortho=False, ortho_scale=None):
    # Create scene
    scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3])
    # Add mesh with a simple material
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.8, 0.8, 0.8, 1.0],
        roughnessFactor=0.5,
        metallicFactor=0.0
    )
    # If the mesh has vertex colors or textures, we can use them, but for simplicity we use a grey material
    # For better quality, we can use the original mesh colors if available
    if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
        # Use vertex colors
        mesh = mesh.copy()
        mesh.visual.vertex_colors = mesh.visual.vertex_colors
        # pyrender doesn't directly support vertex colors, so we need to convert to texture
        # For simplicity, we'll just use a flat grey
        mesh.visual.vertex_colors = None
    # Add mesh
    mesh_py = pyrender.Mesh.from_trimesh(mesh, material=material)
    scene.add(mesh_py)

    # Camera
    cam_pose = np.eye(4)
    # Position: spherical coordinates
    x = radius * np.cos(yaw) * np.cos(pitch)
    y = radius * np.sin(yaw) * np.cos(pitch)
    z = radius * np.sin(pitch)
    cam_pose[:3, 3] = [x, y, z]
    # Look at origin (0,0,0)
    up = np.array([0, 0, 1])
    cam_pose[:3, :3] = trimesh.geometry.align_vectors([0, 0, -1], -cam_pose[:3, 3] / np.linalg.norm(cam_pose[:3, 3]), [0, 1, 0], up)
    if ortho:
        camera = pyrender.OrthographicCamera(xmag=ortho_scale, ymag=ortho_scale)
    else:
        camera = pyrender.PerspectiveCamera(yfov=fov, aspectRatio=1.0)
    scene.add(camera, pose=cam_pose)

    # Lighting
    # Add a directional light from the top-front
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    scene.add(light, pose=cam_pose)  # same direction as camera for simple lighting
    # Add a fill light from behind
    light2 = pyrender.DirectionalLight(color=[0.5, 0.5, 0.5], intensity=1.0)
    scene.add(light2, pose=np.eye(4))  # from world origin

    # Render
    r = pyrender.OffscreenRenderer(resolution, resolution)
    color, depth = r.render(scene, flags=pyrender.RenderFlags.RGBA)
    # color is RGBA, depth is float
    return color, depth, cam_pose

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--views', type=str, required=True)
    parser.add_argument('--object', type=str, required=True)
    parser.add_argument('--output_folder', type=str, default='/tmp')
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--engine', type=str, default='BLENDER_EEVEE')  # ignored
    parser.add_argument('--geo_mode', action='store_true')  # ignored
    parser.add_argument('--save_depth', action='store_true')
    parser.add_argument('--save_normal', action='store_true')  # not supported
    parser.add_argument('--save_albedo', action='store_true')  # not supported
    parser.add_argument('--save_mist', action='store_true')    # not supported
    parser.add_argument('--split_normal', action='store_true') # not supported
    parser.add_argument('--save_mesh', action='store_true')
    parser.add_argument('--unit_radius', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    os.makedirs(os.path.join(args.output_folder, 'image'), exist_ok=True)
    if args.save_depth:
        os.makedirs(os.path.join(args.output_folder, 'depth'), exist_ok=True)

    # Load mesh
    mesh = load_mesh(args.object, unit_radius=args.unit_radius)

    views = json.loads(args.views)
    to_export = {
        "unit_radius": args.unit_radius,
        "aabb": [(-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)],  # approximate
        "scale": 1.0,
        "offset": (0,0,0),
        "frames": []
    }
    if args.save_depth:
        to_export["depth_frames"] = []

    for i, view in enumerate(views):
        fov = view.get('fov', 0.8)  # default if not provided
        yaw = view['yaw']
        pitch = view['pitch']
        radius = view['radius']
        ortho = view.get('ortho', False)
        ortho_scale = view.get('ortho_scale', None)

        color, depth, cam_pose = render_view(
            mesh, yaw, pitch, radius, fov, args.resolution,
            ortho=ortho, ortho_scale=ortho_scale
        )
        # Save image
        img = Image.fromarray(color, 'RGBA')
        img_path = os.path.join(args.output_folder, 'image', f'{i:03d}.webp')
        img.save(img_path, 'WEBP', quality=100)

        # Save depth
        if args.save_depth:
            # depth is in meters; we need to normalize to 0-1 for the pipeline
            # The original used camera distance ± sqrt(3)/2, we can compute min/max from depth
            valid = depth > 0
            if valid.sum() > 0:
                d_min = depth[valid].min()
                d_max = depth[valid].max()
                depth_norm = (depth - d_min) / (d_max - d_min + 1e-8)
                depth_norm = (depth_norm * 65535).astype(np.uint16)
                cv2.imwrite(os.path.join(args.output_folder, 'depth', f'{i:03d}.png'), depth_norm)
                # store min/max in metadata
                depth_min = float(d_min)
                depth_max = float(d_max)
            else:
                depth_min = 0.0
                depth_max = 1.0
        else:
            depth_min = depth_max = None

        # Build frame metadata (matching Blender format)
        frame = {
            "file_path": f'image/{i:03d}.webp',
            "camera_angle_x": fov,
            "camera_ortho_scale": ortho_scale,
            "camera_orthographic": ortho,
            "transform_matrix": cam_pose.T.tolist()  # opencv convention? Blender uses 4x4 matrix
        }
        to_export["frames"].append(frame)
        if args.save_depth:
            to_export["depth_frames"].append({
                "file_path": f'depth/{i:03d}.png',
                "depth": {"min": depth_min, "max": depth_max},
                "camera_angle_x": fov,
                "camera_ortho_scale": ortho_scale,
                "camera_orthographic": ortho,
                "transform_matrix": cam_pose.T.tolist()
            })

    # Save transforms.json
    with open(os.path.join(args.output_folder, 'transforms.json'), 'w') as f:
        json.dump(to_export, f, indent=4)

    # Save mesh if requested
    if args.save_mesh:
        mesh.export(os.path.join(args.output_folder, 'mesh.ply'))

if __name__ == '__main__':
    main()
