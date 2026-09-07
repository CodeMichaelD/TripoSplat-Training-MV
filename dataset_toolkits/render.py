import os
import sys
import json
import argparse
import numpy as np
import trimesh
import pyrender
import cv2
from PIL import Image
from math import cos, sin, pi

# ----------------------------------------------------------------------
# Helpers to build camera pose from yaw/pitch/radius
# ----------------------------------------------------------------------
def look_at(eye, target, up):
    """
    Return a 4x4 world‑to‑camera matrix (OpenGL convention).
    """
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    new_up = np.cross(right, forward)
    new_up = new_up / np.linalg.norm(new_up)
    # Rotation matrix (camera axes in world coords)
    rot = np.eye(4)
    rot[:3, 0] = right
    rot[:3, 1] = new_up
    rot[:3, 2] = -forward
    # Translation
    trans = np.eye(4)
    trans[:3, 3] = -eye
    return rot @ trans

def yaw_pitch_to_pose(yaw, pitch, radius):
    """
    Return camera‑to‑world matrix (the transform matrix expected in transforms.json).
    """
    eye = np.array([
        radius * cos(yaw) * cos(pitch),
        radius * sin(yaw) * cos(pitch),
        radius * sin(pitch)
    ])
    target = np.zeros(3)
    up = np.array([0, 0, 1])   # Z‑up
    view = look_at(eye, target, up)
    # camera‑to‑world = inverse(view)
    pose = np.linalg.inv(view)
    return pose

# ----------------------------------------------------------------------
# Mesh loading & normalisation
# ----------------------------------------------------------------------
def load_mesh(file_path, unit_radius):
    mesh = trimesh.load(file_path)
    if not isinstance(mesh, trimesh.Trimesh):
        # If it's a scene, concatenate all geometries
        meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError("No mesh found in file")
        mesh = trimesh.util.concatenate(meshes)

    if unit_radius:
        bbox = mesh.bounds
        center = (bbox[0] + bbox[1]) / 2
        mesh.vertices -= center
        radius = np.linalg.norm(mesh.vertices, axis=1).max()
        if radius > 1e-6:
            mesh.vertices /= radius
    else:
        # Scale to fit in a unit cube (same as Blender script)
        bbox = mesh.bounds
        scale = 1.0 / (bbox[1] - bbox[0]).max()
        mesh.vertices = (mesh.vertices - bbox[0]) * scale
        mesh.vertices -= 0.5   # centre

    return mesh

# ----------------------------------------------------------------------
# Render a single view
# ----------------------------------------------------------------------
def render_view(mesh, pose, fov, resolution, ortho=False, ortho_scale=None):
    scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3], bg_color=[0,0,0,0])

    # Mesh material (diffuse grey, no specular)
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.8, 0.8, 0.8, 1.0],
        roughnessFactor=0.8,
        metallicFactor=0.0,
        alphaMode='OPAQUE'
    )
    # If the mesh has vertex colors, we could try to use them, but for simplicity we keep grey.
    # For better visual quality, we can add a simple texture or use the original material if present.
    py_mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
    scene.add(py_mesh)

    # Camera
    if ortho:
        if ortho_scale is None:
            ortho_scale = 1.0
        camera = pyrender.OrthographicCamera(xmag=ortho_scale, ymag=ortho_scale)
    else:
        # fov is in radians (horizontal)
        camera = pyrender.PerspectiveCamera(yfov=fov, aspectRatio=1.0)

    scene.add(camera, pose=pose)

    # Lights – mimic Blender's setup: key, top, bottom
    # Key light (front‑top‑right)
    key_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    key_pose = np.eye(4)
    key_pose[:3, 3] = [4, 1, 6]
    scene.add(key_light, pose=key_pose)

    # Top light
    top_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    top_pose = np.eye(4)
    top_pose[:3, 3] = [0, 0, 10]
    scene.add(top_light, pose=top_pose)

    # Bottom light (fill)
    bottom_light = pyrender.DirectionalLight(color=[0.5, 0.5, 0.5], intensity=1.0)
    bottom_pose = np.eye(4)
    bottom_pose[:3, 3] = [0, 0, -10]
    scene.add(bottom_light, pose=bottom_pose)

    # Render
    r = pyrender.OffscreenRenderer(resolution, resolution)
    color, depth = r.render(scene, flags=pyrender.RenderFlags.RGBA)
    r.delete()

    return color, depth

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(args):
    os.makedirs(args.output_folder, exist_ok=True)
    os.makedirs(os.path.join(args.output_folder, 'image'), exist_ok=True)
    if args.save_depth:
        os.makedirs(os.path.join(args.output_folder, 'depth'), exist_ok=True)

    # Load and normalise mesh
    try:
        mesh = load_mesh(args.object, args.unit_radius)
    except Exception as e:
        print(f"Error loading mesh {args.object}: {e}")
        sys.exit(1)

    # Parse views
    views = json.loads(args.views)

    # Prepare output metadata
    to_export = {
        "unit_radius": args.unit_radius,
        "aabb": [(-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)],  # approximate
        "scale": 1.0,
        "offset": [0.0, 0.0, 0.0],
        "frames": []
    }
    if args.save_depth:
        to_export["depth_frames"] = []

    for i, view in enumerate(views):
        yaw = view['yaw']
        pitch = view['pitch']
        radius = view['radius']
        fov = view.get('fov', 0.8)   # default if missing
        ortho = view.get('ortho', False)
        ortho_scale = view.get('ortho_scale', None)

        # Build camera pose (camera‑to‑world)
        pose = yaw_pitch_to_pose(yaw, pitch, radius)

        # Render
        color, depth = render_view(mesh, pose, fov, args.resolution,
                                   ortho=ortho, ortho_scale=ortho_scale)

        # Save image (RGBA)
        img = Image.fromarray(color, 'RGBA')
        img_path = os.path.join(args.output_folder, 'image', f'{i:03d}.webp')
        img.save(img_path, 'WEBP', quality=100, method=6)

        # Save depth if requested
        if args.save_depth:
            # Depth is in world units; we want to store min/max for later
            valid = depth > 0
            if valid.sum() > 0:
                d_min = float(depth[valid].min())
                d_max = float(depth[valid].max())
                # Normalise to 0‑1 for 16‑bit PNG
                depth_norm = (depth - d_min) / (d_max - d_min + 1e-8)
                depth_norm = (depth_norm * 65535).astype(np.uint16)
                cv2.imwrite(os.path.join(args.output_folder, 'depth', f'{i:03d}.png'), depth_norm)
            else:
                d_min = 0.0
                d_max = 1.0
                # Write a black depth map
                depth_black = np.zeros((args.resolution, args.resolution), dtype=np.uint16)
                cv2.imwrite(os.path.join(args.output_folder, 'depth', f'{i:03d}.png'), depth_black)
        else:
            d_min = d_max = None

        # Build frame metadata (same as Blender output)
        frame = {
            "file_path": f'image/{i:03d}.webp',
            "camera_angle_x": fov,
            "camera_ortho_scale": ortho_scale,
            "camera_orthographic": ortho,
            "transform_matrix": pose.tolist()   # camera‑to‑world 4x4
        }
        to_export["frames"].append(frame)

        if args.save_depth:
            depth_frame = {
                "file_path": f'depth/{i:03d}.png',
                "depth": {"min": d_min, "max": d_max},
                "camera_angle_x": fov,
                "camera_ortho_scale": ortho_scale,
                "camera_orthographic": ortho,
                "transform_matrix": pose.tolist()
            }
            to_export["depth_frames"].append(depth_frame)

    # Write transforms.json
    with open(os.path.join(args.output_folder, 'transforms.json'), 'w') as f:
        json.dump(to_export, f, indent=4)

    # Save mesh as PLY if requested
    if args.save_mesh:
        # We need to re‑load the mesh without normalisation? Actually we want the same mesh as used for rendering.
        # We'll just export the normalized mesh.
        mesh.export(os.path.join(args.output_folder, 'mesh.ply'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OpenGL renderer for 3D models (replaces Blender).')
    parser.add_argument('--views', type=str, required=True,
                        help='JSON string of views: list of {yaw, pitch, radius, fov, ortho, ortho_scale}')
    parser.add_argument('--object', type=str, required=True,
                        help='Path to the 3D model file (supported by trimesh).')
    parser.add_argument('--output_folder', type=str, default='/tmp',
                        help='Output directory.')
    parser.add_argument('--resolution', type=int, default=512,
                        help='Rendering resolution (square).')
    parser.add_argument('--engine', type=str, default='BLENDER_EEVEE',
                        help='Ignored (kept for compatibility).')
    parser.add_argument('--geo_mode', action='store_true',
                        help='Ignored (kept for compatibility).')
    parser.add_argument('--save_depth', action='store_true',
                        help='Save depth maps as 16‑bit PNG.')
    parser.add_argument('--save_normal', action='store_true',
                        help='Not supported (ignored).')
    parser.add_argument('--save_albedo', action='store_true',
                        help='Not supported (ignored).')
    parser.add_argument('--save_mist', action='store_true',
                        help='Not supported (ignored).')
    parser.add_argument('--split_normal', action='store_true',
                        help='Not supported (ignored).')
    parser.add_argument('--save_mesh', action='store_true',
                        help='Save the mesh as PLY.')
    parser.add_argument('--unit_radius', action='store_true',
                        help='Normalise to unit radius (instead of unit cube).')

    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    args = parser.parse_args(argv)
    main(args)
