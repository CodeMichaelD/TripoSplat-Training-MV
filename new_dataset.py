import os
import json
import shutil
import subprocess
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import hashlib
from huggingface_hub import HfApi, create_repo

# ==========================================
# 0. ENVIRONMENT SETUP
# ==========================================
os.environ["ATTN_BACKEND"] = "sdpa"
os.environ["PYOPENGL_PLATFORM"] = "egl"

# ==========================================
# 1. CONFIGURATION
# ==========================================
KAGGLE_INPUT_DIR = "/kaggle/input/datasets/codemichaeld/new-data"
MESH_DIR = os.path.join(KAGGLE_INPUT_DIR, "meshes")
CTRL_IMG_DIR = os.path.join(KAGGLE_INPUT_DIR, "ctrl_images")

OUT_DIR = "/kaggle/working/triposplat_dataset"
REPO_DIR = "/kaggle/working/TripoSplat-Training-MV"

from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
HF_TOKEN = user_secrets.get_secret("HF_TOKEN")
# IMPORTANT: export token to environment so subprocesses can use it
os.environ["HF_TOKEN"] = HF_TOKEN

HF_REPO_ID = "codemichaeld/triposplat-control-dataset"
NUM_VIEWS = 150

# ==========================================
os.chdir(REPO_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

# Install dependencies
print("Installing OpenGL headless renderer (pyrender)...")
subprocess.run(["pip", "uninstall", "-y", "PyOpenGL_accelerate"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
subprocess.run(["pip", "install", "pyrender", "PyOpenGL"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print("Installing POT (Python Optimal Transport)...")
subprocess.run(["pip", "install", "POT"], check=True)

import pyrender
import trimesh
import OpenGL.GL as gl
import ctypes
from dataset_toolkits.utils import sphere_hammersley_sequence

# ==========================================
# MONKEY-PATCH PyOpenGL for Python 3.12
# ==========================================
def _patch_gl_gen(func_name):
    orig = getattr(gl, func_name)
    def patched(*args):
        if len(args) == 1:
            n = args[0]
            arr = (gl.GLuint * n)()
            if hasattr(orig, 'baseFunction'):
                orig.baseFunction(n, arr)
            else:
                orig(n, arr)
            return arr[0] if n == 1 else arr
        elif len(args) == 2:
            n, arr = args
            if hasattr(orig, 'baseFunction'):
                orig.baseFunction(n, arr)
            else:
                orig(n, arr)
        else:
            raise TypeError(f"{func_name}() takes 1 or 2 positional arguments but {len(args)} were given")
    return patched

def _patch_gl_delete(func_name):
    orig = getattr(gl, func_name)
    def patched(*args):
        if len(args) == 1:
            ids = args[0]
            if isinstance(ids, int):
                arr = (gl.GLuint * 1)(ids)
                n = 1
            else:
                arr = ids
                n = len(arr)
        elif len(args) == 2:
            n, ids = args
            if isinstance(ids, int):
                arr = (gl.GLuint * 1)(ids)
                n = 1
            else:
                arr = ids
        else:
            raise TypeError(f"{func_name}() takes 1 or 2 positional arguments but {len(args)} were given")
        if hasattr(orig, 'baseFunction'):
            orig.baseFunction(n, arr)
        else:
            orig(n, arr)
    return patched

for func in ['glGenTextures', 'glGenFramebuffers', 'glGenRenderbuffers', 'glGenVertexArrays', 'glGenBuffers']:
    setattr(gl, func, _patch_gl_gen(func))
for func in ['glDeleteTextures', 'glDeleteFramebuffers', 'glDeleteRenderbuffers', 'glDeleteVertexArrays', 'glDeleteBuffers']:
    setattr(gl, func, _patch_gl_delete(func))

import pyrender.texture
import pyrender.offscreen
import pyrender.renderer
for mod in [pyrender.texture, pyrender.offscreen, pyrender.renderer]:
    for func in ['glGenTextures', 'glGenFramebuffers', 'glGenRenderbuffers', 'glGenVertexArrays', 'glGenBuffers',
                 'glDeleteTextures', 'glDeleteFramebuffers', 'glDeleteRenderbuffers', 'glDeleteVertexArrays', 'glDeleteBuffers']:
        if hasattr(mod, func):
            setattr(mod, func, getattr(gl, func))

# ==========================================
# RENDERING & UTILITY FUNCTIONS
# ==========================================
def get_sha256(filepath):
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(4096), b""):
            sha256.update(block)
    return sha256.hexdigest()

def find_front_view(transforms):
    min_z = float('inf')
    front_idx = 0
    for i, frame in enumerate(transforms['frames']):
        c2w = np.array(frame['transform_matrix'])
        pos = c2w[:3, 3]
        if abs(pos[2]) < min_z:
            min_z = abs(pos[2])
            front_idx = i
    return front_idx

def render_mesh_opengl_headless(mesh_path, output_dir, sha256, num_views=150, width=512, height=512):
    mesh = trimesh.load(mesh_path, force='scene')
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())
    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2
    scale = 1.0 / (bounds[1] - bounds[0]).max()
    mesh.apply_translation(-center)
    mesh.apply_scale(scale)
    out_mesh_dir = os.path.join(output_dir, "renders", sha256)
    os.makedirs(out_mesh_dir, exist_ok=True)
    mesh.export(os.path.join(out_mesh_dir, "mesh.ply"))
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.5, 0.5, 0.5])
    pr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)
    scene.add(pr_mesh)
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=5.0)
    scene.add(light, pose=np.eye(4))
    fov = 40.0 / 180.0 * np.pi
    camera = pyrender.PerspectiveCamera(yfov=fov, aspectRatio=1.0)
    camera_node = pyrender.Node(camera=camera, matrix=np.eye(4))
    scene.add_node(camera_node)
    renderer = pyrender.OffscreenRenderer(width, height)
    frames = []
    radius = 2.0
    offset = (np.random.rand(), np.random.rand())
    for i in range(num_views):
        y, p = sphere_hammersley_sequence(i, num_views, offset)
        x = radius * np.cos(y) * np.cos(p)
        y_pos = radius * np.sin(y) * np.cos(p)
        z = radius * np.sin(p)
        C = np.array([x, y_pos, z])
        f = -C / np.linalg.norm(C)
        up_world = np.array([0, 0, 1])
        r = np.cross(f, up_world)
        if np.linalg.norm(r) < 1e-6:
            r = np.array([1, 0, 0])
        else:
            r = r / np.linalg.norm(r)
        u = np.cross(r, f)
        c2w = np.eye(4)
        c2w[:3, 0] = r
        c2w[:3, 1] = u
        c2w[:3, 2] = -f
        c2w[:3, 3] = C
        scene.set_pose(camera_node, pose=c2w)
        color, depth = renderer.render(scene, flags=pyrender.constants.RenderFlags.RGBA)
        img_dir = os.path.join(out_mesh_dir, "image")
        os.makedirs(img_dir, exist_ok=True)
        img_path = os.path.join(img_dir, f"{i:03d}.webp")
        Image.fromarray(color).save(img_path, "WEBP")
        frames.append({
            "file_path": f"image/{i:03d}.webp",
            "camera_angle_x": fov,
            "transform_matrix": c2w.tolist()
        })
    renderer.delete()
    with open(os.path.join(out_mesh_dir, "transforms.json"), "w") as f:
        json.dump({"frames": frames}, f, indent=4)

# ==========================================
# 2. BUILD METADATA & LINK MESHES
# ==========================================
print(" Step 1: Building metadata.csv and linking meshes...")
mesh_files = [f for f in os.listdir(MESH_DIR) if f.endswith(('.glb', '.obj', '.ply', '.usdz'))]
metadata = []
raw_dir = os.path.join(OUT_DIR, "raw")
os.makedirs(raw_dir, exist_ok=True)

for mesh in mesh_files:
    mesh_path = os.path.join(MESH_DIR, mesh)
    sha = get_sha256(mesh_path)
    dst = os.path.join(raw_dir, mesh)
    if not os.path.exists(dst):
        os.symlink(mesh_path, dst)
    metadata.append({
        "sha256": sha,
        "local_path": os.path.join("raw", mesh),
        "aesthetic_score": 5.0,
        "rendered": False,
        "cond_rendered": False
    })

df = pd.DataFrame(metadata)
df.to_csv(os.path.join(OUT_DIR, "metadata.csv"), index=False)

# Create custom.py in the datasets folder
custom_module_path = os.path.join(REPO_DIR, "dataset_toolkits/datasets/custom.py")
with open(custom_module_path, "w") as f:
    f.write("""
import pandas as pd
def add_args(parser): pass
def get_metadata(**kwargs): return pd.read_csv(kwargs['output_dir'] + '/metadata.csv')
def download(metadata, output_dir, **kwargs): return metadata[['sha256', 'local_path']]
def foreach_instance(metadata, output_dir, func, **kwargs):
    import pandas as pd
    records = []
    for _, row in metadata.iterrows():
        res = func(os.path.join(output_dir, row['local_path']), row['sha256'])
        if res: records.append(res)
    return pd.DataFrame.from_records(records)
""")

# ==========================================
# PATCH build_metadata.py to load custom.py correctly
# ==========================================
build_meta_path = os.path.join(REPO_DIR, "dataset_toolkits/build_metadata.py")
with open(build_meta_path, "r") as f:
    content = f.read()

old_import = '    dataset_utils = importlib.import_module(f\'datasets.{sys.argv[1]}\')'
new_import = """    import importlib.util
    _spec = importlib.util.spec_from_file_location("custom_ds", os.path.join(os.path.dirname(__file__), 'datasets', f'{sys.argv[1]}.py'))
    dataset_utils = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(dataset_utils)"""

if old_import in content:
    content = content.replace(old_import, new_import)
    with open(build_meta_path, "w") as f:
        f.write(content)
    print(" Patched build_metadata.py to use custom.py")
else:
    print(" build_metadata.py already patched or line not found")

# ==========================================
# 3. RUN TRIPOSPLAT TOOLKIT PIPELINE
# ==========================================
print(" Step 2: Rendering 150 views with OpenGL Headless (pyrender)...")
for _, row in tqdm(df.iterrows(), total=len(df), desc="Rendering meshes"):
    sha = row['sha256']
    mesh_path = os.path.join(OUT_DIR, row['local_path'])
    render_mesh_opengl_headless(mesh_path, OUT_DIR, sha, num_views=NUM_VIEWS)
print(" Updating metadata.csv to mark rendering as complete...")
df['rendered'] = True
df.to_csv(os.path.join(OUT_DIR, "metadata.csv"), index=False)

print(" Step 3: Extracting 3D Point Features (DINOv3)...")
subprocess.run([
    "python", "dataset_toolkits/extract_pcd_feature.py",
    "--output_dir", OUT_DIR,
    "--model", "dinov3_vith16plus",
    "--num_pcds", "16384",
    "--batch_size", "1"
], check=True)

print(" Step 3.5: Extracting 3D Point Features (FLUX.2 VAE)...")
subprocess.run([
    "python", "dataset_toolkits/extract_pcd_feature.py",
    "--output_dir", OUT_DIR,
    "--model", "flux2_dev_vae",
    "--num_pcds", "16384"
], check=True)

print(" Merging pcd feature records into metadata.csv...")
subprocess.run([
    "python", "dataset_toolkits/build_metadata.py", "custom",
    "--output_dir", OUT_DIR
], check=True)
df = pd.read_csv(os.path.join(OUT_DIR, "metadata.csv"))

print(" Step 4: Encoding 3D Latent Sequences (VAE)...")
# Use check=False so we can inspect output even if it fails
proc = subprocess.run([
    "python", "dataset_toolkits/encode_latentsequence.py",
    "--output_dir", OUT_DIR,
    "--latent_length", "1024",
    "--filter_low_aesthetic_score", "0.0"
], capture_output=True, text=True)

# Always print the output, because even on success there may be hidden errors
print("=== LATENT ENCODING STDOUT ===")
print(proc.stdout)
print("=== LATENT ENCODING STDERR ===")
print(proc.stderr)

if proc.returncode != 0:
    raise RuntimeError(f"Latent encoding subprocess failed with code {proc.returncode}")

# Check if files were created
latent_dir = os.path.join(OUT_DIR, "latents", "triposplat_vae_encoder_fp16")
if not os.path.exists(latent_dir):
    raise RuntimeError(f"Latent directory {latent_dir} does not exist. Encoding likely failed silently.")
npz_files = [f for f in os.listdir(latent_dir) if f.endswith('.npz')]
if len(npz_files) == 0:
    # If no files, print the captured output again and raise
    print(" No latent .npz files found. Full output:")
    print("=== STDOUT ===")
    print(proc.stdout)
    print("=== STDERR ===")
    print(proc.stderr)
    raise RuntimeError(f"No latent .npz files found in {latent_dir}. Encoding may have failed silently.")
else:
    print(f" Found {len(npz_files)} latent files.")

# Verify that .npz files were created
latent_dir = os.path.join(OUT_DIR, "latents", "triposplat_vae_encoder_fp16")
if os.path.exists(latent_dir):
    npz_files = [f for f in os.listdir(latent_dir) if f.endswith('.npz')]
    if len(npz_files) == 0:
        raise RuntimeError(f"No latent .npz files found in {latent_dir}. Encoding may have failed silently.")
    else:
        print(f" Found {len(npz_files)} latent files.")
else:
    raise RuntimeError(f"Latent directory {latent_dir} does not exist. Encoding failed.")

# ==========================================
# *** FIX: Merge latent records into metadata.csv ***
# ==========================================
print(" Merging latent records into metadata.csv...")
subprocess.run([
    "python", "dataset_toolkits/build_metadata.py", "custom",
    "--output_dir", OUT_DIR
], check=True)
df = pd.read_csv(os.path.join(OUT_DIR, "metadata.csv"))

# ==========================================
# 4. ORGANIZE CONDITIONING IMAGES
# ==========================================
print(" Step 5: Organizing renders_cond and renders_ctrl...")
renders_cond_dir = os.path.join(OUT_DIR, "renders_cond")
renders_ctrl_dir = os.path.join(OUT_DIR, "renders_ctrl")
os.makedirs(renders_cond_dir, exist_ok=True)
os.makedirs(renders_ctrl_dir, exist_ok=True)

ctrl_images = {
    os.path.splitext(f)[0]: os.path.join(CTRL_IMG_DIR, f)
    for f in os.listdir(CTRL_IMG_DIR) if f.endswith(('.png', '.jpg', '.webp'))
}

for _, row in tqdm(df.iterrows(), total=len(df), desc="Copying conditioning images"):
    sha = row['sha256']
    mesh_name = os.path.splitext(os.path.basename(row['local_path']))[0]

    render_dir = os.path.join(OUT_DIR, "renders", sha)
    transforms_path = os.path.join(render_dir, "transforms.json")
    if os.path.exists(transforms_path):
        with open(transforms_path, "r") as f:
            transforms = json.load(f)
        front_idx = find_front_view(transforms)
        front_frame = transforms['frames'][front_idx]
        cond_dst_dir = os.path.join(renders_cond_dir, sha, "image")
        os.makedirs(cond_dst_dir, exist_ok=True)
        src_img = os.path.join(render_dir, front_frame['file_path'])
        dst_img = os.path.join(cond_dst_dir, "000.webp")
        shutil.copy(src_img, dst_img)
        cond_transforms = {
            "frames": [{
                "file_path": "image/000.webp",
                "camera_angle_x": front_frame.get("camera_angle_x"),
                "camera_ortho_scale": front_frame.get("camera_ortho_scale"),
                "camera_orthographic": front_frame.get("camera_orthographic", False),
                "transform_matrix": front_frame["transform_matrix"]
            }]
        }
        with open(os.path.join(renders_cond_dir, sha, "transforms.json"), "w") as f:
            json.dump(cond_transforms, f)

    if mesh_name in ctrl_images:
        ctrl_src = ctrl_images[mesh_name]
        ctrl_dst_dir = os.path.join(renders_ctrl_dir, sha, "image")
        os.makedirs(ctrl_dst_dir, exist_ok=True)
        img = Image.open(ctrl_src).convert("RGBA")
        img.save(os.path.join(ctrl_dst_dir, "000.webp"), "WEBP")
        ctrl_transforms = {
            "frames": [{
                "file_path": "image/000.webp",
                "camera_angle_x": 0.6911,
                "camera_orthographic": False,
                "transform_matrix": [[1,0,0,0], [0,1,0,0], [0,0,1,2], [0,0,0,1]]
            }]
        }
        with open(os.path.join(renders_ctrl_dir, sha, "transforms.json"), "w") as f:
            json.dump(ctrl_transforms, f)

# ==========================================
# 5. FINALIZE METADATA & PUSH TO HF
# ==========================================
print(" Step 6: Finalizing metadata.csv...")
df['cond_rendered'] = True
df['rendered'] = True
df.to_csv(os.path.join(OUT_DIR, "metadata.csv"), index=False)

print(" Step 7: Pushing to Hugging Face...")
api = HfApi(token=HF_TOKEN)
create_repo(HF_REPO_ID, repo_type="dataset", token=HF_TOKEN, exist_ok=True)
api.upload_folder(
    folder_path=OUT_DIR,
    repo_id=HF_REPO_ID,
    repo_type="dataset",
    token=HF_TOKEN,
    commit_message="Upload TripoSplat Control LoRA Dataset"
)

print("\n" + "="*50)
print(" SUCCESS! Dataset is now on Hugging Face!")
print(f" View it here: https://huggingface.co/datasets/{HF_REPO_ID}")
print("="*50)
