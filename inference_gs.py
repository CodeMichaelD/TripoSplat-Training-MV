import os
import argparse

import numpy as np
import imageio
from PIL import Image
from tqdm import tqdm

from deg.pipelines import DEGImageTo3DPipeline
from deg.utils import render_utils
from deg.utils.hf_utils import DEFAULT_HF_DENOISER_PATH, default_hf_config_path
from deg.utils.render_utils import decode_latent_camera_params


def load_image(path):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',       type=str, default=None)
    parser.add_argument('--ckpt',         type=str, default=DEFAULT_HF_DENOISER_PATH,
                        help='Hugging Face denoiser reference (used from the local HF cache when offline)')
    parser.add_argument('--image_path',   type=str, default=None)
    parser.add_argument('--output_dir',   type=str, default='outputs/sample_gs')
    parser.add_argument('--cfg_strength', type=float, default=3.0)
    parser.add_argument('--rescale_t',    type=float, default=3.0)
    parser.add_argument('--steps',        type=int, default=50)
    parser.add_argument('--render_test_views', action='store_true')
    parser.add_argument('--no_yaws',      action='store_true')
    parser.add_argument('--user_study_mode', action='store_true')
    parser.add_argument('--save_ply',     action='store_true')
    parser.add_argument('--rank',         type=int, default=0)
    parser.add_argument('--world_size',   type=int, default=1)
    args = parser.parse_args()
    if args.config is None:
        args.config = default_hf_config_path()

    if not args.ckpt.startswith('hf://'):
        parser.error("--ckpt must be an hf:// Hugging Face reference")
    print(f"Loading denoiser: {args.ckpt}")

    # ------------------------------------------------------------------ #
    # Output directory                                                     #
    # ------------------------------------------------------------------ #
    step_name = f"gs_cfg{args.cfg_strength:.1f}_rescale{args.rescale_t:.1f}"
    final_output_dir = os.path.join(args.output_dir, step_name)
    os.makedirs(final_output_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Build pipeline                                                       #
    # ------------------------------------------------------------------ #
    pipeline = DEGImageTo3DPipeline(
        args.config,
        args.ckpt,
    )

    # ------------------------------------------------------------------ #
    # Collect images                                                       #
    # ------------------------------------------------------------------ #
    if os.path.isdir(args.image_path):
        valid_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
        image_files = sorted(
            os.path.join(root, f)
            for root, _, files in os.walk(args.image_path)
            for f in files if os.path.splitext(f)[1].lower() in valid_exts
        )
    else:
        image_files = [args.image_path]
    image_files = image_files[args.rank::args.world_size]
    print(f"Processing {len(image_files)} images.")

    # ------------------------------------------------------------------ #
    # Inference loop                                                       #
    # ------------------------------------------------------------------ #
    for img_path in tqdm(image_files):
        print("=" * 50)
        print(f"  {os.path.basename(img_path)}")
        gaussian, pcd_gaussian, camera_latent = pipeline.run(
            img_path,
            steps=args.steps,
            cfg_strength=args.cfg_strength,
            rescale_t=args.rescale_t,
        )

        # Camera params
        yaw_base, fov, radius, pitch_base = 0.0, 40.0, 2.0, np.deg2rad(20)
        if camera_latent is not None and not args.no_yaws:
            yaw_base, fov, radius, pitch_base = decode_latent_camera_params(camera_latent[0, 0])
            yaw_base = float(yaw_base.detach().cpu())
            fov = float(fov.detach().cpu())
            radius = float(radius.detach().cpu())
            pitch_base = float(pitch_base.detach().cpu())

        img_name = os.path.splitext(os.path.basename(img_path))[0]

        if args.save_ply:
            ply_dir = os.path.join(final_output_dir, 'ply')
            os.makedirs(ply_dir, exist_ok=True)
            gaussian.save_ply(os.path.join(ply_dir, f'{img_name}.ply'))
            if camera_latent is not None and not args.no_yaws:
                np.savetxt(
                    os.path.join(ply_dir, f'{img_name}_camera_params.txt'),
                    [yaw_base, fov, radius, pitch_base],
                )

        if args.render_test_views:
            yaws = [np.deg2rad(45 * k) + yaw_base for k in range(8)]
            pitchs = [np.deg2rad(30)] * 8
            extr, intr = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
                yaws, pitchs, 2.0, 40.0)
            opts = {'resolution': 512, 'bg_color': (0, 0, 0)}
            video = render_utils.render_frames(gaussian, extr, intr, opts)['color']
            video_pcd = render_utils.render_frames(pcd_gaussian, extr, intr, opts)['color']
        elif args.user_study_mode:
            video = render_utils.render_video(gaussian, yaws_base=0.5 * np.pi, pitch_rate=0)['color']
            video_pcd = render_utils.render_video(pcd_gaussian, yaws_base=0.5 * np.pi, pitch_rate=0)['color']
        else:
            video = render_utils.render_video(
                gaussian, yaws_base=yaw_base, pitch_base=pitch_base, r=radius, fov=fov
            )['color']
            video_pcd = render_utils.render_video(
                pcd_gaussian, yaws_base=yaw_base, pitch_base=pitch_base, r=radius, fov=fov
            )['color']

        xyzs = gaussian.get_xyz
        print(f"  BBox: {xyzs.max(0).values - xyzs.min(0).values}")
        print(f"  Dead GS: {(gaussian.get_opacity < 0.05).float().mean() * 100:.1f}%")

        segs = [('default', video, video_pcd)]

        if args.render_test_views:
            for tag, vid, _ in segs:
                tag_slug = tag.replace(' ', '_').lower()
                out_dir = os.path.join(final_output_dir, img_name, tag_slug, 'image')
                os.makedirs(out_dir, exist_ok=True)
                for fi, frame in enumerate(vid):
                    Image.fromarray(frame).save(os.path.join(out_dir, f'{fi:03d}.png'))
        else:
            ref_img = load_image(img_path)
            base_vid = segs[0][1]
            if not base_vid:
                continue
            target_h = base_vid[0].shape[0]
            ref_arr  = np.array(ref_img.resize((int(ref_img.width * target_h / ref_img.height), target_h)))
            dtype_ref = base_vid[0].dtype
            if np.issubdtype(dtype_ref, np.floating) and ref_arr.dtype == np.uint8:
                ref_arr = ref_arr.astype(dtype_ref) / 255.0

            final_frames        = []
            final_frames_no_pcd = []
            for fi in range(len(base_vid)):
                row, row_no_pcd = [], []
                for _, vid, vid_pcd in segs:
                    f  = vid[fi]     if fi < len(vid)     else np.zeros_like(base_vid[0])
                    fp = vid_pcd[fi] if fi < len(vid_pcd) else np.zeros_like(base_vid[0])
                    row.extend([f, fp])
                    row_no_pcd.append(f)
                row.append(ref_arr)
                row_no_pcd.append(ref_arr)
                final_frames.append(np.concatenate(row, axis=1))
                final_frames_no_pcd.append(np.concatenate(row_no_pcd, axis=1))

            imageio.mimsave(os.path.join(final_output_dir, f'full_{img_name}.mp4'), final_frames, fps=30)
            imageio.mimsave(os.path.join(final_output_dir, f'_{img_name}.mp4'), final_frames_no_pcd, fps=30)

        imageio.mimsave(os.path.join(final_output_dir, f'{img_name}.mp4'), segs[-1][1], fps=30)


if __name__ == '__main__':
    main()
