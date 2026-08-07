"""
Minimal inference example for DEG (Deformable Gaussian) model.

Usage:
    python example.py \
        --config  configs/generation/latent1k-latentseq_flow_img_s3dit-L.yaml \
        --ckpt    /path/to/denoiser_stepXXXXXXX.pt \
        --decoder_path /path/to/stage3_vae_dir \
        --decoder_ckpt step0400000 \
        --image   /path/to/image.png \
        --output_dir outputs/
"""

import os
import argparse

import numpy as np
import imageio
from PIL import Image

from deg.pipelines import DEGImageTo3DPipeline
from deg.utils import render_utils
from deg.utils.render_utils import decode_latent_camera_params
from inference_gs import load_image


def main():
    parser = argparse.ArgumentParser(description="DEG minimal inference example")
    parser.add_argument('--config',       required=True, help="Generation config yaml")
    parser.add_argument('--ckpt',         required=True, help="Denoiser checkpoint (.pt)")
    parser.add_argument('--decoder_path', default=None,  help="Stage-3 VAE checkpoint directory")
    parser.add_argument('--decoder_ckpt', default=None,  help="Decoder checkpoint tag, e.g. step0400000")
    parser.add_argument('--decoder_config', default=None, help="Decoder config json/yaml for HF decoder loading")
    parser.add_argument('--image',        required=True, help="Input image path")
    parser.add_argument('--output_dir',   default='outputs')
    parser.add_argument('--seed',         type=int,   default=42)
    parser.add_argument('--steps',        type=int,   default=50)
    parser.add_argument('--cfg_strength', type=float, default=7.0)
    parser.add_argument('--rescale_t',    type=float, default=3.0)
    parser.add_argument('--save_ply',     action='store_true', help="Save .ply file")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.image))[0]

    # Build pipeline
    pipeline = DEGImageTo3DPipeline(
        config_path=args.config,
        ckpt_path=args.ckpt,
        decoder_path=args.decoder_path,
        decoder_ckpt=args.decoder_ckpt,
        decoder_config_path=args.decoder_config,
    )

    # Run inference
    gaussian, pcd_gaussian, camera = pipeline.run(
        args.image,
        seed=args.seed,
        steps=args.steps,
        cfg_strength=args.cfg_strength,
        rescale_t=args.rescale_t,
    )

    # Decode camera params
    yaw_base, pitch_base, radius, fov = 0.0, np.deg2rad(20), 2.0, 40.0
    if camera is not None:
        yaw_base, fov, radius, pitch_base = decode_latent_camera_params(camera[0, 0])
        yaw_base   = float(yaw_base.detach().cpu())
        fov        = float(fov.detach().cpu())
        radius     = float(radius.detach().cpu())
        pitch_base = float(pitch_base.detach().cpu())

    # Save PLY
    if args.save_ply:
        ply_path = os.path.join(args.output_dir, f"{stem}.ply")
        gaussian.save_ply(ply_path)
        print(f"Saved PLY  -> {ply_path}")

    # Render turntable video
    frames = render_utils.render_video(
        gaussian,
        yaws_base=yaw_base,
        pitch_base=pitch_base,
        r=radius,
        fov=fov,
    )['color']
    video_path = os.path.join(args.output_dir, f"{stem}.mp4")
    imageio.mimsave(video_path, frames, fps=30)
    print(f"Saved video -> {video_path}")

    # Side-by-side with input image
    ref = load_image(args.image)
    h = frames[0].shape[0]
    ref_arr = np.array(ref.resize((int(ref.width * h / ref.height), h)))
    if frames[0].dtype != np.uint8:
        ref_arr = ref_arr.astype(frames[0].dtype) / 255.0
    combined = [np.concatenate([f, ref_arr], axis=1) for f in frames]
    combined_path = os.path.join(args.output_dir, f"{stem}_with_input.mp4")
    imageio.mimsave(combined_path, combined, fps=30)
    print(f"Saved side-by-side -> {combined_path}")


if __name__ == '__main__':
    main()
