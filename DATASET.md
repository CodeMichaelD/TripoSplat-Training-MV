# TRELLIS-500K Data Preparation

TRELLIS-500K is a dataset of 500K 3D assets curated from [Objaverse(XL)](https://objaverse.allenai.org/), [ABO](https://amazon-berkeley-objects.s3.amazonaws.com/index.html), [3D-FUTURE](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-future), [HSSD](https://huggingface.co/datasets/hssd/hssd-models), and [Toys4k](https://github.com/rehg-lab/lowshot-shapebias/tree/main/toys4k), filtered based on aesthetic scores.
This dataset serves for 3D generation tasks.

The dataset is provided as csv files containing the 3D assets' metadata.

## Dataset Statistics

The following table summarizes the dataset's filtering and composition:

***NOTE: Some of the 3D assets lack text captions. Please filter out such assets if captions are required.***
| Source | Aesthetic Score Threshold | Filtered Size | With Captions |
|:-:|:-:|:-:|:-:|
| ObjaverseXL (sketchfab) | 5.5 | 168307 | 167638 |
| ObjaverseXL (github) | 5.5 | 311843 | 306790 |
| ABO | 4.5 | 4485 | 4390 |
| 3D-FUTURE | 4.5 | 9472 | 9291 |
| HSSD | 4.5 | 6670 | 6661 |
| All (training set) | - | 500777 | 494770 |
| Toys4k (evaluation set) | 4.5 | 3229 | 3180 |

## Environment

Install the toolkit dependencies:

```bash
. ./dataset_toolkits/setup.sh
```

By default, DINOv3 is loaded from
`facebook/dinov3-vith16plus-pretrain-lvd1689m`. Optionally set
`DINO_V3_PATH` to use an already downloaded model directory:

```bash
export DINO_V3_PATH=/path/to/dinov3-vith16plus-pretrain-lvd1689m
```

For DINOv3 and FLUX.2 feature extraction, follow the access and authentication
requirements of their respective Hugging Face repositories. 

## Step 1: Build Metadata

Create the initial `metadata.csv`:

```bash
python dataset_toolkits/build_metadata.py <SUBSET> \
  --output_dir <DATA_ROOT> \
  [--source <SOURCE>]
```

- `SUBSET`: `ObjaverseXL`, `ABO`, `3D-FUTURE`, `HSSD`, or `Toys4k`.
- `OUTPUT_DIR`: dataset root to create or update.
- `SOURCE`: required for `ObjaverseXL`; common values are `sketchfab` and `github`.

Example:

```bash
python dataset_toolkits/build_metadata.py ObjaverseXL \
  --source sketchfab \
  --output_dir datasets/ObjaverseXL_sketchfab
```

## Step 2: Download Assets

Download source 3D assets:

```bash
python dataset_toolkits/download.py <SUBSET> \
  --output_dir <DATA_ROOT> \
  [--rank <RANK> --world_size <WORLD_SIZE>]
```

Merge records by rerunning `build_metadata.py`:

```bash
python dataset_toolkits/build_metadata.py <SUBSET> --output_dir <DATA_ROOT>
```

For distributed processing, run every rank with the same `world_size`, then run `build_metadata.py` once after all ranks finish.

## Step 3: Render Multiview Images

The render scripts use Blender 4.0.1. By default they download it under
`/tmp/blender` and attempt to install required system libraries with `apt-get`.
On machines without root access, install Blender and its runtime libraries in
advance and set `BLENDER_INSTALLATION_PATH` to the directory containing the
extracted `blender-4.0.1-linux-x64` folder.

Render the canonical multiview images used by feature projection and reconstruction supervision:

```bash
python dataset_toolkits/render.py <SUBSET> \
  --output_dir <DATA_ROOT> \
  --num_views 150 \
  [--engine CYCLES] \
  [--rank <RANK> --world_size <WORLD_SIZE>]
```

Merge records by rerunning `build_metadata.py`:

```bash
python dataset_toolkits/build_metadata.py <SUBSET> --output_dir <DATA_ROOT>
```

This creates `renders/<sha256>/transforms.json`, rendered images, and an exported mesh.

## Step 4: Project Image Features To Points

Extract DINOv3 point-projected features:

```bash
python dataset_toolkits/extract_pcd_feature.py \
  --output_dir <DATA_ROOT> \
  --model dinov3_vith16plus \
  --num_pcds 16384 \
  [--rank <RANK> --world_size <WORLD_SIZE>]
```

Extract FLUX.2 VAE point-projected features:

```bash
python dataset_toolkits/extract_pcd_feature.py \
  --output_dir <DATA_ROOT> \
  --model flux2_dev_vae \
  --num_pcds 16384 \
  [--rank <RANK> --world_size <WORLD_SIZE>]
```

Merge records by rerunning `build_metadata.py`:

```bash
python dataset_toolkits/build_metadata.py <SUBSET> --output_dir <DATA_ROOT>
```

The generated `.npz` files contain sampled points under `pcds` and point-aligned features under `patchtokens`.

## Step 5: Encode Latent Sequences

Use the released VAE encoder checkpoint to encode point features into latent sequences.
By default, the script loads
`hf://VAST-AI/TripoSplat/vae/triposplat_vae_encoder_fp16.safetensors`
with the bundled stage-3 VAE config:

```bash
python dataset_toolkits/encode_latentsequence.py \
  --output_dir <DATA_ROOT> \
  --latent_length 1024 2048 4096 8192 \
  [--rank <RANK> --world_size <WORLD_SIZE>]
```

To use an encoder from a training run instead:

```bash
python dataset_toolkits/encode_latentsequence.py \
  --output_dir <DATA_ROOT> \
  --model_root outputs/vae/stage3-gs_fixlen_vae-XL-fps \
  --enc_model latentseq_xl \
  --ckpt step0400000 \
  --latent_length 1024 2048 4096 8192
```

With the released encoder, this writes
`latents/triposplat_vae_encoder_fp16/<sha256>.npz` and metadata column
`latent_triposplat_vae_encoder_fp16`. Training-run encoders use
`latents/<LATENT_MODEL_PREFIX>_<CKPT_NAME>/<sha256>.npz` and the corresponding
metadata column.

Merge records by rerunning `build_metadata.py`:

```bash
python dataset_toolkits/build_metadata.py <SUBSET> --output_dir <DATA_ROOT>
```

## Step 6: Render Image Conditions

Render image-conditioning views:

```bash
python dataset_toolkits/render_cond.py <SUBSET> \
  --output_dir <DATA_ROOT> \
  --num_views 16 \
  [--engine CYCLES] \
  [--rank <RANK> --world_size <WORLD_SIZE>]
```

Merge records by rerunning `build_metadata.py`:

```bash
python dataset_toolkits/build_metadata.py <SUBSET> --output_dir <DATA_ROOT>
```

## Optional: Calculate Aesthetic Scores

This step is optional for datasets that already contain `aesthetic_score`, and recommended for newly rendered data.

```bash
python dataset_toolkits/calculate_aesthetic_scores.py \
  --output_dir <DATA_ROOT> \
  [--rank <RANK> --world_size <WORLD_SIZE>]
```

Merge records by rerunning `build_metadata.py`:

```bash
python dataset_toolkits/build_metadata.py <SUBSET> --output_dir <DATA_ROOT>
```

## Optional: Train/Validation Split

After all required artifacts have been generated and merged into `metadata.csv`, optionally create train and validation metadata roots:

```bash
python dataset_toolkits/split_valdata.py \
  --output_dir <DATA_ROOT> \
  --num_val 512
```

This writes:

- `<DATA_ROOT>_train/metadata.csv`
- `<DATA_ROOT>_train_val/metadata.csv`

It symlinks heavy data folders (`renders`, `renders_cond`, `pcd_features`, and `latents` when present) back to `<DATA_ROOT>`.

Rerun this split step whenever you add new metadata columns that the split roots need.

## Consistency Checks

Refresh metadata from files if records were interrupted or lost:

```bash
python dataset_toolkits/build_metadata.py <SUBSET> --output_dir <DATA_ROOT> --from_file
```

Check `statistics.txt` for nonzero counts matching the artifacts you need:

- rendered assets
- aesthetic scores, if used for filtering
- `pcd_feature_dinov3_vith16plus`
- `pcd_feature_flux2_dev_vae`                                                        
- `latent_<LATENT_MODEL>`
- image conditions
