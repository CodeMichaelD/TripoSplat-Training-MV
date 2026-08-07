# TripoSplat-Training

TripoSplat converts a single 2D image into high-quality and variable number of 3D Gaussians, developed by [TripoAI](https://www.tripo3d.ai/). It can serve as a powerful pipeline tool for asset creation, AR/VR, game development, simulation environments, and beyond. This is the training repo for TripoSplat. Refer to [TripoSplat](https://github.com/VAST-AI-Research/TripoSplat) for a lightweight inference only installation.

<a href="https://arxiv.org/abs/2605.16355"><img src="https://img.shields.io/badge/Read%20Paper-B31B1B?style=for-the-badge&logo=arxiv" alt="Paper"></a>
<a href="https://www.tripo3d.ai/research/triposplat"><img src="https://img.shields.io/badge/Technical%20Blog-grey?style=for-the-badge&logo=data:image/svg%2bxml;base64,PHN2ZyB3aWR0aD0iNjUiIGhlaWdodD0iNjUiIHZpZXdCb3g9IjAgMCA2NSA2NSIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTkuNDk5MSA5LjYzNDc3TDE2LjQzNzQgMjEuNDU1NkMxNi40MzkzIDIxLjQ1ODkgMTYuNDQxMiAyMS40NjIyIDE2LjQ0MzEgMjEuNDY1NUwzMC4yNjU4IDQ1LjA1NDhDMzEuNTMyNyA0Ny4yMTY3IDM0LjcwNDUgNDcuMjE2NyAzNS45NzE0IDQ1LjA1NDhMNDkuMzg2MiAyMi4xNjE2SDU5LjQ2MThMNDEuMjY2IDUzLjE2MkMzNy42NDQ5IDU5LjMzMTMgMjguNTkyMyA1OS4zMzEzIDI0Ljk3MTIgNTMuMTYyTDYuNjM5NjcgMjEuOTMwMkM0LjAyNCAxNy40NzM5IDUuNjU5NTYgMTIuMjEyNyA5LjQ5OTEgOS42MzQ3N1oiIGZpbGw9IndoaXRlIi8+CjxwYXRoIGQ9Ik0yMC4xMTIxIDE2LjYwODdIMzQuNjkyNkwyOC42MjIgMjcuMDQ0MkMyOC4yMDMzIDI3Ljc2NCAyOC4yMDgzIDI4LjY0OTIgMjguNjM1MSAyOS4zNjQ0TDMxLjA1MjcgMzMuNDE1MUMzMS45NjU0IDM0Ljk0NDUgMzQuMjE2MyAzNC45MzY1IDM1LjExNzggMzMuNDAwNkw0NC45NzM5IDE2LjYwODdINDYuOTQyTDQ2Ljk0NTUgMTYuNjA4N0g2MC44NDQ2QzYwLjQ4MzIgMTIuMDU4NyA1Ni42NzMxIDguMDQ4ODMgNTEuNDUwOSA4LjA0ODgzTDE1LjA4NzkgOC4wNDg4M0wyMC4xMTIxIDE2LjYwODdaIiBmaWxsPSIjRjhDRjAwIi8+Cjwvc3ZnPgo=" alt="Technical Blog"></a>
<a href="https://huggingface.co/spaces/VAST-AI/TripoSplat"><img src="https://img.shields.io/badge/Huggingface%20Demo-grey?style=for-the-badge&logo=huggingface" alt="HuggingFace Demo"></a>

| ![](assets/static/001.webp) | ![](assets/static/002.webp) |
|---|---|
| ![](assets/static/003.webp) | ![](assets/static/004.webp) |

<!-- Features -->
## 🌟 Features
- **High-quality, versatile generation** that handles a wide range of image styles.
- **Arbitrary Gaussian count** (up to 262,144) — trade off visual quality against rendering cost according to your need.
- **Official ComfyUI support**: drop the [official workflow template](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/3d_triposplat_image_to_gaussian_splat.json) into ComfyUI and start playing with TripoSplat right away.

<!-- Updates -->
## ⏩ Updates
- ✅ Training code release
<!-- Installation -->
## 📦 Installation

### Prerequisites
- **System**: The code is currently tested only on **Linux**.
- **Hardware**: An NVIDIA GPU with at least 16GB of memory is necessary. The code has been verified on NVIDIA A800 and 4090 GPUs.  
- **Software**:   
  - The [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) is needed to compile certain submodules. The code has been tested with CUDA versions 12.6.  
  - [Conda](https://docs.anaconda.com/miniconda/install/#quick-command-line-install) is recommended for managing dependencies.  
  - Python version 3.8 or higher is required. 

### Installation Steps
1. Clone the repo:
    ```sh
    git clone --recurse-submodules https://github.com/runjie-yan/TripoSplat-Training
    cd DeG
    ```
2. (Optional) Install miniconda, suggested for managing dependencies. Skip if you have already installed conda. Run the following commands in the terminal to install miniconda:
    ```sh
    curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p "$HOME/miniconda3"
    source "$HOME/miniconda3/bin/activate"
    conda init
    conda config --set auto_activate_base false
    rm ./Miniconda3-latest-Linux-x86_64.sh
    ```
3. Set local environment:
    **Before running the following command there are somethings to note:**
    - By adding `--new-env`, a new conda environment named `deg` will be created. If you want to use an existing conda environment, please remove this flag.
    - By default the `deg` environment will use pytorch 2.9.1 with CUDA 12.6, 12.8 or 13.0. The CUDA version is automatically detected. You can use a previous version of pytorch like 2.6.0 if you already have it installed. If you still want to use a different version of CUDA or pytorch, refer to [PyTorch](https://pytorch.org/get-started/previous-versions/) for the installation command.
    - By default, the code uses the `flash-attn` backend for attention.
    - The installation may take a while due to the large number of dependencies. Please be patient. If you encounter any issues, you can try to install the dependencies one by one, specifying one flag at a time.
    - If you encounter any issues during the installation, feel free to open an issue or contact us.

    Create a new conda environment named `deg` and install the dependencies:
    ```sh
    . ./setup.sh --new-env --basic --train --flash-attn --mipgaussianl1c --kaolin --nvdiffrast --extension-path /tmp/extensions
    ```

    The detailed usage of `setup.sh` can be found by running `. ./setup.sh --help`.


<!-- Pretrained Models -->
## 🤖 Pretrained Models
Pretrained models are released on [Hugging Face](https://huggingface.co/VAST-AI/TripoSplat/tree/main).
- **VAE Encoder**: `vae/triposplat_vae_encoder_fp16.safetensors`
- **VAE Decoder**: `vae/triposplat_vae_decoder_fp16.safetensors`
- **Flow Generator (DiT)**: `diffusion_models/triposplat_fp16.safetensors`
- **DINOv3 image encoder**: [`facebook/dinov3-vith16plus-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m)
- **FLUX.2 image-conditioning VAE**: the `vae` subfolder of [`black-forest-labs/FLUX.2-dev`](https://huggingface.co/black-forest-labs/FLUX.2-dev)


<!-- Inference -->
## 💡 Inference
Run `inference_gs.py` to generate 3D Gaussians from a single image:

```sh
python inference_gs.py \
    --image_path assets/example_inputs/plant_water_lily.webp \
    --output_dir outputs/sample_gs
```

Inference loads the TripoSplat VAE decoder, DINOv3 encoder, and FLUX.2 VAE
above in addition to the denoiser. Download DINOv3 and FLUX.2 from their
respective Hugging Face repositories. Access or authentication may be required
according to each repository's terms.


<!-- Dataset -->
## 📚 Dataset

We use **TRELLIS-500K** dataset following [TRELLIS](https://github.com/Microsoft/TRELLIS). The dataset contains 500K 3D assets curated from [Objaverse(XL)](https://objaverse.allenai.org/), [ABO](https://amazon-berkeley-objects.s3.amazonaws.com/index.html), [3D-FUTURE](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-future), [HSSD](https://huggingface.co/datasets/hssd/hssd-models), and [Toys4k](https://github.com/rehg-lab/lowshot-shapebias/tree/main/toys4k), filtered based on aesthetic scores. Please refer to the [dataset README](DATASET.md) for more details.

<!-- Training -->
## 🏋️‍♂️ Training
### Training Setup

1. **Prepare the Environment:**
   - Ensure all training dependencies are installed.
   - Use a Linux system with an NVIDIA GPU (The models are trained on NVIDIA A100 GPUs).
   - For distributed training, verify that your nodes can communicate through the designated master address and port.

2. **Dataset Preparation:**
   - Organize your dataset similar to TRELLIS-500K. Specify your dataset path using the `--data_dir` argument when launching training. Please refer to the [dataset README](DATASET.md) for more details.

3. **Prepare Pretrained Vision Encoders:**
   - FLUX.2 VAE: Download the `vae` subfolder from [black-forest-labs/FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev). Follow that repository's access and authentication requirements.
   - DinoV3: Download [facebook/dinov3-vith16plus-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m). Follow that repository's access and authentication requirements. Set `DINO_V3_PATH` only to override the default Hugging Face model ID with a local checkpoint directory.
  
4. **Configuration Files:**
   - Training hyperparameters and model architectures are defined in configuration files under the `configs/` directory.
   - The vae and dit model is trained with a multi-stage strategy. The order of training is:
        - VAE: 
            1. `vae/stage1-pcd_fixlen_vae-XL-fps.yaml`
            2. `vae/stage2-gs_fixlen_vae-XL-fps.yaml`
            3. `vae/stage3-gs_fixlen_vae-XL-fps.yaml`
        - DiT: 
            1. `dit/latent1k-latentseq_flow_img_s3dit-L.yaml`
            2. `dit/latent2k-latentseq_flow_img_s3dit-L.yaml`
            3. `dit/latent4k-latentseq_flow_img_s3dit-L.yaml`
            4. `dit/latent8k-latentseq_flow_img_s3dit-L.yaml`
            5. `dit/latent8k-cond1k-latentseq_flow_img_s3dit-L.yaml`
            
### Example Training Commands
#### Single-node Training

To train the vae model, run the following command:

```sh
python train.py \
    --config        configs/vae/stage1-pcd_fixlen_vae-XL-fps.yaml \
    --output_dir    outputs/vae/stage1-pcd_fixlen_vae-XL-fps \
    --data_dir              /path/to/data_train
```
You can optionally use `--data_train_val_dir    /path/to/data_train_val` to specify the validation dataset path.
The script will automatically distribute the training across all available GPUs. Specify the number of GPUs with the `--num_gpus` flag if you want to limit the number of GPUs used.

After the first stage training finish, you should go on with next stage training and replace the `trainer.args.finetune_ckpt` key in `stage2-gs_fixlen_vae-XL-fps.yaml` and `stage3-gs_fixlen_vae-XL-fps.yaml` with the checkpoint of the previous stage training.

To train the dit model, you need to first prepare a latent sequence dataset using either the vae model you trained or the pretrained vae model, following [DATASET](./DATASET.md). To train the dit model, run the following command:

```sh
python train.py \
    --config        configs/dit/latent1k-latentseq_flow_img_s3dit-L.yaml \
    --output_dir    outputs/dit/latent1k-latentseq_flow_img_s3dit-L \
    --data_dir      /path/to/data_train
```

Also, you need to replace the `trainer.args.finetune_ckpt` key in the next stage training with the checkpoint of the previous stage training.

#### Multi-node Training
To train the models on multiple GPUs (e.g. 4 nodes), you need to adajust `num_nodes` `node_rank` `master_addr` `master_port`. For example, to train the vae model, run the following command:

```sh
python train.py \
    --config        configs/vae/stage1-pcd_fixlen_vae-XL-fps.yaml \
    --output_dir    outputs/vae/stage1-pcd_fixlen_vae-XL-fps \
    --data_dir      /path/to/data_train \
    --num_nodes     4 \
    --node_rank     $RANK \
    --master_addr   $MASTER_ADDR \
    --master_port   $MASTER_PORT 
```

### Additional Options

- **Debug Run:** The `--debug` flag allows you to check the training sanity quickly, it will set num_gpus to 1, stop wandb logging and quit after 100 training steps.
- **Dry Run:** The `--tryrun` flag allows you to check your configuration and environment without launching full training.
- **Profiling:** Enable profiling with the `--profile` flag to gain insights into training performance and diagnose bottlenecks.

Adjust the file paths and parameters to match your experimental setup.

<!-- License -->
## ⚖️ License

TripoSplat models and the majority of the code are licensed under the [MIT License](LICENSE). The following components have separate licenses:
- [**mip-gaussian-l1c**](https://github.com/runjie-yan/mip-gaussian-l1c): This externally installed CUDA renderer provides an efficient per-Gaussian L1 contribution metric and is derived from [mip-splatting](https://github.com/autonomousvision/mip-splatting). It is distributed under the upstream [Gaussian-Splatting License](https://github.com/runjie-yan/mip-gaussian-l1c/blob/main/LICENSE.md), which permits non-commercial research and evaluation use only.
- [**io_scene_usdz**](https://github.com/robmcrosby/BlenderUSDZ): The bundled Blender USDZ import/export extension in `dataset_toolkits/blender_script/io_scene_usdz.zip` is distributed under the [GNU General Public License v3.0](https://github.com/robmcrosby/BlenderUSDZ/blob/master/LICENSE), not the root MIT license.


<!-- Acknowledgements -->
## 🌟 Acknowledgements
- **TRELLIS**: The code is mainly based on [TRELLIS](https://github.com/Microsoft/TRELLIS) project.
- **S3-DiT**: The diffusion transformer model architecture follows the design of [Z-Image](https://github.com/Tongyi-MAI/Z-Image).
- **RePo**: We experimentially adopt [RePo](https://github.com/SakanaAI/repo) for multi-modal reindexing in **S3-DiT**.
- **3DShape2VecSet**: Our shape representation is partially based on paper [3DShape2VecSet](https://arxiv.org/abs/2301.11445).

<!-- Citation -->
## 📜 Citation

If you find this work helpful, please consider citing our paper:
```bibtex
@misc{yan2026generative3dgaussianslearned,
      title={Generative 3D Gaussians with Learned Density Control}, 
      author={Runjie Yan and Yan-Pei Cao and Peng Wang and Ding Liang and Yuan-Chen Guo},
      year={2026},
      eprint={2605.16355},
      archivePrefix={arXiv},
      primaryClass={cs.GR},
      url={https://arxiv.org/abs/2605.16355}, 
}
```
