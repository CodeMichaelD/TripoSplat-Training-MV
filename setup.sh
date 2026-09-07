# TODO: create a set up
# Read Arguments
TEMP=`getopt -o h --long help,new-env,basic,train,xformers,flash-attn,kaolin,mipgaussianl1c,nvdiffrast,demo,extension-path: -n 'setup.sh' -- "$@"`

eval set -- "$TEMP"

HELP=false
NEW_ENV=false
BASIC=false
TRAIN=false
XFORMERS=false
FLASHATTN=false
ERROR=false
MIPGAUSSIANL1C=false
KAOLIN=false
NVDIFFRAST=false
DEMO=false
EXTENSION_PATH=/tmp/extensions

if [ "$#" -eq 1 ] ; then
    HELP=true
fi

while true ; do
    case "$1" in
        -h|--help) HELP=true ; shift ;;
        --new-env) NEW_ENV=true ; shift ;;
        --basic) BASIC=true ; shift ;;
        --train) TRAIN=true ; shift ;;
        --xformers) XFORMERS=true ; shift ;;
        --flash-attn) FLASHATTN=true ; shift ;;
        --mipgaussianl1c) MIPGAUSSIANL1C=true ; shift ;;
        --kaolin) KAOLIN=true ; shift ;;
        --nvdiffrast) NVDIFFRAST=true ; shift ;;
        --demo) DEMO=true ; shift ;;
        --extension-path) EXTENSION_PATH="$2" ; shift 2 ;;
        --) shift ; break ;;
        *) ERROR=true ; break ;;
    esac
done

if [ "$ERROR" = true ] ; then
    echo "Error: Invalid argument"
    HELP=true
fi

detect_cuda_version() {
    if command -v nvcc >/dev/null 2>&1 ; then
        nvcc --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n1
        return
    fi

    if command -v nvidia-smi >/dev/null 2>&1 ; then
        nvidia-smi | sed -n 's/.*CUDA Version: \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n1
    fi
}

pick_torch_cuda_index() {
    awk -v cuda="$1" 'BEGIN {
        if (cuda == "" || cuda < 12.7) {
            print "https://download.pytorch.org/whl/cu126"
        } else if (cuda < 12.9) {
            print "https://download.pytorch.org/whl/cu128"
        } else {
            print "https://download.pytorch.org/whl/cu130"
        }
    }'
}

normalize_torch_version() {
    echo "$1" | cut -d'+' -f1
}

cuda_to_tag() {
    case "$1" in
        11.7) echo "cu117" ;;
        11.8) echo "cu118" ;;
        12.1) echo "cu121" ;;
        12.4) echo "cu124" ;;
        12.6) echo "cu126" ;;
        12.8) echo "cu128" ;;
        12.9) echo "cu129" ;;
        *) echo "" ;;
    esac
}

pick_kaolin_cuda_tag() {
    torch_ver=$(normalize_torch_version "$1")
    cuda_tag=$(cuda_to_tag "$2")

    case "$torch_ver:$cuda_tag" in
        2.0.0:cu117|2.0.0:cu118|2.0.1:cu117|2.0.1:cu118) echo "$cuda_tag" ;;
        2.1.0:cu118|2.1.0:cu121|2.1.1:cu118|2.1.1:cu121|2.1.2:cu118|2.1.2:cu121) echo "$cuda_tag" ;;
        2.2.0:cu118|2.2.0:cu121|2.2.1:cu118|2.2.1:cu121) echo "$cuda_tag" ;;
        2.3.0:cu118|2.3.0:cu121|2.3.1:cu118|2.3.1:cu121) echo "$cuda_tag" ;;
        2.4.0:cu118|2.4.0:cu121|2.4.0:cu124|2.4.1:cu118|2.4.1:cu121|2.4.1:cu124) echo "$cuda_tag" ;;
        2.5.0:cu118|2.5.0:cu121|2.5.0:cu124|2.5.1:cu118|2.5.1:cu121|2.5.1:cu124) echo "$cuda_tag" ;;
        2.6.0:cu118|2.6.0:cu124|2.6.0:cu126) echo "$cuda_tag" ;;
        2.7.0:cu118|2.7.0:cu126|2.7.0:cu128|2.7.1:cu118|2.7.1:cu126|2.7.1:cu128) echo "$cuda_tag" ;;
        2.8.0:cu126|2.8.0:cu128|2.8.0:cu129) echo "$cuda_tag" ;;
        *) echo "" ;;
    esac
}

install_kaolin_from_source() {
    echo "[KAOLIN] No compatible prebuilt wheel found for PyTorch $PYTORCH_VERSION and CUDA $CUDA_VERSION."
    echo "[KAOLIN] Building Kaolin 0.18.0 from source with IGNORE_TORCH_VER=1."
    mkdir -p "$EXTENSION_PATH"
    rm -rf "$EXTENSION_PATH/kaolin"
    git clone --recursive https://github.com/NVIDIAGameWorks/kaolin.git "$EXTENSION_PATH/kaolin"
    cd "$EXTENSION_PATH/kaolin"
    git checkout v0.18.0
    git submodule update --init --recursive
    pip install -r tools/build_requirements.txt -r tools/viz_requirements.txt -r tools/requirements.txt
    IGNORE_TORCH_VER=1 pip install . --no-build-isolation
    cd "$WORKDIR"
}

if [ "$HELP" = true ] ; then
    echo "Usage: setup.sh [OPTIONS]"
    echo "Options:"
    echo "  -h, --help              Display this help message"
    echo "  --new-env               Create a new conda environment"
    echo "  --basic                 Install basic dependencies"
    echo "  --train                 Install training dependencies"
    echo "  --xformers              Install xformers"
    echo "  --flash-attn            Install flash-attn"
    echo "  --mipgaussianl1c        Install mip-splatting with l1-contribution"
    echo "  --kaolin                Install kaolin"
    echo "  --nvdiffrast            Install nvdiffrast"
    echo "  --demo                  Install all dependencies for demo"
    echo "  --extension-path PATH   Directory for cloned/built extension packages (default: /tmp/extensions)"
    return
fi

if [ "$NEW_ENV" = true ] ; then
    conda create -n deg python=3.10 -y
    conda activate deg
    CUDA_VERSION=$(detect_cuda_version)
    TORCH_INDEX_URL=$(pick_torch_cuda_index "$CUDA_VERSION")

    if [ -n "$CUDA_VERSION" ] ; then
        echo "[SYSTEM] Detected CUDA Version: $CUDA_VERSION"
    else
        echo "[SYSTEM] CUDA not detected, defaulting to cu126"
    fi

    echo "[SYSTEM] Installing PyTorch 2.9.1 from $TORCH_INDEX_URL"
    python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url "$TORCH_INDEX_URL"
fi

# Get system information
WORKDIR=$(pwd)
PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
PLATFORM=$(python -c "import torch; print(('cuda' if torch.version.cuda else ('hip' if torch.version.hip else 'unknown')) if torch.cuda.is_available() else 'cpu')")
case $PLATFORM in
    cuda)
        CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)")
        CUDA_MAJOR_VERSION=$(echo $CUDA_VERSION | cut -d'.' -f1)
        CUDA_MINOR_VERSION=$(echo $CUDA_VERSION | cut -d'.' -f2)
        echo "[SYSTEM] PyTorch Version: $PYTORCH_VERSION, CUDA Version: $CUDA_VERSION"
        ;;
    hip)
        HIP_VERSION=$(python -c "import torch; print(torch.version.hip)")
        HIP_MAJOR_VERSION=$(echo $HIP_VERSION | cut -d'.' -f1)
        HIP_MINOR_VERSION=$(echo $HIP_VERSION | cut -d'.' -f2)
        # Install pytorch 2.4.1 for hip
        if [ "$PYTORCH_VERSION" != "2.4.1+rocm6.1" ] ; then
            echo "[SYSTEM] Installing PyTorch 2.4.1 for HIP ($PYTORCH_VERSION -> 2.4.1+rocm6.1)"
            pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/rocm6.1 --user
            mkdir -p "$EXTENSION_PATH"
            cp /opt/rocm/share/amd_smi "$EXTENSION_PATH/amd_smi" -r
            cd "$EXTENSION_PATH/amd_smi"
            chmod -R 777 .
            pip install .
            cd "$WORKDIR"
            PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
        fi
        echo "[SYSTEM] PyTorch Version: $PYTORCH_VERSION, HIP Version: $HIP_VERSION"
        ;;
    *)
        ;;
esac

if [ "$BASIC" = true ] ; then
    pip install pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless scipy ninja rembg onnxruntime trimesh open3d xatlas pyvista pymeshfix igraph transformers diffusers accelerate pandas redis objaverse tensordict gpustat matplotlib einops POT
    pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
fi

if [ "$TRAIN" = true ] ; then
    pip install tensorboard pandas lpips wandb safetensors
fi

if [ "$XFORMERS" = true ] ; then
    # install xformers
    if [ "$PLATFORM" = "cuda" ] ; then
        if [ "$CUDA_VERSION" = "11.8" ] ; then
            case $PYTORCH_VERSION in
                2.0.1) pip install https://files.pythonhosted.org/packages/52/ca/82aeee5dcc24a3429ff5de65cc58ae9695f90f49fbba71755e7fab69a706/xformers-0.0.22-cp310-cp310-manylinux2014_x86_64.whl ;;
                2.1.0) pip install xformers==0.0.22.post7 --index-url https://download.pytorch.org/whl/cu118 ;;
                2.1.1) pip install xformers==0.0.23 --index-url https://download.pytorch.org/whl/cu118 ;;
                2.1.2) pip install xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu118 ;;
                2.2.0) pip install xformers==0.0.24 --index-url https://download.pytorch.org/whl/cu118 ;;
                2.2.1) pip install xformers==0.0.25 --index-url https://download.pytorch.org/whl/cu118 ;;
                2.2.2) pip install xformers==0.0.25.post1 --index-url https://download.pytorch.org/whl/cu118 ;;
                2.3.0) pip install xformers==0.0.26.post1 --index-url https://download.pytorch.org/whl/cu118 ;;
                2.4.0) pip install xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu118 ;;
                2.4.1) pip install xformers==0.0.28 --index-url https://download.pytorch.org/whl/cu118 ;;
                2.5.0) pip install xformers==0.0.28.post2 --index-url https://download.pytorch.org/whl/cu118 ;;
                *) echo "[XFORMERS] Unsupported PyTorch & CUDA version: $PYTORCH_VERSION & $CUDA_VERSION" ;;
            esac
        elif [ "$CUDA_VERSION" = "12.1" ] ; then
            case $PYTORCH_VERSION in
                2.1.0) pip install xformers==0.0.22.post7 --index-url https://download.pytorch.org/whl/cu121 ;;
                2.1.1) pip install xformers==0.0.23 --index-url https://download.pytorch.org/whl/cu121 ;;
                2.1.2) pip install xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu121 ;;
                2.2.0) pip install xformers==0.0.24 --index-url https://download.pytorch.org/whl/cu121 ;;
                2.2.1) pip install xformers==0.0.25 --index-url https://download.pytorch.org/whl/cu121 ;;
                2.2.2) pip install xformers==0.0.25.post1 --index-url https://download.pytorch.org/whl/cu121 ;;
                2.3.0) pip install xformers==0.0.26.post1 --index-url https://download.pytorch.org/whl/cu121 ;;
                2.4.0) pip install xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121 ;;
                2.4.1) pip install xformers==0.0.28 --index-url https://download.pytorch.org/whl/cu121 ;;
                2.5.0) pip install xformers==0.0.28.post2 --index-url https://download.pytorch.org/whl/cu121 ;;
                *) echo "[XFORMERS] Unsupported PyTorch & CUDA version: $PYTORCH_VERSION & $CUDA_VERSION" ;;
            esac
        elif [ "$CUDA_VERSION" = "12.4" ] ; then
            case $PYTORCH_VERSION in
                2.5.0) pip install xformers==0.0.28.post2 --index-url https://download.pytorch.org/whl/cu124 ;;
                *) echo "[XFORMERS] Unsupported PyTorch & CUDA version: $PYTORCH_VERSION & $CUDA_VERSION" ;;
            esac
        else
            echo "[XFORMERS] Unsupported CUDA version: $CUDA_MAJOR_VERSION"
        fi
    elif [ "$PLATFORM" = "hip" ] ; then
        case $PYTORCH_VERSION in
            2.4.1\+rocm6.1) pip install xformers==0.0.28 --index-url https://download.pytorch.org/whl/rocm6.1 ;;
            *) echo "[XFORMERS] Unsupported PyTorch version: $PYTORCH_VERSION" ;;
        esac
    else
        echo "[XFORMERS] Unsupported platform: $PLATFORM"
    fi
fi

if [ "$FLASHATTN" = true ] ; then
    if [ "$PLATFORM" = "cuda" ] ; then
        pip install flash-attn==2.7.3 --no-build-isolation
    elif [ "$PLATFORM" = "hip" ] ; then
        echo "[FLASHATTN] Prebuilt binaries not found. Building from source..."
        mkdir -p "$EXTENSION_PATH"
        git clone --recursive https://github.com/ROCm/flash-attention.git "$EXTENSION_PATH/flash-attention"
        cd "$EXTENSION_PATH/flash-attention"
        git checkout tags/v2.6.3-cktile
        GPU_ARCHS=gfx942 python setup.py install #MI300 series
        cd "$WORKDIR"
    else
        echo "[FLASHATTN] Unsupported platform: $PLATFORM"
    fi
fi

if [ "$KAOLIN" = true ] ; then
    if [ "$PLATFORM" = "cuda" ] ; then
        TORCH_BASE_VERSION=$(normalize_torch_version "$PYTORCH_VERSION")
        KAOLIN_CUDA_TAG=$(pick_kaolin_cuda_tag "$PYTORCH_VERSION" "$CUDA_VERSION")

        if [ -n "$KAOLIN_CUDA_TAG" ] ; then
            KAOLIN_INDEX_URL="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-${TORCH_BASE_VERSION}_${KAOLIN_CUDA_TAG}.html"
            echo "[KAOLIN] Installing kaolin==0.18.0 from $KAOLIN_INDEX_URL"
            pip install kaolin==0.18.0 -f "$KAOLIN_INDEX_URL"
        else
            install_kaolin_from_source
        fi
    else
        echo "[KAOLIN] Unsupported platform: $PLATFORM"
    fi
fi

if [ "$NVDIFFRAST" = true ] ; then
    if [ "$PLATFORM" = "cuda" ] ; then
        mkdir -p "$EXTENSION_PATH"
        rm -rf "$EXTENSION_PATH/nvdiffrast"
        git clone https://github.com/NVlabs/nvdiffrast.git "$EXTENSION_PATH/nvdiffrast"
        pip install "$EXTENSION_PATH/nvdiffrast" --no-build-isolation
    else
        echo "[NVDIFFRAST] Unsupported platform: $PLATFORM"
    fi
fi

if [ "$MIPGAUSSIANL1C" = true ] ; then
    if [ "$PLATFORM" = "cuda" ] ; then
        mkdir -p "$EXTENSION_PATH"
        rm -rf "$EXTENSION_PATH/mip-gaussian-l1c"
        git clone --recurse-submodules https://github.com/runjie-yan/mip-gaussian-l1c.git "$EXTENSION_PATH/mip-gaussian-l1c"
        pip install "$EXTENSION_PATH/mip-gaussian-l1c/submodules/diff-gaussian-rasterization/" --no-build-isolation
    else
        echo "[MIPGAUSSIANL1C] Unsupported platform: $PLATFORM"
    fi
fi

if [ "$DEMO" = true ] ; then
    pip install gradio==4.44.1 gradio_litmodel3d==0.0.1
fi
