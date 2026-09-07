import os
import sys
import torch
import yaml
from easydict import EasyDict as edict

# Add repo to path
sys.path.insert(0, "/kaggle/working/TripoSplat-Training-MV")
os.environ["ATTN_BACKEND"] = "sdpa" # Bypass flash_attn

from deg import models
from deg.utils.lora_utils import inject_lora

print(" Loading 1k config to save VRAM...")
with open("configs/dit/latent1k-latentseq_flow_img_s3dit-L.yaml", "r") as f:
    cfg = yaml.safe_load(f)

cfg = edict(cfg)
# Override for T4 VRAM safety
cfg.models.denoiser.args.use_fp16 = True
cfg.models.denoiser.args.ctrl_channels = 1280 # Match DINOv3 output

print(" Building Model...")
model = getattr(models, cfg.models.denoiser.name)(**cfg.models.denoiser.args).cuda()

print(" Injecting Control LoRA...")
# Freeze base
for p in model.parameters():
    p.requires_grad = False
# Inject LoRA into last 4 blocks
inject_lora(model, target_blocks=[20, 21, 22, 23], rank=16, alpha=1.0)
# Unfreeze ctrl_embedder
for p in model.ctrl_embedder.parameters():
    p.requires_grad = True

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f" Trainable parameters: {trainable_params:,} (Should be small!)")

print(" Generating Dummy Batch...")
B = 1
L = 1024

# ─── FIX: The model expects float32 inputs and handles fp16 conversion internally! ───
dummy_x = {
    'latent': torch.randn(B, L, 16, device='cuda', dtype=torch.float32),
    'camera': torch.randn(B, 1, 5, device='cuda', dtype=torch.float32)
}
from tensordict import TensorDict
dummy_x = TensorDict(dummy_x, batch_size=B)

dummy_cond = TensorDict({'feature1': torch.randn(B, 257, 1280, device='cuda', dtype=torch.float32)}, batch_size=B)
dummy_ctrl_tokens = torch.randn(B, 256, 1280, device='cuda', dtype=torch.float32)

t = torch.tensor([500.0], device='cuda', dtype=torch.float32)

print(" Running Forward & Backward Pass...")
model.train()

# Forward
out = model(dummy_x, t, dummy_cond, ctrl_tokens=dummy_ctrl_tokens)

# FIX: Cast to float32 before computing mean to avoid fp16 overflow/underflow
loss = out['latent'].float().mean() 

# Backward
loss.backward()

# Check gradients
lora_grad_exists = False
for name, p in model.named_parameters():
    if 'lora_A' in name and p.grad is not None and p.grad.abs().sum() > 0:
        lora_grad_exists = True
        break

if lora_grad_exists:
    print(" SUCCESS: LoRA gradients are flowing correctly!")
else:
    print(" FAIL: No gradients found in LoRA weights.")

# Test Saving
print(" Testing Checkpoint Save...")
from safetensors.torch import save_file
lora_state = {}
for name, param in model.named_parameters():
    if 'lora_A' in name or 'lora_B' in name or 'ctrl_embedder' in name:
        lora_state[name] = param.data.cpu().half()
save_file(lora_state, "dummy_lora_test.safetensors")
print(" Dummy LoRA saved to dummy_lora_test.safetensors")
