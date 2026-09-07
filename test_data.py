import os
import sys
import torch
import yaml
from easydict import EasyDict as edict
from tensordict import TensorDict
from safetensors.torch import save_file

sys.path.insert(0, "/kaggle/working/TripoSplat-Training-MV")
os.environ["ATTN_BACKEND"] = "sdpa"

from deg import models
from deg.utils.lora_utils import inject_lora

print(" Loading 1k config...")
with open("/kaggle/working/TripoSplat-Training-MV/configs/dit/latent1k-latentseq_flow_img_s3dit-L.yaml", "r") as f:
    cfg = yaml.safe_load(f)

cfg = edict(cfg)
cfg.models.denoiser.args.use_fp16 = True
cfg.models.denoiser.args.ctrl_channels = 1280

print(" Building Model...")
model = getattr(models, cfg.models.denoiser.name)(**cfg.models.denoiser.args).cuda()

print(" Bypassing ALL zero-init barriers for gradient test...")
# 1. out_layer
model.out_layer.weight.data.normal_(std=0.02)

# 2. Shared adaLN modulation (CRUCIAL FIX: The 1k config uses share_mod=True)
if hasattr(model, 'adaLN_modulation') and model.adaLN_modulation is not None:
    model.adaLN_modulation[-1].weight.data.normal_(std=0.02)
    model.adaLN_modulation[-1].bias.data.normal_(std=0.02)

# 3. Block-level adaLN (Fallback just in case share_mod=False)
for idx in [20, 21, 22, 23]:
    block = model.blocks[idx]
    if hasattr(block, 'adaLN_modulation') and block.adaLN_modulation is not None:
        block.adaLN_modulation[-1].weight.data.normal_(std=0.02)
        block.adaLN_modulation[-1].bias.data.normal_(std=0.02)

# 4. ctrl_embedder
if hasattr(model, 'ctrl_embedder') and model.ctrl_embedder is not None:
    model.ctrl_embedder.weight.data.normal_(std=0.02)

print(" Injecting Control LoRA...")
for p in model.parameters():
    p.requires_grad = False

inject_lora(model, target_blocks=[20, 21, 22, 23], rank=16, alpha=1.0)

# Inject random noise into lora_B so lora_A can receive gradients too
for name, p in model.named_parameters():
    if 'lora_B' in name:
        p.data.normal_(std=0.01)

for p in model.ctrl_embedder.parameters():
    p.requires_grad = True

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f" Trainable parameters: {trainable_params:,}")

print(" Generating Dummy Batch...")
B = 1
L = 1024

dummy_x = TensorDict({
    'latent': torch.randn(B, L, 16, device='cuda', dtype=torch.float32),
    'camera': torch.randn(B, 1, 5, device='cuda', dtype=torch.float32)
}, batch_size=B)

dummy_cond = TensorDict({
    'feature1': torch.randn(B, 257, 1280, device='cuda', dtype=torch.float32),
    'feature2': torch.randn(B, 257, 128, device='cuda', dtype=torch.float32)
}, batch_size=B)

dummy_ctrl_tokens = torch.randn(B, 256, 1280, device='cuda', dtype=torch.float32)
t = torch.tensor([500.0], device='cuda', dtype=torch.float32)

print(" Running Forward & Backward Pass...")
model.train()

out = model(dummy_x, t, dummy_cond, ctrl_tokens=dummy_ctrl_tokens)
loss = out['latent'].float().mean()
loss.backward()

print("\n Gradient Report:")
lora_A_ok = False
lora_B_ok = False
ctrl_ok = False

for name, p in model.named_parameters():
    if not p.requires_grad:
        continue
    has_grad = p.grad is not None and p.grad.abs().sum() > 0
    
    if 'lora_A' in name:
        lora_A_ok = lora_A_ok or has_grad
    elif 'lora_B' in name:
        lora_B_ok = lora_B_ok or has_grad
    elif 'ctrl_embedder' in name:
        ctrl_ok = ctrl_ok or has_grad

print("\n" + "="*50)
if lora_A_ok and lora_B_ok:
    print(" SUCCESS: LoRA gradients flowing correctly!")
else:
    print(f" lora_A grads: {lora_A_ok}, lora_B grads: {lora_B_ok}, ctrl grads: {ctrl_ok}")

if not ctrl_ok and hasattr(model, 'ctrl_embedder'):
    print(" NOTE: ctrl_embedder has no gradients. This is expected if you haven't patched the `forward` method to use `ctrl_tokens` yet.")

print(" Saving checkpoint...")
lora_state = {}
for name, param in model.named_parameters():
    if 'lora_A' in name or 'lora_B' in name or 'ctrl_embedder' in name:
        lora_state[name] = param.data.cpu().half()
save_file(lora_state, "dummy_lora_test.safetensors")
print(" Saved to dummy_lora_test.safetensors")
