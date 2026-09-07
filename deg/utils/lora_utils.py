import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank=16, alpha=1.0):
        super().__init__()
        self.linear = linear
        self.rank = rank
        self.alpha = alpha
        
        # Freeze original linear layer
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False
            
        # LoRA parameters
        self.lora_A = nn.Parameter(torch.zeros(rank, linear.in_features))
        self.lora_B = nn.Parameter(torch.zeros(linear.out_features, rank))
        
        # Initialize
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return self.linear(x) + self.alpha * (x @ self.lora_A.T @ self.lora_B.T)

def inject_lora(model, target_blocks, rank=16, alpha=1.0):
    """Injects LoRA into the qkv and out projections of the specified transformer blocks."""
    for idx in target_blocks:
        block = model.blocks[idx]
        attn = block.attn
        if hasattr(attn, 'qkv'):
            attn.qkv = LoRALinear(attn.qkv, rank=rank, alpha=alpha)
        if hasattr(attn, 'out'):
            attn.out = LoRALinear(attn.out, rank=rank, alpha=alpha)