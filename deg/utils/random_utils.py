import numpy as np
import torch
from typing import *

PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]

def radical_inverse(base, n):
    val = 0
    inv_base = 1.0 / base
    inv_base_n = inv_base
    while n > 0:
        digit = n % base
        val += digit * inv_base_n
        n //= base
        inv_base_n *= inv_base
    return val

def halton_sequence(dim, n):
    return [radical_inverse(PRIMES[dim], n) for dim in range(dim)]

def hammersley_sequence(dim, n, num_samples):
    return [n / num_samples] + halton_sequence(dim - 1, n)

def sphere_hammersley_sequence(n, num_samples, offset=(0, 0), remap=False):
    u, v = hammersley_sequence(2, n, num_samples)
    u += offset[0] / num_samples
    v += offset[1]
    if remap:
        u = 2 * u if u < 0.25 else 2 / 3 * u + 1 / 3
    theta = np.arccos(1 - 2 * u) - np.pi / 2
    phi = v * 2 * np.pi
    return [phi, theta]



@torch.no_grad()
def sample_probs(
    probs: torch.Tensor,
    counts: torch.Tensor,
    algo: Literal["iid", "residual", "systematic"] = "systematic",
) -> torch.Tensor:
    """
    probs  : [..., P] nonnegative; rows need not be normalized
    counts : [...] nonnegative ints
    returns: [..., P] long tensor of counts per bin
    handle properly for conunts = 0 case
    """
    batch_shape = counts.shape
    B = counts.numel()
    P = probs.size(-1)
    assert B * P == probs.numel() and probs.dim() - counts.dim() == 1
    device = probs.device
    probs = probs.view(B, P)
    counts = counts.view(B)

    # Normalize with uniform fallback for zero-sum rows
    probs = probs.to(torch.float32).clamp_min_(0)
    row_sums = probs.sum(1, keepdim=True)
    zero_mask = row_sums.eq(0)
    probs = probs / row_sums.clamp_min_(1)
    if zero_mask.any():
        probs = probs.clone()
        probs[zero_mask.expand_as(probs)] = 1.0 / P

    counts = counts.to(device=device, dtype=torch.long)
    out = torch.zeros(B, P, dtype=torch.long, device=device)

    if algo == "iid":
        # Pure multinomial IID, batched by unique k
        unique_k, inv = counts.unique(sorted=False, return_inverse=True)
        for i, k in enumerate(unique_k.tolist()):
            if k == 0:
                continue
            rows = (inv == i).nonzero(as_tuple=False).squeeze(1)  # indices for this k
            samples = torch.multinomial(
                probs.index_select(0, rows), num_samples=int(k), replacement=True
            )
            buf = torch.zeros(rows.numel(), P, dtype=torch.float32, device=device)
            buf.scatter_add_(1, samples, torch.ones_like(samples, dtype=buf.dtype))
            out.index_copy_(0, rows, buf.to(torch.long))

    elif algo in ("residual", "owen_scrambled_residual"):
        # Step 1: deterministic floor allocation (vectorized)
        expected = probs * counts.unsqueeze(1).to(torch.float32)  # [B, P]
        base = torch.floor(expected)  # [B, P]
        out += base.to(torch.long)

        # Step 2: compute residuals
        residual_mass = expected - base  # [B, P]
        m = counts - base.sum(1).to(torch.long)  # [B]

        # Guard: skip rows with no residuals
        active_mask = m > 0
        if active_mask.any():
            rows = active_mask.nonzero(as_tuple=False).squeeze(1)
            m_active = m[rows]
            res_probs = residual_mass[rows]
            res_probs /= res_probs.sum(1, keepdim=True)  # normalize residuals

            if algo == "residual":
                # --- Batched multinomial ---
                unique_m, invm = m_active.unique(sorted=False, return_inverse=True)
                for j, mm in enumerate(unique_m.tolist()):
                    rows_m = (invm == j).nonzero(as_tuple=False).squeeze(1)  # indices for this m
                    rsel = rows[rows_m]
                    if mm == 0:
                        continue
                    samples = torch.multinomial(
                        res_probs.index_select(0, rows_m),
                        num_samples=int(mm),
                        replacement=True,
                    )
                    buf = torch.zeros(rows_m.numel(), P, dtype=torch.float32, device=device)
                    buf.scatter_add_(1, samples, torch.ones_like(samples, dtype=buf.dtype))
                    out.index_copy_(0, rsel, out[rsel] + buf.to(torch.long))
            else:
                raise NotImplementedError("Owen-scrambled residual not implemented yet")
        assert out.sum() == counts.sum(), f"sampling count mismatch, input probs max: {probs.max().item()}, counts: {counts.sum().item()}, output sum: {out.sum().item()}"

    elif algo == "systematic":
        cdf = probs.cumsum(dim=1).clamp(max=1.0 - 1e-12)
        unique_n, inv = counts.unique(sorted=False, return_inverse=True)
        for i, n in enumerate(unique_n.tolist()):
            if n == 0:
                continue
            rows = (inv == i).nonzero(as_tuple=False).squeeze(1)
            r = rows.numel()

            # random offset per row in [0,1/n)
            U0 = torch.rand(r, 1, device=device) / float(n)                 # [r,1]
            # stratified points
            grid = torch.arange(n, device=device, dtype=torch.float32)[None, :] / float(n)  # [1,n]
            us = (U0 + grid).clamp(max=1.0 - 1e-12)                         # [r,n]

            cdf_rows = cdf.index_select(0, rows)                             # [r,P]
            idx = torch.searchsorted(cdf_rows, us).clamp_max(probs.size(1) - 1)                           # [r,n]

            buf = torch.zeros(r, P, dtype=torch.float32, device=device)
            buf.scatter_add_(1, idx, torch.ones_like(idx, dtype=buf.dtype))
            out.index_copy_(0, rows, buf.to(torch.long))

    else:
        raise ValueError(f"Unknown algo {algo}")
    
    out = out.view(*batch_shape, P)
    return out
