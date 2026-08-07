import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from lpips import LPIPS
import einops
from tensordict import TensorDict

def general_mse_loss(pred, target, reduction='mean', mse_lambda={}):
    if isinstance(pred, torch.Tensor):
        return F.mse_loss(pred, target, reduction=reduction), {}
    elif isinstance(pred, TensorDict) or isinstance(pred, dict):
        loss_dict = {}
        total_loss = 0.0
        total_numel = 0
        for key in pred.keys():
            weight = mse_lambda.get(key, 1.0)
            loss_dict[key] = F.mse_loss(pred[key], target[key], reduction=reduction)
            
            # Always compute SSE for aggregation to handle proper weighting
            sse = F.mse_loss(pred[key], target[key], reduction='sum')
            total_loss = total_loss + weight * sse
            total_numel += pred[key].numel()

        if reduction == 'mean':
            loss = total_loss / max(total_numel, 1)
        else:
            loss = total_loss
        return loss, loss_dict

def smooth_l1_loss(pred, target, beta=1.0):
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    return loss.mean()


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def psnr(img1, img2, max_val=1.0, reduction="mean"):
    mse = F.mse_loss(img1, img2, reduction='none')
    mse = einops.reduce(mse, "B ... -> B", reduction="mean")
    psnr = 10 * torch.log10(max_val**2 / mse)
    psnr = einops.reduce(psnr, "B ->", reduction=reduction)
    return psnr


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def f_score_voxels(pred_grid: torch.Tensor, gt_grid: torch.Tensor, reduction: str = 'mean') -> float:
    # Compute true positives (TP): elements where both pred and gt are occupied.
    tp = einops.reduce((pred_grid & gt_grid).float(), "B ... -> B", reduction="sum")

    # Compute the total predicted positive elements.
    pred_sum = einops.reduce(pred_grid.float(), "B ... -> B", reduction="sum")
    
    # Compute the total ground truth positive elements.
    gt_sum = einops.reduce(gt_grid.float(), "B ... -> B", reduction="sum")

    # Calculate precision and recall; guard against division by zero.
    precision = tp / pred_sum
    recall = tp / gt_sum
    precision[pred_sum==0] = 0.0
    recall[gt_sum==0] = 0.0
    

    # Compute F1-score; handle the case where precision and recall are both zero.
    fscore = 2 * precision * recall / (precision + recall)
    fscore[precision + recall == 0] = 0.0

    if reduction == 'mean':
        fscore = fscore.mean()
    elif reduction == 'sum':
        fscore = fscore.sum()
    elif reduction == 'none':
        pass
    return fscore

def extract_surface_points(grid):
    """
    Extracts surface points (value == 1) from a binary (B, H, W, D) tensor.
    Returns a list of (Ni, 3) point clouds (in float32) for each batch.
    """
    B, H, W, D = grid.shape
    grid_size = max(H, W, D)
    point_clouds = []

    for b in range(B):
        coords = torch.nonzero(grid[b], as_tuple=False).float() / grid_size # (Ni, 3)
        point_clouds.append(coords)

    return point_clouds

def chamfer_distance_single(pc1, pc2, training=False):
    """
    Computes Chamfer Distance between two point clouds of shape (N1, 3) and (N2, 3).
    """
    N1, N2 = pc1.shape[0], pc2.shape[0]

    if N1 == 0 or N2 == 0:
        return torch.tensor(1.0).to(pc1.device)

    # (N1, N2, 3) -> (N1, N2)
    if training:
        dist_matrix = torch.cdist(pc1, pc2, p=2, compute_mode='donot_use_mm_for_euclid_dist') ** 2  # squared L2
    else:
        dist_matrix = torch.cdist(pc1, pc2, p=2, compute_mode='donot_use_mm_for_euclid_dist')  # L2

    cd1 = dist_matrix.min(dim=1)[0].mean()  # from pc1 to pc2
    cd2 = dist_matrix.min(dim=0)[0].mean()  # from pc2 to pc1

    return cd1 + cd2

def chamfer_distance(pcs1, pcs2, reduction='mean', training=False):
    """
    Compute Chamfer Distance between two lists of point clouds.
    Each list contains B point clouds of shape (Ni, 3).
    Returns a tensor of shape (B,) with Chamfer Distances.
    """
    dists = []
    for pc1, pc2 in zip(pcs1, pcs2):
        d = chamfer_distance_single(pc1, pc2, training=training)
        dists.append(d)

    dists = torch.stack(dists)  # (B,)
    if reduction == 'mean':
        return dists.mean()
    elif reduction == 'sum':
        return dists.sum()
    elif reduction == 'none':
        return dists
    else:
        raise ValueError(f"Unknown reduction mode: {reduction}")


def chamfer_distance_voxel(grid1, grid2, reduction='mean'):
    """
    Compute Chamfer Distance between two (B, H, W, D) binary grids.
    Returns a tensor of shape (B,) with Chamfer Distances.
    """
    pcs1 = extract_surface_points(grid1)
    pcs2 = extract_surface_points(grid2)

    dists = []
    for pc1, pc2 in zip(pcs1, pcs2):
        d = chamfer_distance_single(pc1, pc2)
        dists.append(d)

    dists = torch.stack(dists)  # (B,)
    if reduction == 'mean':
        return dists.mean()
    elif reduction == 'sum':
        return dists.sum()
    elif reduction == 'none':
        return dists
    else:
        raise ValueError(f"Unknown reduction mode: {reduction}")
    
def fill_surface_voxel_grid(grid):
    """
    Takes a binary (B, H, W, D) voxel grid with surface voxels (value==1) and fills inside.
    Returns a solid binary voxel grid where the object is concrete.
    """

    B, H, W, D = grid.shape

    grid = grid.float()
    # Step 1: Invert grid: 1 (surface) -> 0, 0 (potential fill) -> 1
    inv_grid = 1 - grid

    # Step 2: Create mask for flood fill starting points (edges)
    fill_mask = torch.zeros_like(inv_grid)

    fill_mask[:,  0, :, :] = 1
    fill_mask[:, -1, :, :] = 1
    fill_mask[:, :,  0, :] = 1
    fill_mask[:, :, -1, :] = 1
    fill_mask[:, :, :,  0] = 1
    fill_mask[:, :, :, -1] = 1

    # Only consider empty voxels on the border
    queue = fill_mask * inv_grid

    # Step 3: Flood fill from the outside using 6-connected neighbors
    kernel = torch.zeros((1, 1, 3, 3, 3), device=grid.device)
    kernel[0, 0, 1, 1, 0] = 1
    kernel[0, 0, 1, 1, 2] = 1
    kernel[0, 0, 1, 0, 1] = 1
    kernel[0, 0, 1, 2, 1] = 1
    kernel[0, 0, 0, 1, 1] = 1
    kernel[0, 0, 2, 1, 1] = 1

    filled = queue.clone()
    while True:
        # 3D dilation to expand the filled region
        expanded = F.conv3d(filled.unsqueeze(1), kernel, padding=1).clamp_max(1)
        expanded = (expanded.squeeze(1) > 0).float() * inv_grid  # only spread inside empty space

        if torch.all(expanded <= filled):
            break
        filled = torch.max(filled, expanded)

    # Step 4: Invert the result: inside = 1, outside = 0
    solid = 1 - filled

    return solid.bool()

loss_fn_vgg = None
def lpips(img1, img2, value_range=(0, 1), reduction='mean'):
    global loss_fn_vgg
    if loss_fn_vgg is None:
        loss_fn_vgg = LPIPS(net='vgg').cuda().eval()
    # normalize to [-1, 1]
    img1 = (img1 - value_range[0]) / (value_range[1] - value_range[0]) * 2 - 1
    img2 = (img2 - value_range[0]) / (value_range[1] - value_range[0]) * 2 - 1
    loss = loss_fn_vgg(img1, img2)
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise ValueError(f"Unknown reduction mode: {reduction}")


def normal_angle(pred, gt):
    pred = pred * 2.0 - 1.0
    gt = gt * 2.0 - 1.0
    norms = pred.norm(dim=-1) * gt.norm(dim=-1)
    cos_sim = (pred * gt).sum(-1) / (norms + 1e-9)
    cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
    ang = torch.rad2deg(torch.acos(cos_sim[norms > 1e-9])).mean()
    if ang.isnan():
        return -1
    return ang