import torch
from torch import nn
from typing import Optional, Tuple, List, Union, Callable
import numpy as np

# chunks to save memory
def get_chunks(
        inputs: torch.Tensor,
        chunksize: int = 2 ** 15
) -> List[torch.Tensor]:
    r"""
  Divide an input into chunks.
  """
    return [inputs[i:i + chunksize] for i in range(0, inputs.shape[0], chunksize)]


def prepare_chunks(
        points: torch.Tensor,
        # encoding_function: Callable[[torch.Tensor], torch.Tensor],
        chunksize: int = 2 ** 15
) -> List[torch.Tensor]:
    r"""
  Encode and chunkify points to prepare for NeRF model.
  """
    points = points.reshape((-1, points.shape[-1]))
    # points = encoding_function(points)
    points = get_chunks(points, chunksize=chunksize)
    return points

def get_rays(
        height: int,
        width: int,
        focal_length: float,
        c2w: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Find origin and direction of rays through every pixel and camera origin.

    # Apply pinhole camera model to gather directions at each pixel
    i, j = torch.meshgrid(
        torch.arange(width, dtype=torch.float32).to(c2w),
        torch.arange(height, dtype=torch.float32).to(c2w),
        indexing='ij')
    i, j = i.transpose(-1, -2), j.transpose(-1, -2)
    directions = torch.stack([(i - width * .5) / focal_length,
                              -(j - height * .5) / focal_length,
                              -torch.ones_like(i)
                              ], dim=-1)

    # Apply camera pose to directions
    rays_d = torch.sum(directions[..., None, :] * c2w[:3, :3], dim=-1)

    # Origin is same for all directions (the optical center)
    rays_o = c2w[:3, -1].expand(rays_d.shape)

    return rays_o, rays_d


def sample_stratified(
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        near: float,
        far: float,
        n_samples: int,
        perturb: Optional[bool] = True,
        inverse_depth: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
  Sample along ray from regularly-spaced bins.
  """

    # Grab samples for space integration along ray
    t_vals = torch.linspace(0., 1., n_samples, device=rays_o.device)
    if not inverse_depth:
        # Sample linearly between `near` and `far`
        z_vals = near * (1. - t_vals) + far * (t_vals)
    else:
        # Sample linearly in inverse depth (disparity)
        z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))

    # Draw uniform samples from bins along ray
    if perturb:
        mids = .5 * (z_vals[1:] + z_vals[:-1])
        upper = torch.concat([mids, z_vals[-1:]], dim=-1)
        lower = torch.concat([z_vals[:1], mids], dim=-1)
        t_rand = torch.rand([n_samples], device=z_vals.device)
        z_vals = lower + (upper - lower) * t_rand
    z_vals = z_vals.expand(list(rays_o.shape[:-1]) + [n_samples])

    # Apply scale from `rays_d` and offset from `rays_o` to samples
    # pts: (width, height, n_samples, 3)
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

    return pts, z_vals

def NeDF_forward(
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        near: float,
        far: float,
        model: nn.Module,
        chunksize: int = 2 ** 15,
        sample_num: int = 64,
) -> dict:
    r"""
    Compute forward pass through model(s).
    """

    # Sample query points along each ray.
    query_points, z_vals = sample_stratified(
        rays_o, rays_d, near, far, sample_num)
    # Prepare batches.
    batches = prepare_chunks(query_points, chunksize=chunksize)
    predictions = []
    for batch in batches:
        predictions.append(model(batch))
    raw = torch.cat(predictions, dim=0)
    raw = raw.reshape(list(query_points.shape[:2]) + [raw.shape[-1]])

    # Perform differentiable volume rendering to re-synthesize the RGB image.
    rgb_map, rgb_each_point = render_func(raw, z_vals, rays_d)

    outputs = {
        'rgb_map': rgb_map,
        'rgb_each_point': rgb_each_point,
        'query_points': query_points}

    # Store outputs.
    return outputs


def render_func(raw, z_vals, rays_d):
    # todo: sum version render
    return 0, 0
