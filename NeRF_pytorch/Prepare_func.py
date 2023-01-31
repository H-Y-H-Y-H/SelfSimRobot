import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import os
from typing import Optional, Tuple, List, Union, Callable


def get_rays(
        height: int,
        width: int,
        focal_length: float,
        c2w: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
  Find origin and direction of rays through every pixel and camera origin.
  """

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


