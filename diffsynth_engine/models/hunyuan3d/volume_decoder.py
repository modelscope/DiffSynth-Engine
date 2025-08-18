from typing import Callable, List, Tuple, Union
from tqdm import tqdm
from einops import repeat
import numpy as np
import torch


def generate_dense_grid_points(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    octree_resolution: int,
    indexing: str = "ij",
):
    length = bbox_max - bbox_min
    num_cells = octree_resolution

    x = np.linspace(bbox_min[0], bbox_max[0], int(num_cells) + 1, dtype=np.float32)
    y = np.linspace(bbox_min[1], bbox_max[1], int(num_cells) + 1, dtype=np.float32)
    z = np.linspace(bbox_min[2], bbox_max[2], int(num_cells) + 1, dtype=np.float32)
    [xs, ys, zs] = np.meshgrid(x, y, z, indexing=indexing)
    xyz = np.stack((xs, ys, zs), axis=-1)
    grid_size = [int(num_cells) + 1, int(num_cells) + 1, int(num_cells) + 1]

    return xyz, grid_size, length


class VanillaVolumeDecoder:
    @torch.no_grad()
    def __call__(
        self,
        latents: torch.FloatTensor,
        geo_decoder: Callable,
        bounds: Union[Tuple[float], List[float], float] = 1.01,
        num_chunks: int = 10000,
        octree_resolution: int = None,
        enable_pbar: bool = True,
        **kwargs,
    ):
        device = latents.device
        dtype = latents.dtype
        batch_size = latents.shape[0]

        # 1. generate query points
        if isinstance(bounds, float):
            bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]

        bbox_min, bbox_max = np.array(bounds[0:3]), np.array(bounds[3:6])
        xyz_samples, grid_size, length = generate_dense_grid_points(
            bbox_min=bbox_min, bbox_max=bbox_max, octree_resolution=octree_resolution, indexing="ij"
        )
        xyz_samples = torch.from_numpy(xyz_samples).to(device, dtype=dtype).contiguous().reshape(-1, 3)

        # 2. latents to 3d volume
        batch_logits = []
        for start in tqdm(range(0, xyz_samples.shape[0], num_chunks), desc="Volume Decoding", disable=not enable_pbar):
            chunk_queries = xyz_samples[start : start + num_chunks, :]
            chunk_queries = repeat(chunk_queries, "p c -> b p c", b=batch_size)
            logits = geo_decoder(queries=chunk_queries, latents=latents)
            batch_logits.append(logits)

        grid_logits = torch.cat(batch_logits, dim=1)
        grid_logits = grid_logits.view((batch_size, *grid_size)).float()

        return grid_logits
