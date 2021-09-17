import numpy as np

import torch
import torch.nn.functional as F


if __name__ == "__main__":
    heatmap_npy_file = "demo_heatmap.npy"
    heatmap2_npy_file = "demo2_heatmap.npy"
    demo_heatmap = np.load(heatmap_npy_file)
    demo2_heatmap = np.load(heatmap2_npy_file)
    dist = np.linalg.norm(demo_heatmap - demo2_heatmap)
    print(dist)

    coord_npy_file = "demo_coord.npy"
    coord2_npy_file = "demo2_coord.npy"
    demo_coord = np.load(coord_npy_file)
    demo2_coord = np.load(coord2_npy_file)
    dist = np.linalg.norm(demo_coord - demo2_coord)
    print(dist)
