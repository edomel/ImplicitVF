#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

# Copyright 2023-present Edoardo Mello Rella, Ajad Chhatkuli, Ender Konukoglu & Luc Van Gool


import time
import torch
import implicit_vf
import plyfile

import torch.nn.functional as F
import numpy as np

from marching_cubes_utils.gifs_generation import contrastive_marching_cubes


def create_mesh_vf(
    decoder,
    latent_vec,
    filename,
    N=256,
    max_batch=32**3,
):
    start = time.time()
    ply_filename = filename

    decoder.eval()

    inc = np.array(
        [
            [0, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
            [1, 0, 1],
        ]
    )

    selected_indices = np.mgrid[: int(N / 2), : int(N / 2), : int(N / 2)]
    selected_indices = np.moveaxis(selected_indices, 0, -1).reshape(-1, 3)
    selected_indices = (selected_indices[:, None] * 2 + inc[None]).reshape(-1, 3)

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)

    overall_index = torch.arange(0, N**3, 1, out=torch.LongTensor())
    samples = torch.zeros(N**3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() // N) % N
    samples[:, 0] = ((overall_index.long() // N) // N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    prediction = implicit_vf.utils.get_set_predictions(
        decoder, latent_vec, samples, max_batch
    )
    divergence_values = implicit_vf.utils.extract_divergence(prediction, N)

    norms = torch.norm(prediction.clone(), dim=1)
    vf_values = F.normalize(prediction, dim=1).reshape(N, N, N, 3)

    chosen_direction = implicit_vf.utils.unify_direction(
        divergence_values, vf_values.permute(3, 0, 1, 2), N=N
    )
    comb_values, norms = implicit_vf.utils.make_comb_format(chosen_direction, norms, N)

    comb_values = comb_values.reshape(N, N, N, 28)
    norms = norms.reshape(N, N, N, 28, 2)
    comb_values = comb_values[
        selected_indices[:, 0], selected_indices[:, 1], selected_indices[:, 2]
    ].reshape(N, N, N, 28)
    norms = norms[
        selected_indices[:, 0], selected_indices[:, 1], selected_indices[:, 2]
    ]

    udf = norms.clone().cpu().numpy()
    comb_values = comb_values.clone().cpu().numpy().reshape(-1, 28)
    mask = comb_values.sum(-1)
    selected_indices = selected_indices[mask > 0]
    udf = udf[mask > 0].reshape(-1, 2)
    comb_values = comb_values[mask > 0].reshape(-1)

    vs, fs = contrastive_marching_cubes(
        comb_values,
        isovalue=0.0,
        size=2.0,
        selected_indices=selected_indices,
        res=N,
        udf=udf,
    )

    vs, fs = np.array(list(vs.keys())), np.array(fs) - 1
    num_verts = vs.shape[0]
    num_faces = fs.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(vs[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((fs[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])

    ply_data.write(ply_filename + ".ply")

    end = time.time()
    print("Generation takes: %f" % (end - start))
    return
