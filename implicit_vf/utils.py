#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

# Copyright 2023-present Edoardo Mello Rella, Ajad Chhatkuli, Ender Konukoglu & Luc Van Gool

import logging
import torch
import trimesh
import math

import torch.nn.functional as F
import numpy as np


def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(
                    trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()
                )
            )
    else:
        assert isinstance(scene_or_mesh, trimesh.Trimesh)
        mesh = trimesh.Trimesh(
            vertices=scene_or_mesh.vertices, faces=scene_or_mesh.faces
        )
    return mesh


def add_common_args(arg_parser):
    arg_parser.add_argument(
        "--debug",
        dest="debug",
        default=False,
        action="store_true",
        help="If set, debugging messages will be printed",
    )
    arg_parser.add_argument(
        "--quiet",
        "-q",
        dest="quiet",
        default=False,
        action="store_true",
        help="If set, only warnings will be printed",
    )
    arg_parser.add_argument(
        "--log",
        dest="logfile",
        default=None,
        help="If set, the log will be saved using the specified filename.",
    )


def configure_logging(args):
    logger = logging.getLogger()
    if args.debug:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.WARNING)
    else:
        logger.setLevel(logging.INFO)
    logger_handler = logging.StreamHandler()
    formatter = logging.Formatter("ImplicitVF - %(levelname)s - %(message)s")
    logger_handler.setFormatter(formatter)
    logger.addHandler(logger_handler)

    if args.logfile is not None:
        file_logger_handler = logging.FileHandler(args.logfile)
        file_logger_handler.setFormatter(formatter)
        logger.addHandler(file_logger_handler)


def get_set_predictions(decoder, latent_vec, samples, max_batch):

    samples.requires_grad = False
    num_samples = samples.shape[0]
    predicions = torch.zeros_like(samples)

    head = 0

    while head < num_samples:
        sample_subset = samples[head : min(head + max_batch, num_samples)]
        latent_repeat = latent_vec.expand(sample_subset.shape[0], -1)
        inputs = torch.cat([latent_repeat.cuda(), sample_subset.cuda()], 1).cuda()

        predicions[head : min(head + max_batch, num_samples)] = (
            decoder(inputs).detach().cpu()
        )
        head += max_batch

    return predicions


def extract_divergence(vf_values, N):
    # Threshold set to determine which voxel blocks have a surface
    threshold = -0.7

    internal_vf = (
        F.normalize(vf_values.clone(), dim=1).reshape(N, N, N, 3).permute(3, 0, 1, 2)
    )

    N_box = 2
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N_box - 1)
    overall_index = torch.arange(0, N_box**3, 1, out=torch.LongTensor())
    filter = torch.zeros(N_box**3, 3)
    filter[:, 2] = overall_index % N_box
    filter[:, 1] = (overall_index.long() // N_box) % N_box
    filter[:, 0] = ((overall_index.long() // N_box) // N_box) % N_box
    filter[:, 0] = (filter[:, 0] * voxel_size) + voxel_origin[2]
    filter[:, 1] = (filter[:, 1] * voxel_size) + voxel_origin[1]
    filter[:, 2] = (filter[:, 2] * voxel_size) + voxel_origin[0]
    filter = (
        F.normalize(filter, dim=1)
        .reshape(N_box, N_box, N_box, 3)
        .permute(3, 0, 1, 2)
        .unsqueeze(0)
    ).to(vf_values.device)
    scatter_indices = (
        torch.arange(N_box**3)
        .reshape(N_box, N_box, N_box)
        .unsqueeze(0)
        .unsqueeze(0)
        .repeat_interleave(3, dim=1)
    ).to(vf_values.device)
    complete_filter = torch.zeros_like(filter).repeat_interleave(N_box**3, dim=0)
    complete_filter.scatter_(dim=0, index=scatter_indices, src=filter)
    face_area = math.sqrt(3.0) / 4.0
    shape_area = 2 * math.sqrt(2.0) / 3.0

    divergence = F.conv3d(internal_vf.unsqueeze(0), complete_filter).squeeze()
    divergence = (divergence * torch.abs(divergence) * face_area).sum(
        dim=0
    ) / shape_area

    divergence_values = torch.zeros((N, N, N)).type_as(divergence).to(vf_values.device)
    divergence_values[:-1, :-1, :-1] += divergence

    divergence_values[divergence_values > threshold] = 0
    divergence_values[divergence_values <= threshold] = 1

    return divergence_values


def make_comb_format(choice_side, norms, N):
    n_reduction = 2

    selection_filter = torch.zeros(
        (n_reduction**3, 1, n_reduction, n_reduction, n_reduction)
    ).type_as(norms)
    selection_filter[0, 0, 0, 0, 0] = 1
    selection_filter[1, 0, 0, 1, 0] = 1
    selection_filter[2, 0, 1, 1, 0] = 1
    selection_filter[3, 0, 1, 0, 0] = 1
    selection_filter[4, 0, 0, 0, 1] = 1
    selection_filter[5, 0, 0, 1, 1] = 1
    selection_filter[6, 0, 1, 1, 1] = 1
    selection_filter[7, 0, 1, 0, 1] = 1

    norms = F.conv3d(
        norms.clone().reshape(N, N, N).unsqueeze(0).unsqueeze(0),
        selection_filter,
        padding=1,
    )
    norms = norms.permute(2, 3, 4, 1, 0)[1:, 1:, 1:].reshape(N**3, 8)

    different_side = torch.zeros((N**3, 28)).type_as(norms)
    different_side_norms = torch.zeros((N**3, 28, 2)).type_as(norms)

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

    combs = []
    comb_to_idx = [0] * 64
    dist = [0] * 64
    for i in range(7):
        for j in range(i + 1, 8):
            comb_to_idx[i * 8 + j] = len(combs)
            dist[i * 8 + j] = np.linalg.norm(inc[i] - inc[j])
            combs.append([i, j])

    for i, indices in enumerate(combs):
        different_side[:, i] = choice_side[:, indices[0]] != choice_side[:, indices[1]]
        different_side_norms[:, i, 0] = norms[:, indices[0]]
        different_side_norms[:, i, 1] = norms[:, indices[1]]

    return different_side, different_side_norms


def unify_direction(divergence_grid, vf_grid, N=64):

    n_reduction = 2

    selection_filter = torch.zeros(
        (n_reduction**3, 1, n_reduction, n_reduction, n_reduction)
    ).type_as(vf_grid)
    selection_filter[0, 0, 0, 0, 0] = 1
    selection_filter[1, 0, 0, 1, 0] = 1
    selection_filter[2, 0, 1, 1, 0] = 1
    selection_filter[3, 0, 1, 0, 0] = 1
    selection_filter[4, 0, 0, 0, 1] = 1
    selection_filter[5, 0, 0, 1, 1] = 1
    selection_filter[6, 0, 1, 1, 1] = 1
    selection_filter[7, 0, 1, 0, 1] = 1

    temp_vf_grid = F.conv3d(
        vf_grid.clone().unsqueeze(1),
        selection_filter,
        padding=1,
    )
    temp_vf_grid = temp_vf_grid.permute(2, 3, 4, 1, 0)[1:, 1:, 1:]
    surface_vf = temp_vf_grid[divergence_grid == 1]
    distance_matrix = 1.0 - (
        torch.bmm(surface_vf[:, :, 0].unsqueeze(-1), surface_vf[:, :, 0].unsqueeze(-2))
        + torch.bmm(
            surface_vf[:, :, 1].unsqueeze(-1), surface_vf[:, :, 1].unsqueeze(-2)
        )
        + torch.bmm(
            surface_vf[:, :, 2].unsqueeze(-1), surface_vf[:, :, 2].unsqueeze(-2)
        )
    ).reshape(surface_vf.shape[0], n_reduction**6)
    extreme_indices = torch.argmax(distance_matrix, dim=-1)
    first = (
        (extreme_indices // (n_reduction**3))
        .unsqueeze(-1)
        .unsqueeze(-1)
        .repeat_interleave(3, dim=-1)
    )
    second = (
        (extreme_indices % (n_reduction**3))
        .unsqueeze(-1)
        .unsqueeze(-1)
        .repeat_interleave(3, dim=-1)
    )
    first_vec = torch.gather(surface_vf, dim=1, index=first)
    second_vec = torch.gather(surface_vf, dim=1, index=second)

    first_vec = first_vec.repeat_interleave(n_reduction**3, dim=1)
    second_vec = second_vec.repeat_interleave(n_reduction**3, dim=1)

    first_distance = torch.norm(first_vec - surface_vf, dim=-1)
    second_distance = torch.norm(second_vec - surface_vf, dim=-1)
    choice = torch.argmin(
        torch.stack((first_distance, second_distance), dim=-1), dim=-1
    )

    direction_choice = torch.zeros((N, N, N, 8)).type_as(choice)
    direction_choice[divergence_grid == 1] = choice

    return direction_choice.reshape(-1, 8)
