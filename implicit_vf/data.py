#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

# Copyright 2023-present Edoardo Mello Rella, Ajad Chhatkuli, Ender Konukoglu & Luc Van Gool

import glob
import logging
import numpy as np
import os
import random
import torch
import torch.utils.data

import implicit_vf.workspace as ws


class NoMeshFileError(RuntimeError):
    """Raised when a mesh file is not found in a shape directory"""

    pass


class MultipleMeshFileError(RuntimeError):
    """ "Raised when a there a multiple mesh files in a shape directory"""

    pass


def find_mesh_in_directory(shape_dir):
    mesh_filenames = list(glob.iglob(shape_dir + "/**/*.obj")) + list(
        glob.iglob(shape_dir + "/*.obj")
    )
    if len(mesh_filenames) == 0:
        raise NoMeshFileError()
    elif len(mesh_filenames) > 1:
        raise MultipleMeshFileError()
    return mesh_filenames[0]


def remove_nans(tensor):
    tensor_nan = torch.isnan(tensor.sum(dim=-1))
    return tensor[~tensor_nan, :]


def load_vf_subset(block, part):
    block = remove_nans(torch.from_numpy(block))
    indices = (torch.rand(part) * block.shape[0]).long()
    samples = torch.index_select(block, 0, indices)
    return samples


def unpack_open_vf_samples(filename, subsample=None):
    npz = np.load(filename)

    block_names = [
        "first",
        "second",
        "third",
        "fourth",
        "fifth",
        "sixth",
        "seventh",
        "eighth",
    ]
    sample_blocks = [
        load_vf_subset(npz[block_n], int(subsample / len(block_names)))
        for block_n in block_names
    ]

    samples = torch.cat(sample_blocks, 0).type(torch.FloatTensor)

    return samples


def read_open_vf_samples_into_ram(filename):
    npz = np.load(filename)

    first_set = torch.from_numpy(npz["first"])
    second_set = torch.from_numpy(npz["second"])
    third_set = torch.from_numpy(npz["third"])
    fourth_set = torch.from_numpy(npz["fourth"])
    fifth_set = torch.from_numpy(npz["fifth"])
    sixth_set = torch.from_numpy(npz["sixth"])
    seventh_set = torch.from_numpy(npz["seventh"])
    eighth_set = torch.from_numpy(npz["eighth"])

    return [
        first_set,
        second_set,
        third_set,
        fourth_set,
        fifth_set,
        sixth_set,
        seventh_set,
        eighth_set,
    ]


def unpack_open_vf_samples_from_ram(data, subsample=None):
    if subsample is None:
        return data

    data_tensor = torch.cat(data, 0)
    data_tensor = remove_nans(data_tensor)
    data_size = data_tensor.shape[0]
    start_ind = random.randint(0, data_size - subsample)
    sample_data = data_tensor[start_ind : (start_ind + subsample)]
    return sample_data.type(torch.FloatTensor)


def get_instance_filenames(data_source, split):
    npzfiles = []
    for dataset in split:
        for class_name in split[dataset]:
            for instance_name in split[dataset][class_name]:
                instance_filename = os.path.join(
                    dataset, class_name, instance_name + ".npz"
                )
                if not os.path.isfile(
                    os.path.join(data_source, ws.vf_samples_subdir, instance_filename)
                ):
                    logging.warning(
                        "Requested non-existent file '{}'".format(instance_filename)
                    )
                    continue
                npzfiles += [instance_filename]
    return npzfiles


class OpenVFSamples(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        split,
        subsample,
    ):
        self.subsample = subsample

        self.data_source = data_source
        self.npyfiles = get_instance_filenames(data_source, split)

        logging.debug(
            "using "
            + str(len(self.npyfiles))
            + " shapes from data source "
            + data_source
        )

    def __len__(self):
        return len(self.npyfiles)

    def __getitem__(self, idx):
        filename = os.path.join(
            self.data_source, ws.vf_samples_subdir, self.npyfiles[idx]
        )
        return (
            unpack_open_vf_samples(filename, self.subsample),
            idx,
        )
