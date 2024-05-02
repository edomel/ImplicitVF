#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

# Copyright 2023-present Edoardo Mello Rella, Ajad Chhatkuli, Ender Konukoglu & Luc Van Gool

import argparse
import json
import logging
import os
import torch
import implicit_vf

import implicit_vf.workspace as ws

from torch.utils.data import DataLoader
from implicit_vf.reconstruction_utils import optimize_and_reconstruct
from implicit_vf.decoder import Decoder


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(
        description="Use a trained ImplicitVF autodecoder to reconstruct new shapes given VF samples"
    )
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory which includes specifications and saved model "
        + "files to use for reconstruction",
    )
    arg_parser.add_argument(
        "--checkpoint",
        "-c",
        dest="checkpoint",
        default="latest",
        help="The checkpoint weights to use. This can be a number indicated an epoch "
        + "or 'latest' for the latest weights (this is the default)",
    )
    arg_parser.add_argument(
        "--grid_size",
        default=256,
        type=int,
        help="grid size for MC mesh reconstruction",
    )
    implicit_vf.add_common_args(arg_parser)
    args = arg_parser.parse_args()
    implicit_vf.configure_logging(args)

    specs_filename = os.path.join(args.experiment_directory, "specs.json")

    if not os.path.isfile(specs_filename):
        raise Exception(
            'The experiment directory does not include specifications file "specs.json"'
        )

    specs = json.load(open(specs_filename))

    latent_size = specs["CodeLength"]
    data_source = specs["DataSource"]
    test_split_file = specs["TestSplit"]

    decoder = Decoder(
        latent_size,
        **specs["NetworkSpecs"],
    ).cuda()

    decoder = torch.nn.DataParallel(decoder)

    saved_model_state = torch.load(
        os.path.join(
            args.experiment_directory,
            ws.model_params_subdir,
            args.checkpoint + ".pth",
        )
    )
    saved_model_epoch = saved_model_state["epoch"]

    decoder.load_state_dict(saved_model_state["model_state_dict"])
    decoder = decoder.module.cuda()

    with open(test_split_file, "r") as f:
        split = json.load(f)

    npz_filenames = implicit_vf.data.get_instance_filenames(data_source, split)
    print("Number of files to be used:", len(npz_filenames))

    logging.debug(decoder)

    try:
        reconstruction_dir = os.path.join(
            args.experiment_directory,
            ws.reconstructions_subdir,
            str(saved_model_epoch),
        )
    except:
        reconstruction_dir = os.path.join(
            args.experiment_directory,
            ws.reconstructions_subdir,
            str(args.checkpoint),
        )

    if not os.path.isdir(reconstruction_dir):
        os.makedirs(reconstruction_dir)

    reconstruction_meshes_dir = os.path.join(
        reconstruction_dir, ws.reconstruction_meshes_subdir
    )
    if not os.path.isdir(reconstruction_meshes_dir):
        os.makedirs(reconstruction_meshes_dir)

    reconstruction_codes_dir = os.path.join(
        reconstruction_dir, ws.reconstruction_codes_subdir
    )
    if not os.path.isdir(reconstruction_codes_dir):
        os.makedirs(reconstruction_codes_dir)

    optimize_and_reconstruct(
        args,
        npz_filenames,
        reconstruction_meshes_dir,
        decoder,
        latent_size,
        grid_size=args.grid_size,
    )
