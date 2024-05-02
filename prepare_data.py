#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

# Copyright 2023-present Edoardo Mello Rella, Ajad Chhatkuli, Ender Konukoglu & Luc Van Gool

import argparse
import json
import logging
import os
import subprocess

import implicit_vf
import implicit_vf.workspace as ws

from sampling_loader.vf_sampling import VFSampling
from torch.utils.data import DataLoader


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Prepare dataset to use for training, inference, and evaluation from a set of meshes",
    )
    arg_parser.add_argument(
        "--data_dir",
        "-d",
        dest="data_dir",
        required=True,
        help="The directory which holds all preprocessed data.",
    )
    arg_parser.add_argument(
        "--source",
        "-s",
        dest="source_dir",
        required=True,
        help="The directory which holds the data to preprocess and append.",
    )
    arg_parser.add_argument(
        "--name",
        "-n",
        dest="source_name",
        default=None,
        help="The name to use for the data source. If unspecified, it defaults to the "
        + "directory name.",
    )
    arg_parser.add_argument(
        "--split",
        dest="split_filename",
        required=True,
        help="A split filename defining the shapes to be processed.",
    )
    arg_parser.add_argument(
        "--skip",
        dest="skip",
        default=False,
        action="store_true",
        help="If set, previously-processed shapes will be skipped",
    )
    arg_parser.add_argument(
        "--threads",
        dest="num_threads",
        default=8,
        help="The number of threads to use to process the data.",
    )
    arg_parser.add_argument(
        "--test",
        "-t",
        dest="test_sampling",
        default=False,
        action="store_true",
        help="If set, the script will produce VF samplies for testing."
        + "Otherwise for training",
    )

    implicit_vf.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    implicit_vf.configure_logging(args)

    subdir = ws.vf_samples_subdir
    extension = ".npz"
    executable = VFSampling(
        args=args,
        subdir=subdir,
        extension=extension,
        test_sampling=args.test_sampling,
    )

    sampling_loader = DataLoader(
        executable,
        batch_size=1,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
        num_workers=8,
    )

    for el, file_name in enumerate(sampling_loader):
        print(
            "Processed file {} at position {} of {}".format(
                file_name, el, len(executable)
            )
        )
