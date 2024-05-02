#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

# Copyright 2023-present Edoardo Mello Rella, Ajad Chhatkuli, Ender Konukoglu & Luc Van Gool

import json
import argparse
import os
import random


shape_to_class = {
    "lamps": "03636649",
}


def main():
    arg_parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Create split file from raw data",
    )
    arg_parser.add_argument(
        "--data_dir",
        "-d",
        dest="data_dir",
        required=True,
        help="The directory which holds the raw data",
    )
    arg_parser.add_argument(
        "--class",
        "-c",
        dest="shape_class",
        required=True,
        help="The name of the class for which to create the split",
    )
    args = arg_parser.parse_args()

    shape_name = args.shape_class
    shape_code = shape_to_class[shape_name]

    train_data = {}
    train_data["ShapeNetV2"] = {}

    test_data = {}
    test_data["ShapeNetV2"] = {}

    class_dir = os.path.join(args.data_dir, shape_code)

    data_list = []
    for dirpath, dirnames, filenames in os.walk(class_dir):
        if "model_normalized.obj" in filenames:
            data_list.append(str(dirpath.split("/")[-2]))
    random.shuffle(data_list)
    num_shapes = len(data_list)
    num_train_shapes = int(num_shapes * 0.8)
    print(
        "From {} files, use {} shapes for training and {} for evaliation".format(
            num_shapes, num_train_shapes, num_shapes - num_train_shapes
        )
    )
    train_shapes = data_list[:num_train_shapes]
    test_shapes = data_list[num_train_shapes:]

    train_data["ShapeNetV2"][shape_code] = train_shapes
    test_data["ShapeNetV2"][shape_code] = test_shapes

    train_split_name = "sv2_{}_train.json".format(shape_name)
    train_target = os.path.join(args.data_dir, train_split_name)
    test_split_name = "sv2_{}_test.json".format(shape_name)
    test_target = os.path.join(args.data_dir, test_split_name)

    with open(train_target, "w") as fp:
        json.dump(train_data, fp)
    with open(test_target, "w") as fp:
        json.dump(test_data, fp)

    return


if __name__ == "__main__":
    main()
