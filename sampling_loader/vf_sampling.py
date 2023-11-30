#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

# Copyright 2023-present Edoardo Mello Rella, Ajad Chhatkuli, Ender Konukoglu & Luc Van Gool

import torch
import os
import logging
import json
import implicit_vf
import trimesh
import igl

import numpy as np

from torch.utils.data import Dataset
from sklearn.preprocessing import normalize

import implicit_vf.workspace as ws


def append_data_source_map(data_dir, name, source):
    data_source_map_filename = ws.get_data_source_map_filename(data_dir)

    print("data sources stored to " + data_source_map_filename)

    data_source_map = {}

    if os.path.isfile(data_source_map_filename):
        with open(data_source_map_filename, "r") as f:
            data_source_map = json.load(f)

    if name in data_source_map:
        print(data_source_map[name], os.path.abspath(source))
        if not data_source_map[name] == os.path.abspath(source):
            raise RuntimeError(
                "Cannot add data with the same name and a different source."
            )

    else:
        data_source_map[name] = os.path.abspath(source)

        with open(data_source_map_filename, "w") as f:
            json.dump(data_source_map, f, indent=2)


class VFSampling(Dataset):
    def __init__(
        self,
        args,
        subdir,
        extension,
        test_sampling=False,
        n_surface_points=250000,
        n_sphere_points=25000,
    ):
        self.test_sampling = test_sampling
        self.n_surface_points = n_surface_points
        self.n_sphere_points = n_sphere_points
        self.std_first_perturbation = 0.0707106781
        self.std_second_perturbation = 0.0223606798
        if self.test_sampling:
            self.n_surface_points = 125000
            self.std_first_perturbation = 0.223606798

        with open(args.split_filename, "r") as f:
            split = json.load(f)

        if args.source_name is None:
            args.source_name = os.path.basename(os.path.normpath(args.source_dir))

        dest_dir = os.path.join(args.data_dir, subdir, args.source_name)

        logging.info(
            "Preprocessing data from "
            + args.source_dir
            + " and placing the results in "
            + dest_dir
        )

        if not os.path.isdir(dest_dir):
            os.makedirs(dest_dir)

        append_data_source_map(args.data_dir, args.source_name, args.source_dir)

        class_directories = split[args.source_name]

        meshes_targets_and_specific_args = []

        for class_dir in class_directories:
            class_path = os.path.join(args.source_dir, class_dir)
            instance_dirs = class_directories[class_dir]

            logging.debug(
                "Processing "
                + str(len(instance_dirs))
                + " instances of class "
                + class_dir
            )

            target_dir = os.path.join(dest_dir, class_dir)

            if not os.path.isdir(target_dir):
                os.mkdir(target_dir)

            for instance_dir in instance_dirs:
                shape_dir = os.path.join(class_path, instance_dir)

                processed_filepath = os.path.join(target_dir, instance_dir + extension)
                if args.skip and os.path.isfile(processed_filepath):
                    logging.debug("skipping " + processed_filepath)
                    continue

                try:
                    mesh_filename = implicit_vf.data.find_mesh_in_directory(shape_dir)
                    print("found something at", mesh_filename)

                    meshes_targets_and_specific_args.append(
                        (
                            mesh_filename,
                            processed_filepath,
                        )
                    )

                except implicit_vf.data.NoMeshFileError:
                    logging.warning("No mesh found for instance " + instance_dir)
                except implicit_vf.data.MultipleMeshFileError:
                    logging.warning(
                        "Multiple meshes found for instance " + instance_dir
                    )

        self.mesh_targets = meshes_targets_and_specific_args

    def __len__(self):
        return len(self.mesh_targets)

    def __getitem__(self, index):
        (mesh_name, target_path) = self.mesh_targets[index]
        scene = trimesh.load(mesh_name)
        source_mesh = implicit_vf.utils.as_mesh(scene)

        source_mesh.vertices = self.center_and_normalize_mesh(
            mesh_vertices=source_mesh.vertices
        )
        point_samples = self.sample_points(source_mesh=source_mesh)
        vf_udf = self.extract_vf_udf(source_mesh=source_mesh, points=point_samples)

        points_with_vf = np.concatenate((point_samples, vf_udf), axis=1)

        self.group_points_by_vf_and_save(
            points_with_vf=points_with_vf, out_file=target_path
        )

        return mesh_name

    def center_and_normalize_mesh(self, mesh_vertices):
        buffer = 1.03

        v_min = np.min(mesh_vertices, axis=0)
        v_max = np.max(mesh_vertices, axis=0)
        v_mid = (v_min + v_max) / 2.0
        mesh_vertices -= v_mid

        max_distance = np.max(np.linalg.norm(mesh_vertices, axis=1))
        mesh_vertices /= max_distance * buffer

        return mesh_vertices

    def sample_points(self, source_mesh):
        # Uniformely sample unit sphere
        phi = np.random.uniform(0.0, 2.0 * np.pi, self.n_sphere_points)
        costheta = np.random.uniform(-1.0, 1.0, self.n_sphere_points)
        u = np.random.uniform(0.0, 1.0, self.n_sphere_points)
        theta = np.arccos(costheta)
        r = np.cbrt(u)
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        sphere_samples = np.stack((x, y, z), axis=1)

        # Sample from mesh surface and perturb around it
        surface_samples = trimesh.sample.sample_surface(
            source_mesh, self.n_surface_points
        )[0]
        first_perturbation = np.random.normal(
            loc=0.0, scale=self.std_first_perturbation, size=(self.n_surface_points, 3)
        )
        second_perturbation = np.random.normal(
            loc=0.0, scale=self.std_second_perturbation, size=(self.n_surface_points, 3)
        )
        perturbed_surface_samples = np.concatenate(
            (
                first_perturbation + surface_samples,
                second_perturbation + surface_samples,
            ),
            axis=0,
        )

        return np.concatenate((sphere_samples, perturbed_surface_samples), axis=0)

    def extract_vf_udf(self, source_mesh, points):
        surface_dist, _, closest_points = igl.signed_distance(
            points, source_mesh.vertices, source_mesh.faces
        )
        vf = normalize(closest_points - points, axis=1)
        udf = surface_dist

        return np.concatenate((vf, np.expand_dims(udf, axis=1)), axis=1)

    def group_points_by_vf_and_save(self, points_with_vf, out_file):
        first_set = points_with_vf[
            (points_with_vf[:, 3] >= 0)
            & (points_with_vf[:, 4] >= 0)
            & (points_with_vf[:, 5] >= 0)
        ]
        second_set = points_with_vf[
            (points_with_vf[:, 3] >= 0)
            & (points_with_vf[:, 4] >= 0)
            & (points_with_vf[:, 5] < 0)
        ]
        third_set = points_with_vf[
            (points_with_vf[:, 3] >= 0)
            & (points_with_vf[:, 4] < 0)
            & (points_with_vf[:, 5] >= 0)
        ]
        fourth_set = points_with_vf[
            (points_with_vf[:, 3] >= 0)
            & (points_with_vf[:, 4] < 0)
            & (points_with_vf[:, 5] < 0)
        ]
        fifth_set = points_with_vf[
            (points_with_vf[:, 3] < 0)
            & (points_with_vf[:, 4] >= 0)
            & (points_with_vf[:, 5] >= 0)
        ]
        sixth_set = points_with_vf[
            (points_with_vf[:, 3] < 0)
            & (points_with_vf[:, 4] >= 0)
            & (points_with_vf[:, 5] < 0)
        ]
        seventh_set = points_with_vf[
            (points_with_vf[:, 3] < 0)
            & (points_with_vf[:, 4] < 0)
            & (points_with_vf[:, 5] >= 0)
        ]
        eighth_set = points_with_vf[
            (points_with_vf[:, 3] < 0)
            & (points_with_vf[:, 4] < 0)
            & (points_with_vf[:, 5] < 0)
        ]
        np.savez(
            out_file,
            first=first_set,
            second=second_set,
            third=third_set,
            fourth=fourth_set,
            fifth=fifth_set,
            sixth=sixth_set,
            seventh=seventh_set,
            eighth=eighth_set,
        )

        return
