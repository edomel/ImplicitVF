#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

# Copyright 2023-present Edoardo Mello Rella, Ajad Chhatkuli, Ender Konukoglu & Luc Van Gool


import os
import logging
import torch
import time

import implicit_vf
import implicit_vf.workspace as ws


def reconstruct(
    decoder,
    num_iterations,
    latent_size,
    test_vf,
    stat,
    clamp_dist,
    num_samples=800,
    lr=5e-4,
    l2reg=False,
):
    def adjust_learning_rate(
        initial_lr, optimizer, num_iterations, decreased_by, adjust_lr_every
    ):
        lr = initial_lr * ((1 / decreased_by) ** (num_iterations // adjust_lr_every))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    decreased_by = 10
    adjust_lr_every = int(num_iterations / 2)

    if type(stat) == type(0.1):
        latent = torch.ones(1, latent_size).normal_(mean=0, std=stat).cuda()
    else:
        latent = torch.normal(stat[0].detach(), stat[1].detach()).cuda()

    latent.requires_grad = True
    optimizer = torch.optim.Adam([latent], lr=lr)
    decoder.eval()

    loss_num = 0
    loss_l2 = torch.nn.MSELoss()

    for e in range(num_iterations):
        vf_data = implicit_vf.data.unpack_open_vf_samples_from_ram(
            test_vf,
            num_samples,
        ).cuda()

        xyz = vf_data[:, :3]
        xyz.requires_grad = True
        vf_gt = vf_data[:, 3:]

        adjust_learning_rate(lr, optimizer, e, decreased_by, adjust_lr_every)

        optimizer.zero_grad()
        latent_inputs = latent.expand(num_samples, -1)
        inputs = torch.cat([latent_inputs, xyz], 1).cuda()

        pred = decoder(inputs)

        loss = loss_l2(pred, vf_gt)

        if l2reg:
            loss += 1e-4 * torch.mean(latent.pow(2))

        loss.backward()
        optimizer.step()

        if e % 50 == 0:
            logging.debug(loss.cpu().data.numpy())
            logging.debug(e)
            logging.debug(latent.norm())
        loss_num = loss.cpu().data.numpy()

    return loss_num, latent


def optimize_and_reconstruct(
    args,
    npz_filenames,
    reconstruction_meshes_dir,
    decoder,
    latent_size,
    grid_size=256,
):
    err_sum = 0.0

    for ii, npz in enumerate(npz_filenames):

        if "npz" not in npz:
            continue

        full_filename = os.path.join(args.data_source, ws.vf_samples_subdir, npz)

        logging.debug("loading {}".format(npz))

        data_vf = implicit_vf.data.read_open_vf_samples_into_ram(full_filename)

        mesh_filename = os.path.join(reconstruction_meshes_dir, npz[:-4])

        logging.info("reconstructing {}".format(npz))

        for set in range(8):
            data_vf[set] = data_vf[set][torch.randperm(data_vf[set].shape[0])][
                : min(data_vf[set].shape[0], 100)
            ]

        start = time.time()
        err, latent = reconstruct(
            decoder,
            800,
            latent_size,
            data_vf,
            0.01,
            0.1,
            num_samples=800,
            lr=5e-3,
            l2reg=True,
        )
        logging.debug("reconstruct time: {}".format(time.time() - start))
        err_sum += err
        logging.debug("current_error avg: {}".format((err_sum / (ii + 1))))
        logging.debug(ii)

        logging.debug("latent: {}".format(latent.detach().cpu().numpy()))

        decoder.eval()

        if not os.path.exists(os.path.dirname(mesh_filename)):
            os.makedirs(os.path.dirname(mesh_filename))

        start = time.time()
        with torch.no_grad():
            implicit_vf.mesh.create_mesh_vf(
                decoder,
                latent,
                mesh_filename,
                N=grid_size,
                max_batch=int(2**18),
            )

        logging.debug("total time: {}".format(time.time() - start))

    return
