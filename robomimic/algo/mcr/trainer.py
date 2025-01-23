# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import hydra
import numpy as np
from numpy.core.numeric import full
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from pathlib import Path
from torchvision.utils import save_image
import time
import copy
import torchvision.transforms as T

epsilon = 1e-8


def do_nothing(x):
    return x


class Trainer:
    def __init__(self, eval_freq):
        self.eval_freq = eval_freq

    def update(self, model, batch, step, eval=False):
        t0 = time.time()
        metrics = dict()
        if eval:
            model.eval()
            if model.module.align_state_weight > 0:
                model.module.state_encoder.eval()
        else:
            model.train()
            if model.module.align_state_weight > 0:
                model.module.state_encoder.train()

        t1 = time.time()
        ## Batch
        b_im, b_lang, b_state, b_fullstate, b_actions = batch
        t2 = time.time()

        ## Encode Start and End Frames
        bs = b_im.shape[0]
        img_stack_len = b_im.shape[1]
        b_im_r = b_im.reshape(
            bs * img_stack_len, 3, 224, 224
        )  # (bs*5, 3, 224, 224) or 180 320
        alles = model(b_im_r)  # [0, 255] tensor input
        alle = alles.reshape(bs, img_stack_len, -1)
        e0 = alle[:, 0]
        eg = alle[:, 1]
        es0 = alle[:, 2]
        es1 = alle[:, 3]
        es2 = alle[:, 4]

        full_loss = 0

        t3 = time.time()
        ## Within Video TCN Loss
        if model.module.tcnweight > 0:
            ## Number of negative video examples to use
            num_neg_v = model.module.num_negatives

            ## Computing distance from t0-t2, t1-t2, t1-t0
            sim_0_2 = model.module.sim(es2, es0)
            sim_1_2 = model.module.sim(es2, es1)
            sim_0_1 = model.module.sim(es1, es0)

            ## For the specified number of negatives from other videos
            ## Add it as a negative
            neg2 = []
            neg0 = []
            for _ in range(num_neg_v):
                es0_shuf = es0[
                    torch.randperm(es0.size()[0])
                ]  # shuffle, bs, 1, embedding_shape
                es2_shuf = es2[torch.randperm(es2.size()[0])]
                neg0.append(model.module.sim(es0, es0_shuf))
                neg2.append(model.module.sim(es2, es2_shuf))
            neg0 = torch.stack(neg0, -1)
            neg2 = torch.stack(neg2, -1)

            ## TCN Loss
            smoothloss1 = -torch.log(
                epsilon
                + (
                    torch.exp(sim_1_2)
                    / (
                        epsilon
                        + torch.exp(sim_0_2)
                        + torch.exp(sim_1_2)
                        + torch.exp(neg2).sum(-1)
                    )
                )
            )
            smoothloss2 = -torch.log(
                epsilon
                + (
                    torch.exp(sim_0_1)
                    / (
                        epsilon
                        + torch.exp(sim_0_1)
                        + torch.exp(sim_0_2)
                        + torch.exp(neg0).sum(-1)
                    )
                )
            )
            smoothloss = ((smoothloss1 + smoothloss2) / 2.0).mean()
            a_state = (
                (1.0 * (sim_0_2 < sim_1_2)) * (1.0 * (sim_0_1 > sim_0_2))
            ).mean()  # % of correctly aligned sample pairs
            metrics["tcnloss"] = smoothloss.item()
            metrics["aligned"] = a_state.item()
            full_loss += model.module.tcnweight * smoothloss

        ## bc loss
        if model.module.bc_weight > 0.0:
            b_actions = b_actions.reshape(bs * img_stack_len, 7)
            pred_action = model.module.bc_policy(model.module.bc_trunk(alles))
            bc_loss = model.module.bc_loss(pred_action, b_actions.detach())
            metrics["bc_loss"] = bc_loss.item()
            full_loss += model.module.bc_weight * bc_loss

        ## state align loss
        if model.module.align_state_weight > 0:
            s0_proj = model.module.state_encoder(b_fullstate["s0"].to(es0.device))
            s2_proj = model.module.state_encoder(b_fullstate["s2"].to(es0.device))
            assert (
                s0_proj.shape == es0.shape
            ), f"img embedding and statewind embedding shape mismatch, {es0.shape} and {s0_proj.shape}"
            sim_0_0s = model.module.sim(es0, s0_proj)
            sim_2_2s = model.module.sim(es2, s2_proj)
            sim_0_2s = model.module.sim(es0, s2_proj)
            sim_2_0s = model.module.sim(es2, s0_proj)

            s0loss = -torch.log(
                epsilon
                + (
                    torch.exp(sim_0_0s)
                    / (epsilon + torch.exp(sim_0_0s) + torch.exp(sim_0_2s))
                )
            )
            s2loss = -torch.log(
                epsilon
                + (
                    torch.exp(sim_2_2s)
                    / (epsilon + torch.exp(sim_2_2s) + torch.exp(sim_2_0s))
                )
            )
            state_align_loss = ((s0loss + s2loss) / 2.0).mean()
            a_img_state = (
                (1.0 * (sim_0_2s < sim_0_0s)) * (1.0 * (sim_2_0s < sim_2_2s))
            ).mean()
            metrics["state_align_loss"] = state_align_loss.item()
            metrics["stateimg_aligned"] = a_img_state.item()
            full_loss += model.module.align_state_weight * state_align_loss

        metrics["full_loss"] = full_loss.item()

        t4 = time.time()
        if not eval:
            model.module.encoder_opt.zero_grad()
            full_loss.backward()
            model.module.encoder_opt.step()

        t5 = time.time()
        st = f"Load time {t1-t0}, Batch time {t2-t1}, MCR time {t4-t3}, Backprop time {t5-t4}"
        return metrics, st
