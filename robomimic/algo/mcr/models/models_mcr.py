# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
from numpy.core.numeric import full
import torch
import torch.nn as nn
from torch.nn.modules.activation import Sigmoid
from torch.nn.modules.linear import Identity
import torchvision
from torchvision import transforms
from robomimic.algo.mcr import utils
from pathlib import Path
from torchvision.utils import save_image
import torchvision.transforms as T

epsilon = 1e-8


def do_nothing(x):
    return x


class MCR(nn.Module):
    def __init__(
        self,
        device,
        lr,
        hidden_dim,
        size=34,
        tcnweight=0.0,
        l2dist=True,
        bs=16,
        align_state_weight=0.0,
        state_list=None,
        state_window=1,
        use_action=False,
        bc_weight=0.0,
    ):
        super().__init__()

        self.device = device
        self.use_tb = False
        self.tcnweight = tcnweight  ## Weight on TCN loss (states closer in same clip closer in embedding)
        self.l2dist = l2dist  ## Use -l2 or cosine sim
        self.size = size  ## Size ResNet or ViT
        self.num_negatives = 3

        self.align_state_weight = align_state_weight  ## weight on dynamics alignment
        self.state_list = state_list
        self.state_window = state_window
        self.use_action = use_action
        self.bc_weight = bc_weight  ## weight on actor prediction loss

        ## Distances and Metrics
        self.cs = torch.nn.CosineSimilarity(1)
        self.bce = nn.BCELoss(reduce=False)
        self.sigm = Sigmoid()
        self.bc_loss = nn.MSELoss()

        params = []
        ######################################################################## Sub Modules
        ## Visual Encoder
        if size == 18:
            self.outdim = 512
            self.convnet = torchvision.models.resnet18(pretrained=False)
        elif size == 34:
            self.outdim = 512
            self.convnet = torchvision.models.resnet34(pretrained=False)
        elif size == 50:
            self.outdim = 2048
            self.convnet = torchvision.models.resnet50(pretrained=False)
        elif size == 0:
            from transformers import AutoConfig, AutoModel

            self.outdim = 768
            self.convnet = AutoModel.from_config(
                config=AutoConfig.from_pretrained("google/vit-base-patch32-224-in21k")
            ).to(
                self.device
            )  # guangqi: if cannot visit hf, this may help: https://github.com/LetheSec/HuggingFace-Download-Accelerator

        if self.size == 0:
            self.normlayer = transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
            )
        else:
            self.normlayer = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        self.convnet.fc = Identity()
        self.convnet.train()
        params += list(self.convnet.parameters())

        ## State projector
        if self.align_state_weight > 0.0:
            state_input_dim = 14 * self.state_window

            if self.use_action:
                state_input_dim += 7 * (self.state_window - 1)

            self.state_encoder = nn.Sequential(
                nn.Linear(state_input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.outdim),
            ).to(self.device)
            params += list(self.state_encoder.parameters())

        ## Actor loss
        if self.bc_weight > 0.0:
            feature_dim = 50
            bc_hidden_dim = 512
            action_dim = 7

            self.bc_trunk = nn.Sequential(
                nn.Linear(self.outdim, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.Tanh(),
            ).to(self.device)

            self.bc_policy = nn.Sequential(
                nn.Linear(feature_dim, bc_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(bc_hidden_dim, bc_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(bc_hidden_dim, action_dim),
            ).to(self.device)
            params += list(self.bc_trunk.parameters())
            params += list(self.bc_policy.parameters())

        ## Optimizer
        self.encoder_opt = torch.optim.Adam(params, lr=lr)

    ## Forward Call (im --> representation)
    def forward(self, obs, num_ims=1, obs_shape=[3, 224, 224]):
        if obs_shape != [3, 224, 224]:
            preprocess = nn.Sequential(
                transforms.Resize(256),
                transforms.CenterCrop(224),
                self.normlayer,
            )
        else:
            preprocess = nn.Sequential(
                self.normlayer,
            )

        ## Input here must be [0, 255], this is consistent with R3M
        obs = obs.float() / 255.0
        obs_p = preprocess(obs)
        h = self.convnet(obs_p)
        return h

    def sim(self, tensor1, tensor2):
        if self.l2dist:
            d = -torch.linalg.norm(tensor1 - tensor2, dim=-1)
        else:
            d = self.cs(tensor1, tensor2)
        return d
