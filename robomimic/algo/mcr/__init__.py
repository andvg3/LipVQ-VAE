# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
from robomimic.algo.mcr.models.models_mcr import MCR

VALID_ARGS = [
    "_target_",
    "device",
    "lr",
    "hidden_dim",
    "size",
    "l2weight",
    "l1weight",
    "langweight",
    "tcnweight",
    "l2dist",
    "bs",
]
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def load_mcr(ckpt_path="./ckpts/mcr_resnet50.pth"):
    """
    Load our pre-trained ResNet50 evaluated in our paper.
    Return:
        A torchvision.models.resnet.ResNet in eval mode, whose fc layer is substituted with Identity().
    """
    from torchvision.models import resnet50
    import torch.nn as nn

    model = resnet50(weights=None)
    model.fc = nn.Identity()
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))

    return model.eval().to(device)
