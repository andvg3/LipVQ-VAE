# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import omegaconf
import hydra
import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image

from robomimic.algo.mcr import load_mcr

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

mcr = load_mcr(ckpt_path="expdata/robocasa/mcr/mcr_resnet50.pth")  # mcr is a resnet
mcr.to(device)

## DEFINE PREPROCESSING
transforms = T.Compose(
    [T.Resize(256), T.CenterCrop(224), T.ToTensor()]
)  # ToTensor() divides by 255

## ENCODE IMAGE
image = np.random.randint(0, 255, (500, 500, 3))
preprocessed_image = transforms(Image.fromarray(image.astype(np.uint8))).reshape(
    -1, 3, 224, 224
)
preprocessed_image.to(device)
with torch.no_grad():
    embedding = mcr(preprocessed_image.cuda())
import time

time.sleep(10)
print(embedding.shape)  # [1, 2048]
