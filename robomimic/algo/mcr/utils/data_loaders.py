# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings

import torchvision

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "egl"

from pathlib import Path

import hydra
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import IterableDataset
import pandas as pd
import json
import time
import pickle
from torchvision.utils import save_image
import json
import random


def get_ind(vid, index, ds):
    if ds == "ego4d":
        return torchvision.io.read_image(f"{vid}/{index:06}.jpg")
    elif ds == "droid":
        return torchvision.io.read_image(
            f"{vid}/{index}.png", mode=torchvision.io.image.ImageReadMode.RGB
        )
    else:
        raise NameError("Invalid Dataset")


## Data Loader for Ego4D
class MCRBuffer(IterableDataset):
    def __init__(
        self, ego4dpath, num_workers, source1, source2, alpha, datasources, doaug="none"
    ):
        self._num_workers = max(1, num_workers)
        self.alpha = alpha
        self.curr_same = 0
        self.data_sources = datasources
        self.doaug = doaug

        # Augmentations
        if doaug in ["rc", "rctraj"]:
            self.aug = torch.nn.Sequential(
                transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            )
        else:
            self.aug = lambda a: a

        # Load Data
        if "ego4d" in self.data_sources:
            print("Ego4D")
            self.manifest = pd.read_csv(f"{ego4dpath}manifest.csv")
            print(self.manifest)
            self.ego4dlen = len(self.manifest)
        else:
            raise NameError("Invalid Dataset")

    def _sample(self):
        t0 = time.time()
        ds = random.choice(self.data_sources)

        vidid = np.random.randint(0, self.ego4dlen)
        m = self.manifest.iloc[vidid]
        vidlen = m["len"]
        txt = m["txt"]
        label = txt[2:]  ## Cuts of the "C " part of the text
        vid = m["path"]

        start_ind = np.random.randint(1, 2 + int(self.alpha * vidlen))
        end_ind = np.random.randint(int((1 - self.alpha) * vidlen) - 1, vidlen)
        s1_ind = np.random.randint(2, vidlen)
        s0_ind = np.random.randint(1, s1_ind)
        s2_ind = np.random.randint(s1_ind, vidlen + 1)  # start, s0, s1, s2, end

        if self.doaug == "rctraj":
            ### Encode each image in the video at once the same way
            im0 = get_ind(vid, start_ind, ds)
            img = get_ind(vid, end_ind, ds)
            imts0 = get_ind(vid, s0_ind, ds)
            imts1 = get_ind(vid, s1_ind, ds)
            imts2 = get_ind(vid, s2_ind, ds)
            allims = torch.stack([im0, img, imts0, imts1, imts2], 0)
            allims_aug = self.aug(allims / 255.0) * 255.0

            im0 = allims_aug[0]
            img = allims_aug[1]
            imts0 = allims_aug[2]
            imts1 = allims_aug[3]
            imts2 = allims_aug[4]
        else:
            ### Encode each image individually
            im0 = self.aug(get_ind(vid, start_ind, ds) / 255.0) * 255.0
            img = self.aug(get_ind(vid, end_ind, ds) / 255.0) * 255.0
            imts0 = self.aug(get_ind(vid, s0_ind, ds) / 255.0) * 255.0
            imts1 = self.aug(get_ind(vid, s1_ind, ds) / 255.0) * 255.0
            imts2 = self.aug(get_ind(vid, s2_ind, ds) / 255.0) * 255.0

        im = torch.stack([im0, img, imts0, imts1, imts2])
        return (im, label)

    def __iter__(self):
        while True:
            yield self._sample()


## Data Loader for Droid
class MCRBufferDroid(IterableDataset):
    def __init__(
        self,
        droidpath,
        num_workers,
        source1,
        source2,
        alpha,
        datasources,
        doaug="none",
        state_list_used=None,
        state_window=1,
        use_action=False,
        view_keys_used=None,
    ):
        self._num_workers = max(1, num_workers)
        self.alpha = alpha
        self.curr_same = 0
        self.data_sources = datasources
        self.doaug = doaug
        self.dataset_path = droidpath
        self.state_keys = ["cartesian_position", "gripper_position", "joint_position"]
        self.lang_keys = [
            "language_instruction",
            "language_instruction_2",
            "language_instruction_3",
        ]
        self.view_keys = view_keys_used  # ['exterior_image_1_left', 'exterior_image_2_left', 'wrist_image_left']
        self.state_list_used = state_list_used
        self.state_window = state_window
        self.use_action = use_action

        # Augmentations
        if doaug in ["rc", "rctraj"]:
            self.aug = torch.nn.Sequential(
                transforms.RandomResizedCrop(
                    224, scale=(0.5, 1.0)
                ),  # first crop, then resize
            )
        elif doaug in ["rctraj_eval"]:
            self.aug = torch.nn.Sequential(
                transforms.Resize(256),
                transforms.CenterCrop(224),
            )
        else:
            self.aug = lambda a: a

        # Load Data
        if "droid" in self.data_sources:
            print("Droid")
            self.loaded_dataset = os.listdir(droidpath)
            print(self.loaded_dataset[:5])
            self.datasetlen = len(self.loaded_dataset)
        else:
            raise NameError("Invalid Dataset")

    def _sample(self):
        t0 = time.time()
        ds = random.choice(self.data_sources)

        vidid = np.random.randint(0, self.datasetlen)
        traj_path = self.loaded_dataset[vidid]  # 2023-02-28_Tue_Feb_28_20:31:42_2023
        vidlen = min(
            len(
                os.listdir(
                    os.path.join(self.dataset_path, traj_path, "exterior_image_1_left")
                )
            ),
            len(
                os.listdir(
                    os.path.join(self.dataset_path, traj_path, "exterior_image_2_left")
                )
            ),
        )
        txt_path = random.choice(self.lang_keys)
        with open(
            os.path.join(self.dataset_path, traj_path, txt_path, "0.txt"), "r"
        ) as file:
            label = file.read()
        vid_path = random.choice(self.view_keys)  # time contrastive within same view
        vid = os.path.join(self.dataset_path, traj_path, vid_path)  # video path
        otherdata_path = os.path.join(self.dataset_path, traj_path, "other_data.pkl")

        start_ind = np.random.randint(1, 2 + int(self.alpha * vidlen))  # [low, high)
        end_ind = np.random.randint(int((1 - self.alpha) * vidlen) - 1, vidlen)
        s1_ind = np.random.randint(2, vidlen)
        s0_ind = np.random.randint(1, s1_ind)
        s2_ind = np.random.randint(s1_ind, vidlen)  # start, s0, s1, s2, end

        # for state encode
        with open(otherdata_path, "rb") as f:
            loaded_data = pickle.load(f)
        state_array, full_state_dict = np.empty(0), {
            "s0": np.empty(0),
            "s2": np.empty(0),
        }
        for key in self.state_list_used:
            state_array = np.concatenate((state_array, loaded_data[key][s0_ind]))

        s0wind_start = max(1, s0_ind - self.state_window // 2)
        s2wind_start = max(1, s2_ind - self.state_window // 2)
        for i in range(self.state_window):
            for key in self.state_keys:
                full_state_dict["s0"] = np.concatenate(
                    (
                        full_state_dict["s0"],
                        loaded_data[key][min(s0wind_start + i, vidlen - 1)],
                    )
                )
                full_state_dict["s2"] = np.concatenate(
                    (
                        full_state_dict["s2"],
                        loaded_data[key][min(s2wind_start + i, vidlen - 1)],
                    )
                )
            if self.use_action and i != self.state_window - 1:
                full_state_dict["s0"] = np.concatenate(
                    (
                        full_state_dict["s0"],
                        loaded_data["action"][min(s0wind_start + i, vidlen - 1)],
                    )
                )
                full_state_dict["s2"] = np.concatenate(
                    (
                        full_state_dict["s2"],
                        loaded_data["action"][min(s2wind_start + i, vidlen - 1)],
                    )
                )

        full_state_dict["s0"] = torch.tensor(full_state_dict["s0"]).float()
        full_state_dict["s2"] = torch.tensor(full_state_dict["s2"]).float()

        # for bc, sample action
        actions = torch.tensor(
            np.stack(
                [
                    loaded_data["action"][start_ind],
                    loaded_data["action"][end_ind],
                    loaded_data["action"][s0_ind],
                    loaded_data["action"][s1_ind],
                    loaded_data["action"][s2_ind],
                ]
            )
        ).float()
        # actions = torch.tensor(loaded_data['action'][s0_ind]).float()

        if self.doaug == ["rctraj", "rctraj_eval"]:
            ### Encode each image in the video at once the same way
            im0 = get_ind(vid, start_ind, ds)
            img = get_ind(vid, end_ind, ds)
            imts0 = get_ind(vid, s0_ind, ds)
            imts1 = get_ind(vid, s1_ind, ds)
            imts2 = get_ind(vid, s2_ind, ds)
            allims = torch.stack([im0, img, imts0, imts1, imts2], 0)
            allims_aug = self.aug(allims / 255.0) * 255.0

            im0 = allims_aug[0]
            img = allims_aug[1]
            imts0 = allims_aug[2]
            imts1 = allims_aug[3]
            imts2 = allims_aug[4]
        else:
            ### Encode each image individually
            im0 = self.aug(get_ind(vid, start_ind, ds) / 255.0) * 255.0
            img = self.aug(get_ind(vid, end_ind, ds) / 255.0) * 255.0
            imts0 = self.aug(get_ind(vid, s0_ind, ds) / 255.0) * 255.0
            imts1 = self.aug(get_ind(vid, s1_ind, ds) / 255.0) * 255.0
            imts2 = self.aug(get_ind(vid, s2_ind, ds) / 255.0) * 255.0

        im = torch.stack([im0, img, imts0, imts1, imts2])
        return (im, label, torch.tensor(state_array).float(), full_state_dict, actions)

    def __iter__(self):
        while True:
            yield self._sample()
