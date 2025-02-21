"""
Contains torch Modules that help deal with inputs consisting of multiple
modalities. This is extremely common when networks must deal with one or 
more observation dictionaries, where each input dictionary can have
observation keys of a certain modality and shape.

As an example, an observation could consist of a flat "robot0_eef_pos" observation key,
and a 3-channel RGB "agentview_image" observation key.
"""
import sys
import numpy as np
import textwrap
from copy import deepcopy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import torchvision.transforms as T

# from mamba_ssm import Mamba
from transformers import AutoProcessor
import clip

from robomimic.utils.python_utils import extract_class_init_kwargs_from_dict
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.models.base_nets import (
    Module,
    Sequential,
    MLP,
    RNN_Base,
    ResNet18Conv,
    SpatialSoftmax,
    FeatureAggregator,
)
from robomimic.models.obs_core import (
    VisualCore,
    Randomizer,
    VisualCoreLanguageConditioned,
)

from robomimic.models.vq_vae.backbone import VQVAE
from robomimic.models.vq_vae.backbone_lfqvae import LFQVAE
from robomimic.models.vq_vae.backbone_lfqvae_v2 import LLFQVAE
from robomimic.models.vq_vae.backbone_lfqvae_v3 import LFQVAE_V3
from robomimic.models.vq_vae.backbone_lfqvae_v4 import LLFQVAE_V3
from robomimic.models.vq_vae.backbone_lfqvae_lipschitz import LFQVAE
from robomimic.models.bin_action.backbone import AdaptiveBinActionEmbedding
from robomimic.models.transformers import PositionalEncoding, GPT_Backbone
from robomimic.macros import LANG_EMB_KEY


def obs_encoder_factory(
    obs_shapes,
    feature_activation=nn.ReLU,
    encoder_kwargs=None,
):
    """
    Utility function to create an @ObservationEncoder from kwargs specified in config.

    Args:
        obs_shapes (OrderedDict): a dictionary that maps observation key to
            expected shapes for observations.

        feature_activation: non-linearity to apply after each obs net - defaults to ReLU. Pass
            None to apply no activation.

        encoder_kwargs (dict or None): If None, results in default encoder_kwargs being applied. Otherwise, should be
            nested dictionary containing relevant per-modality information for encoder networks.
            Should be of form:

            obs_modality1: dict
                feature_dimension: int
                core_class: str
                core_kwargs: dict
                    ...
                    ...
                obs_randomizer_class: str
                obs_randomizer_kwargs: dict
                    ...
                    ...
            obs_modality2: dict
                ...
    """
    enc = ObservationEncoder(feature_activation=feature_activation)
    for k, obs_shape in obs_shapes.items():
        obs_modality = ObsUtils.OBS_KEYS_TO_MODALITIES[k]
        enc_kwargs = (
            deepcopy(ObsUtils.DEFAULT_ENCODER_KWARGS[obs_modality])
            if encoder_kwargs is None
            else deepcopy(encoder_kwargs[obs_modality])
        )

        # Sanity check for kwargs in case they don't exist / are None
        if enc_kwargs.get("core_kwargs", None) is None:
            enc_kwargs["core_kwargs"] = {}
        # Add in input shape info
        enc_kwargs["core_kwargs"]["input_shape"] = obs_shape
        # If group class is specified, then make sure corresponding kwargs only contain relevant kwargs
        if enc_kwargs["core_class"] is not None:
            enc_kwargs["core_kwargs"] = extract_class_init_kwargs_from_dict(
                cls=ObsUtils.OBS_ENCODER_CORES[enc_kwargs["core_class"]],
                dic=enc_kwargs["core_kwargs"],
                copy=False,
            )

        # Add in input shape info
        randomizers = []
        obs_randomizer_class_list = enc_kwargs["obs_randomizer_class"]
        obs_randomizer_kwargs_list = enc_kwargs["obs_randomizer_kwargs"]

        if not isinstance(obs_randomizer_class_list, list):
            obs_randomizer_class_list = [obs_randomizer_class_list]

        if not isinstance(obs_randomizer_kwargs_list, list):
            obs_randomizer_kwargs_list = [obs_randomizer_kwargs_list]

        for rand_class, rand_kwargs in zip(
            obs_randomizer_class_list, obs_randomizer_kwargs_list
        ):
            rand = None
            if rand_class is not None:
                rand_kwargs["input_shape"] = obs_shape
                rand_kwargs = extract_class_init_kwargs_from_dict(
                    cls=ObsUtils.OBS_RANDOMIZERS[rand_class],
                    dic=rand_kwargs,
                    copy=False,
                )
                rand = ObsUtils.OBS_RANDOMIZERS[rand_class](**rand_kwargs)
            randomizers.append(rand)

        enc.register_obs_key(
            name=k,
            shape=obs_shape,
            net_class=enc_kwargs["core_class"],
            net_kwargs=enc_kwargs["core_kwargs"],
            randomizers=randomizers,
        )

    enc.make()
    return enc


def mcr_obs_encoder_factory(
    obs_shapes,
    feature_activation=nn.ReLU,
    encoder_kwargs=None,
    mcr_model=None,
):
    """
    Utility function to create an @ObservationEncoder from kwargs specified in config.

    Args:
        obs_shapes (OrderedDict): a dictionary that maps observation key to
            expected shapes for observations.

        feature_activation: non-linearity to apply after each obs net - defaults to ReLU. Pass
            None to apply no activation.

        encoder_kwargs (dict or None): If None, results in default encoder_kwargs being applied. Otherwise, should be
            nested dictionary containing relevant per-modality information for encoder networks.
            Should be of form:

            obs_modality1: dict
                feature_dimension: int
                core_class: str
                core_kwargs: dict
                    ...
                    ...
                obs_randomizer_class: str
                obs_randomizer_kwargs: dict
                    ...
                    ...
            obs_modality2: dict
                ...
    """
    enc = MCRObservationEncoder(
        feature_activation=feature_activation, mcr_model=mcr_model
    )
    for k, obs_shape in obs_shapes.items():
        obs_modality = ObsUtils.OBS_KEYS_TO_MODALITIES[k]

        enc_kwargs = (
            deepcopy(ObsUtils.DEFAULT_ENCODER_KWARGS[obs_modality])
            if encoder_kwargs is None
            else deepcopy(encoder_kwargs[obs_modality])
        )

        # Sanity check for kwargs in case they don't exist / are None
        if enc_kwargs.get("core_kwargs", None) is None:
            enc_kwargs["core_kwargs"] = {}
        # Add in input shape info
        enc_kwargs["core_kwargs"]["input_shape"] = obs_shape
        # If group class is specified, then make sure corresponding kwargs only contain relevant kwargs
        if enc_kwargs["core_class"] is not None:
            enc_kwargs["core_kwargs"] = extract_class_init_kwargs_from_dict(
                cls=ObsUtils.OBS_ENCODER_CORES[enc_kwargs["core_class"]],
                dic=enc_kwargs["core_kwargs"],
                copy=False,
            )

        # Add in input shape info
        randomizers = []
        obs_randomizer_class_list = enc_kwargs["obs_randomizer_class"]
        obs_randomizer_kwargs_list = enc_kwargs["obs_randomizer_kwargs"]

        if not isinstance(obs_randomizer_class_list, list):
            obs_randomizer_class_list = [obs_randomizer_class_list]

        if not isinstance(obs_randomizer_kwargs_list, list):
            obs_randomizer_kwargs_list = [obs_randomizer_kwargs_list]

        for rand_class, rand_kwargs in zip(
            obs_randomizer_class_list, obs_randomizer_kwargs_list
        ):
            rand = None
            if rand_class is not None:
                rand_kwargs["input_shape"] = obs_shape
                rand_kwargs = extract_class_init_kwargs_from_dict(
                    cls=ObsUtils.OBS_RANDOMIZERS[rand_class],
                    dic=rand_kwargs,
                    copy=False,
                )
                rand = ObsUtils.OBS_RANDOMIZERS[rand_class](**rand_kwargs)
            randomizers.append(rand)

        enc.register_obs_key(
            name=k,
            shape=obs_shape,
            net_class=enc_kwargs["core_class"],
            net_kwargs=enc_kwargs["core_kwargs"],
            randomizers=randomizers,
        )

    enc.make()
    return enc


class ObservationEncoder(Module):
    """
    Module that processes inputs by observation key and then concatenates the processed
    observation keys together. Each key is processed with an encoder head network.
    Call @register_obs_key to register observation keys with the encoder and then
    finally call @make to create the encoder networks.
    """

    def __init__(self, feature_activation=nn.ReLU):
        """
        Args:
            feature_activation: non-linearity to apply after each obs net - defaults to ReLU. Pass
                None to apply no activation.
        """
        super(ObservationEncoder, self).__init__()
        self.obs_shapes = OrderedDict()
        self.obs_nets_classes = OrderedDict()
        self.obs_nets_kwargs = OrderedDict()
        self.obs_share_mods = OrderedDict()
        self.obs_nets = nn.ModuleDict()
        self.obs_randomizers = nn.ModuleDict()
        self.feature_activation = feature_activation
        self._locked = False

    def register_obs_key(
        self,
        name,
        shape,
        net_class=None,
        net_kwargs=None,
        net=None,
        randomizers=None,
        share_net_from=None,
    ):
        """
        Register an observation key that this encoder should be responsible for.

        Args:
            name (str): modality name
            shape (int tuple): shape of modality
            net_class (str): name of class in base_nets.py that should be used
                to process this observation key before concatenation. Pass None to flatten
                and concatenate the observation key directly.
            net_kwargs (dict): arguments to pass to @net_class
            net (Module instance): if provided, use this Module to process the observation key
                instead of creating a different net
            randomizer (Randomizer instance): if provided, use this Module to augment observation keys
                coming in to the encoder, and possibly augment the processed output as well
            share_net_from (str): if provided, use the same instance of @net_class
                as another observation key. This observation key must already exist in this encoder.
                Warning: Note that this does not share the observation key randomizer
        """
        assert (
            not self._locked
        ), "ObservationEncoder: @register_obs_key called after @make"
        assert (
            name not in self.obs_shapes
        ), "ObservationEncoder: modality {} already exists".format(name)

        if net is not None:
            assert isinstance(
                net, Module
            ), "ObservationEncoder: @net must be instance of Module class"
            assert (
                (net_class is None)
                and (net_kwargs is None)
                and (share_net_from is None)
            ), "ObservationEncoder: @net provided - ignore other net creation options"

        if share_net_from is not None:
            # share processing with another modality
            assert (net_class is None) and (net_kwargs is None)
            assert share_net_from in self.obs_shapes

        net_kwargs = deepcopy(net_kwargs) if net_kwargs is not None else {}
        for rand in randomizers:
            if rand is not None:
                assert isinstance(rand, Randomizer)
                if net_kwargs is not None:
                    # update input shape to visual core
                    net_kwargs["input_shape"] = rand.output_shape_in(shape)

        self.obs_shapes[name] = shape
        self.obs_nets_classes[name] = net_class
        self.obs_nets_kwargs[name] = net_kwargs
        self.obs_nets[name] = net
        self.obs_randomizers[name] = nn.ModuleList(randomizers)
        self.obs_share_mods[name] = share_net_from

    def make(self):
        """
        Creates the encoder networks and locks the encoder so that more modalities cannot be added.
        """
        assert not self._locked, "ObservationEncoder: @make called more than once"
        self._create_layers()
        self._locked = True

    def _create_layers(self):
        """
        Creates all networks and layers required by this encoder using the registered modalities.
        """
        assert not self._locked, "ObservationEncoder: layers have already been created"

        for k in self.obs_shapes:
            if self.obs_nets_classes[k] is not None:
                # create net to process this modality
                self.obs_nets[k] = ObsUtils.OBS_ENCODER_CORES[self.obs_nets_classes[k]](
                    **self.obs_nets_kwargs[k]
                )
            elif self.obs_share_mods[k] is not None:
                # make sure net is shared with another modality
                self.obs_nets[k] = self.obs_nets[self.obs_share_mods[k]]

        self.activation = None
        if self.feature_activation is not None:
            self.activation = self.feature_activation()

    def forward(self, obs_dict):
        """
        Processes modalities according to the ordering in @self.obs_shapes. For each
        modality, it is processed with a randomizer (if present), an encoder
        network (if present), and again with the randomizer (if present), flattened,
        and then concatenated with the other processed modalities.

        Args:
            obs_dict (OrderedDict): dictionary that maps modalities to torch.Tensor
                batches that agree with @self.obs_shapes. All modalities in
                @self.obs_shapes must be present, but additional modalities
                can also be present.

        Returns:
            feats (torch.Tensor): flat features of shape [B, D]
        """
        assert self._locked, "ObservationEncoder: @make has not been called yet"

        # ensure all modalities that the encoder handles are present
        assert set(self.obs_shapes.keys()).issubset(
            obs_dict
        ), "ObservationEncoder: {} does not contain all modalities {}".format(
            list(obs_dict.keys()), list(self.obs_shapes.keys())
        )

        # process modalities by order given by @self.obs_shapes
        feats = []
        for k in self.obs_shapes:
            x = obs_dict[k]
            # maybe process encoder input with randomizer
            for rand in self.obs_randomizers[k]:
                if rand is not None:
                    x = rand.forward_in(x)
            # maybe process with obs net
            if self.obs_nets[k] is not None:
                # special case: ResNet18ConvFiLM also expects lang embedding
                if isinstance(self.obs_nets[k], VisualCoreLanguageConditioned):
                    x = self.obs_nets[k](x, lang_emb=obs_dict[LANG_EMB_KEY])
                else:
                    x = self.obs_nets[k](x)

                if self.activation is not None:
                    x = self.activation(x)
            # maybe process encoder output with randomizer
            for rand in self.obs_randomizers[k]:
                if rand is not None:
                    x = rand.forward_out(x)
            # flatten to [B, D]
            if k != LANG_EMB_KEY or not isinstance(
                self.obs_nets[k], VisualCoreLanguageConditioned
            ):
                x = TensorUtils.flatten(x, begin_axis=1)
                feats.append(x)

        # concatenate all features together
        return torch.cat(feats, dim=-1)

    def output_shape(self, input_shape=None):
        """
        Compute the output shape of the encoder.
        """
        feat_dim = 0
        for k in self.obs_shapes:
            feat_shape = self.obs_shapes[k]
            for rand in self.obs_randomizers[k]:
                if rand is not None:
                    feat_shape = rand.output_shape_in(feat_shape)
            if self.obs_nets[k] is not None:
                feat_shape = self.obs_nets[k].output_shape(feat_shape)
            for rand in self.obs_randomizers[k]:
                if rand is not None:
                    feat_shape = rand.output_shape_out(feat_shape)

            if k != LANG_EMB_KEY or not isinstance(
                self.obs_nets[k], VisualCoreLanguageConditioned
            ):
                feat_dim += int(np.prod(feat_shape))
        return [feat_dim]

    def __repr__(self):
        """
        Pretty print the encoder.
        """
        header = "{}".format(str(self.__class__.__name__))
        msg = ""
        for k in self.obs_shapes:
            msg += textwrap.indent("\nKey(\n", " " * 4)
            indent = " " * 8
            msg += textwrap.indent(
                "name={}\nshape={}\n".format(k, self.obs_shapes[k]), indent
            )
            msg += textwrap.indent(
                "modality={}\n".format(ObsUtils.OBS_KEYS_TO_MODALITIES[k]), indent
            )
            msg += textwrap.indent(
                "randomizer={}\n".format(self.obs_randomizers[k]), indent
            )
            msg += textwrap.indent("net={}\n".format(self.obs_nets[k]), indent)
            msg += textwrap.indent(
                "sharing_from={}\n".format(self.obs_share_mods[k]), indent
            )
            msg += textwrap.indent(")", " " * 4)
        msg += textwrap.indent("\noutput_shape={}".format(self.output_shape()), " " * 4)
        msg = header + "(" + msg + "\n)"
        return msg


class MCRObservationEncoder(Module):
    """
    Module that processes inputs by observation key and then concatenates the processed
    observation keys together. Each key is processed with an encoder head network.
    Call @register_obs_key to register observation keys with the encoder and then
    finally call @make to create the encoder networks.
    """

    def __init__(
        self,
        feature_activation=nn.ReLU,
        mcr_model=None,
    ):
        """
        Args:
            feature_activation: non-linearity to apply after each obs net - defaults to ReLU. Pass
                None to apply no activation.
        """
        super(MCRObservationEncoder, self).__init__()
        self.obs_shapes = OrderedDict()
        self.obs_nets_classes = OrderedDict()
        self.obs_nets_kwargs = OrderedDict()
        self.obs_share_mods = OrderedDict()
        self.obs_nets = nn.ModuleDict()
        self.obs_randomizers = nn.ModuleDict()
        self.feature_activation = feature_activation
        self.mcr_model = mcr_model
        self.mcr_transform = T.Resize((224, 224))  # Resize to 224x224
        for param in self.mcr_model.parameters():
            param.requires_grad = False

        self._locked = False

    def register_obs_key(
        self,
        name,
        shape,
        net_class=None,
        net_kwargs=None,
        net=None,
        randomizers=None,
        share_net_from=None,
    ):
        """
        Register an observation key that this encoder should be responsible for.

        Args:
            name (str): modality name
            shape (int tuple): shape of modality
            net_class (str): name of class in base_nets.py that should be used
                to process this observation key before concatenation. Pass None to flatten
                and concatenate the observation key directly.
            net_kwargs (dict): arguments to pass to @net_class
            net (Module instance): if provided, use this Module to process the observation key
                instead of creating a different net
            randomizer (Randomizer instance): if provided, use this Module to augment observation keys
                coming in to the encoder, and possibly augment the processed output as well
            share_net_from (str): if provided, use the same instance of @net_class
                as another observation key. This observation key must already exist in this encoder.
                Warning: Note that this does not share the observation key randomizer
        """
        assert (
            not self._locked
        ), "ObservationEncoder: @register_obs_key called after @make"
        assert (
            name not in self.obs_shapes
        ), "ObservationEncoder: modality {} already exists".format(name)

        if net is not None:
            assert isinstance(
                net, Module
            ), "ObservationEncoder: @net must be instance of Module class"
            assert (
                (net_class is None)
                and (net_kwargs is None)
                and (share_net_from is None)
            ), "ObservationEncoder: @net provided - ignore other net creation options"

        if share_net_from is not None:
            # share processing with another modality
            assert (net_class is None) and (net_kwargs is None)
            assert share_net_from in self.obs_shapes

        net_kwargs = deepcopy(net_kwargs) if net_kwargs is not None else {}
        for rand in randomizers:
            if rand is not None:
                assert isinstance(rand, Randomizer)
                if net_kwargs is not None:
                    # update input shape to visual core
                    net_kwargs["input_shape"] = rand.output_shape_in(shape)

        self.obs_shapes[name] = shape
        self.obs_nets_classes[name] = net_class
        self.obs_nets_kwargs[name] = net_kwargs
        self.obs_nets[name] = net
        self.obs_randomizers[name] = nn.ModuleList(randomizers)
        self.obs_share_mods[name] = share_net_from

    def make(self):
        """
        Creates the encoder networks and locks the encoder so that more modalities cannot be added.
        """
        assert not self._locked, "ObservationEncoder: @make called more than once"
        self._create_layers()
        self._locked = True

    def _create_layers(self):
        """
        Creates all networks and layers required by this encoder using the registered modalities.
        """
        assert not self._locked, "ObservationEncoder: layers have already been created"

        for k in self.obs_shapes:
            if "image" in k:
                # Use MCR Encoder
                self.obs_nets[k] = nn.Sequential(
                    self.mcr_model,
                    nn.Linear(2048, 512),
                    nn.GELU(),
                    nn.Linear(512, self.obs_nets_kwargs[k]["feature_dimension"]),
                    nn.GELU(),
                )
            else:
                # print(self.obs_nets_kwargs[k])
                if self.obs_nets_classes[k] is not None:
                    # create net to process this modality
                    self.obs_nets[k] = ObsUtils.OBS_ENCODER_CORES[
                        self.obs_nets_classes[k]
                    ](**self.obs_nets_kwargs[k])
                elif self.obs_share_mods[k] is not None:
                    # make sure net is shared with another modality
                    self.obs_nets[k] = self.obs_nets[self.obs_share_mods[k]]
        self.activation = None
        if self.feature_activation is not None:
            self.activation = self.feature_activation()

    def forward(self, obs_dict):
        """
        Processes modalities according to the ordering in @self.obs_shapes. For each
        modality, it is processed with a randomizer (if present), an encoder
        network (if present), and again with the randomizer (if present), flattened,
        and then concatenated with the other processed modalities.

        Args:
            obs_dict (OrderedDict): dictionary that maps modalities to torch.Tensor
                batches that agree with @self.obs_shapes. All modalities in
                @self.obs_shapes must be present, but additional modalities
                can also be present.

        Returns:
            feats (torch.Tensor): flat features of shape [B, D]
        """
        assert self._locked, "ObservationEncoder: @make has not been called yet"

        # ensure all modalities that the encoder handles are present
        assert set(self.obs_shapes.keys()).issubset(
            obs_dict
        ), "ObservationEncoder: {} does not contain all modalities {}".format(
            list(obs_dict.keys()), list(self.obs_shapes.keys())
        )

        # process modalities by order given by @self.obs_shapes
        feats = []
        for k in self.obs_shapes:
            x = obs_dict[k]

            if "image" in k:
                x = torch.stack([self.mcr_transform(img) for img in x])
                x = self.obs_nets[k](x)
            else:
                # maybe process encoder input with randomizer
                for rand in self.obs_randomizers[k]:
                    if rand is not None:
                        x = rand.forward_in(x)
                # maybe process with obs net
                if self.obs_nets[k] is not None:
                    # special case: ResNet18ConvFiLM also expects lang embedding
                    if isinstance(self.obs_nets[k], VisualCoreLanguageConditioned):
                        x = self.obs_nets[k](x, lang_emb=obs_dict[LANG_EMB_KEY])
                    else:
                        x = self.obs_nets[k](x)

                    if self.activation is not None:
                        x = self.activation(x)
                # maybe process encoder output with randomizer
                for rand in self.obs_randomizers[k]:
                    if rand is not None:
                        x = rand.forward_out(x)

            # flatten to [B, D]
            if k != LANG_EMB_KEY or not isinstance(
                self.obs_nets[k], VisualCoreLanguageConditioned
            ):
                x = TensorUtils.flatten(x, begin_axis=1)
                feats.append(x)

        # concatenate all features together
        return torch.cat(feats, dim=-1)

    def output_shape(self, input_shape=None):
        """
        Compute the output shape of the encoder.
        """
        feat_dim = 0
        for k in self.obs_shapes:
            feat_shape = self.obs_shapes[k]
            for rand in self.obs_randomizers[k]:
                if rand is not None:
                    feat_shape = rand.output_shape_in(feat_shape)
            if self.obs_nets[k] is not None:
                # feat_shape = self.obs_nets[k].output_shape(feat_shape)
                feat_shape = [self.obs_nets_kwargs[k]["feature_dimension"]]
            for rand in self.obs_randomizers[k]:
                if rand is not None:
                    feat_shape = rand.output_shape_out(feat_shape)

            if k != LANG_EMB_KEY or not isinstance(
                self.obs_nets[k], VisualCoreLanguageConditioned
            ):
                feat_dim += int(np.prod(feat_shape))
        return [feat_dim]

    def __repr__(self):
        """
        Pretty print the encoder.
        """
        header = "{}".format(str(self.__class__.__name__))
        msg = ""
        for k in self.obs_shapes:
            msg += textwrap.indent("\nKey(\n", " " * 4)
            indent = " " * 8
            msg += textwrap.indent(
                "name={}\nshape={}\n".format(k, self.obs_shapes[k]), indent
            )
            msg += textwrap.indent(
                "modality={}\n".format(ObsUtils.OBS_KEYS_TO_MODALITIES[k]), indent
            )
            msg += textwrap.indent(
                "randomizer={}\n".format(self.obs_randomizers[k]), indent
            )
            msg += textwrap.indent("net={}\n".format(self.obs_nets[k]), indent)
            msg += textwrap.indent(
                "sharing_from={}\n".format(self.obs_share_mods[k]), indent
            )
            msg += textwrap.indent(")", " " * 4)
        msg += textwrap.indent("\noutput_shape={}".format(self.output_shape()), " " * 4)
        msg = header + "(" + msg + "\n)"
        return msg


class ObservationDecoder(Module):
    """
    Module that can generate observation outputs by modality. Inputs are assumed
    to be flat (usually outputs from some hidden layer). Each observation output
    is generated with a linear layer from these flat inputs. Subclass this
    module in order to implement more complex schemes for generating each
    modality.
    """

    def __init__(
        self,
        decode_shapes,
        input_feat_dim,
    ):
        """
        Args:
            decode_shapes (OrderedDict): a dictionary that maps observation key to
                expected shape. This is used to generate output modalities from the
                input features.

            input_feat_dim (int): flat input dimension size
        """
        super(ObservationDecoder, self).__init__()

        # important: sort observation keys to ensure consistent ordering of modalities
        assert isinstance(decode_shapes, OrderedDict)
        self.obs_shapes = OrderedDict()
        for k in decode_shapes:
            self.obs_shapes[k] = decode_shapes[k]

        self.input_feat_dim = input_feat_dim
        self._create_layers()

    def _create_layers(self):
        """
        Create a linear layer to predict each modality.
        """
        self.nets = nn.ModuleDict()
        for k in self.obs_shapes:
            layer_out_dim = int(np.prod(self.obs_shapes[k]))
            self.nets[k] = nn.Linear(self.input_feat_dim, layer_out_dim)

    def output_shape(self, input_shape=None):
        """
        Returns output shape for this module, which is a dictionary instead
        of a list since outputs are dictionaries.
        """
        return {k: list(self.obs_shapes[k]) for k in self.obs_shapes}

    def forward(self, feats):
        """
        Predict each modality from input features, and reshape to each modality's shape.
        """
        output = {}
        for k in self.obs_shapes:
            out = self.nets[k](feats)
            output[k] = out.reshape(-1, *self.obs_shapes[k])
        return output

    def __repr__(self):
        """Pretty print network."""
        header = "{}".format(str(self.__class__.__name__))
        msg = ""
        for k in self.obs_shapes:
            msg += textwrap.indent("\nKey(\n", " " * 4)
            indent = " " * 8
            msg += textwrap.indent(
                "name={}\nshape={}\n".format(k, self.obs_shapes[k]), indent
            )
            msg += textwrap.indent(
                "modality={}\n".format(ObsUtils.OBS_KEYS_TO_MODALITIES[k]), indent
            )
            msg += textwrap.indent("net=({})\n".format(self.nets[k]), indent)
            msg += textwrap.indent(")", " " * 4)
        msg = header + "(" + msg + "\n)"
        return msg


class ICLObservationDecoder(Module):
    """
    Module that can generate observation outputs by modality. Inputs are assumed
    to be flat (usually outputs from some hidden layer). Each observation output
    is generated with a linear layer from these flat inputs. Subclass this
    module in order to implement more complex schemes for generating each
    modality.
    """

    def __init__(
        self,
        decode_shapes,
        input_feat_dim,
    ):
        """
        Args:
            decode_shapes (OrderedDict): a dictionary that maps observation key to
                expected shape. This is used to generate output modalities from the
                input features.

            input_feat_dim (int): flat input dimension size
        """
        super(ObservationDecoder, self).__init__()

        # important: sort observation keys to ensure consistent ordering of modalities
        assert isinstance(decode_shapes, OrderedDict)
        self.obs_shapes = OrderedDict()
        for k in decode_shapes:
            self.obs_shapes[k] = decode_shapes[k]

        self.input_feat_dim = input_feat_dim
        self._create_layers()

    def _create_layers(self):
        """
        Create a linear layer to predict each modality.
        """
        self.nets = nn.ModuleDict()
        for k in self.obs_shapes:
            layer_out_dim = int(np.prod(self.obs_shapes[k]))
            self.nets[k] = nn.Linear(self.input_feat_dim, layer_out_dim)

    def output_shape(self, input_shape=None):
        """
        Returns output shape for this module, which is a dictionary instead
        of a list since outputs are dictionaries.
        """
        return {k: list(self.obs_shapes[k]) for k in self.obs_shapes}

    def forward(self, feats):
        """
        Predict each modality from input features, and reshape to each modality's shape.
        """
        output = {}
        for k in self.obs_shapes:
            out = self.nets[k](feats)
            output[k] = out.reshape(-1, *self.obs_shapes[k])
        return output

    def __repr__(self):
        """Pretty print network."""
        header = "{}".format(str(self.__class__.__name__))
        msg = ""
        for k in self.obs_shapes:
            msg += textwrap.indent("\nKey(\n", " " * 4)
            indent = " " * 8
            msg += textwrap.indent(
                "name={}\nshape={}\n".format(k, self.obs_shapes[k]), indent
            )
            msg += textwrap.indent(
                "modality={}\n".format(ObsUtils.OBS_KEYS_TO_MODALITIES[k]), indent
            )
            msg += textwrap.indent("net=({})\n".format(self.nets[k]), indent)
            msg += textwrap.indent(")", " " * 4)
        msg = header + "(" + msg + "\n)"
        return msg


class ObservationGroupEncoder(Module):
    """
    This class allows networks to encode multiple observation dictionaries into a single
    flat, concatenated vector representation. It does this by assigning each observation
    dictionary (observation group) an @ObservationEncoder object.

    The class takes a dictionary of dictionaries, @observation_group_shapes.
    Each key corresponds to a observation group (e.g. 'obs', 'subgoal', 'goal')
    and each OrderedDict should be a map between modalities and
    expected input shapes (e.g. { 'image' : (3, 120, 160) }).
    """

    def __init__(
        self,
        observation_group_shapes,
        feature_activation=nn.ReLU,
        encoder_kwargs=None,
    ):
        """
        Args:
            observation_group_shapes (OrderedDict): a dictionary of dictionaries.
                Each key in this dictionary should specify an observation group, and
                the value should be an OrderedDict that maps modalities to
                expected shapes.

            feature_activation: non-linearity to apply after each obs net - defaults to ReLU. Pass
                None to apply no activation.

            encoder_kwargs (dict or None): If None, results in default encoder_kwargs being applied. Otherwise, should
                be nested dictionary containing relevant per-modality information for encoder networks.
                Should be of form:

                obs_modality1: dict
                    feature_dimension: int
                    core_class: str
                    core_kwargs: dict
                        ...
                        ...
                    obs_randomizer_class: str
                    obs_randomizer_kwargs: dict
                        ...
                        ...
                obs_modality2: dict
                    ...
        """
        super(ObservationGroupEncoder, self).__init__()

        # type checking
        assert isinstance(observation_group_shapes, OrderedDict)
        assert np.all(
            [
                isinstance(observation_group_shapes[k], OrderedDict)
                for k in observation_group_shapes
            ]
        )

        self.observation_group_shapes = observation_group_shapes

        # create an observation encoder per observation group
        self.nets = nn.ModuleDict()
        for obs_group in self.observation_group_shapes:
            self.nets[obs_group] = obs_encoder_factory(
                obs_shapes=self.observation_group_shapes[obs_group],
                feature_activation=feature_activation,
                encoder_kwargs=encoder_kwargs,
            )

    def forward(self, **inputs):
        """
        Process each set of inputs in its own observation group.

        Args:
            inputs (dict): dictionary that maps observation groups to observation
                dictionaries of torch.Tensor batches that agree with
                @self.observation_group_shapes. All observation groups in
                @self.observation_group_shapes must be present, but additional
                observation groups can also be present. Note that these are specified
                as kwargs for ease of use with networks that name each observation
                stream in their forward calls.

        Returns:
            outputs (torch.Tensor): flat outputs of shape [B, D]
        """

        # ensure all observation groups we need are present
        assert set(self.observation_group_shapes.keys()).issubset(
            inputs
        ), "{} does not contain all observation groups {}".format(
            list(inputs.keys()), list(self.observation_group_shapes.keys())
        )

        outputs = []
        # Deterministic order since self.observation_group_shapes is OrderedDict
        for obs_group in self.observation_group_shapes:
            # pass through encoder
            outputs.append(self.nets[obs_group].forward(inputs[obs_group]))

        return torch.cat(outputs, dim=-1)

    def output_shape(self):
        """
        Compute the output shape of this encoder.
        """
        feat_dim = 0
        for obs_group in self.observation_group_shapes:
            # get feature dimension of these keys
            feat_dim += self.nets[obs_group].output_shape()[0]
        return [feat_dim]

    def __repr__(self):
        """Pretty print network."""
        header = "{}".format(str(self.__class__.__name__))
        msg = ""
        for k in self.observation_group_shapes:
            msg += "\n"
            indent = " " * 4
            msg += textwrap.indent("group={}\n{}".format(k, self.nets[k]), indent)
        msg = header + "(" + msg + "\n)"
        return msg


class MCRObservationGroupEncoder(Module):
    """
    This class allows networks to encode multiple observation dictionaries into a single
    flat, concatenated vector representation. It does this by assigning each observation
    dictionary (observation group) an @ObservationEncoder object.

    The class takes a dictionary of dictionaries, @observation_group_shapes.
    Each key corresponds to a observation group (e.g. 'obs', 'subgoal', 'goal')
    and each OrderedDict should be a map between modalities and
    expected input shapes (e.g. { 'image' : (3, 120, 160) }).
    """

    def __init__(
        self,
        observation_group_shapes,
        feature_activation=nn.ReLU,
        encoder_kwargs=None,
        mcr_model=None,
    ):
        """
        Args:
            observation_group_shapes (OrderedDict): a dictionary of dictionaries.
                Each key in this dictionary should specify an observation group, and
                the value should be an OrderedDict that maps modalities to
                expected shapes.

            feature_activation: non-linearity to apply after each obs net - defaults to ReLU. Pass
                None to apply no activation.

            encoder_kwargs (dict or None): If None, results in default encoder_kwargs being applied. Otherwise, should
                be nested dictionary containing relevant per-modality information for encoder networks.
                Should be of form:

                obs_modality1: dict
                    feature_dimension: int
                    core_class: str
                    core_kwargs: dict
                        ...
                        ...
                    obs_randomizer_class: str
                    obs_randomizer_kwargs: dict
                        ...
                        ...
                obs_modality2: dict
                    ...
        """
        super(MCRObservationGroupEncoder, self).__init__()

        # type checking
        assert isinstance(observation_group_shapes, OrderedDict)
        assert np.all(
            [
                isinstance(observation_group_shapes[k], OrderedDict)
                for k in observation_group_shapes
            ]
        )

        self.observation_group_shapes = observation_group_shapes

        self.mcr_model = mcr_model

        # create an observation encoder per observation group
        self.nets = nn.ModuleDict()
        for obs_group in self.observation_group_shapes:
            self.nets[obs_group] = mcr_obs_encoder_factory(
                obs_shapes=self.observation_group_shapes[obs_group],
                feature_activation=feature_activation,
                encoder_kwargs=encoder_kwargs,
                mcr_model=self.mcr_model,
            )
            # self.nets[obs_group] = obs_encoder_factory(
            #     obs_shapes=self.observation_group_shapes[obs_group],
            #     feature_activation=feature_activation,
            #     encoder_kwargs=encoder_kwargs,
            # )

    def forward(self, **inputs):
        """
        Process each set of inputs in its own observation group.

        Args:
            inputs (dict): dictionary that maps observation groups to observation
                dictionaries of torch.Tensor batches that agree with
                @self.observation_group_shapes. All observation groups in
                @self.observation_group_shapes must be present, but additional
                observation groups can also be present. Note that these are specified
                as kwargs for ease of use with networks that name each observation
                stream in their forward calls.

        Returns:
            outputs (torch.Tensor): flat outputs of shape [B, D]
        """

        # ensure all observation groups we need are present
        assert set(self.observation_group_shapes.keys()).issubset(
            inputs
        ), "{} does not contain all observation groups {}".format(
            list(inputs.keys()), list(self.observation_group_shapes.keys())
        )

        outputs = []
        # Deterministic order since self.observation_group_shapes is OrderedDict
        for obs_group in self.observation_group_shapes:
            # pass through encoder
            outputs.append(self.nets[obs_group].forward(inputs[obs_group]))
        return torch.cat(outputs, dim=-1)

    def output_shape(self):
        """
        Compute the output shape of this encoder.
        """
        feat_dim = 0
        for obs_group in self.observation_group_shapes:
            # get feature dimension of these keys
            feat_dim += self.nets[obs_group].output_shape()[0]
        return [feat_dim]

    def __repr__(self):
        """Pretty print network."""
        header = "{}".format(str(self.__class__.__name__))
        msg = ""
        for k in self.observation_group_shapes:
            msg += "\n"
            indent = " " * 4
            msg += textwrap.indent("group={}\n{}".format(k, self.nets[k]), indent)
        msg = header + "(" + msg + "\n)"
        return msg


class ICLObservationGroupEncoder(Module):
    """
    This class allows networks to encode multiple observation dictionaries into a single
    flat, concatenated vector representation. It does this by assigning each observation
    dictionary (observation group) an @ObservationEncoder object.

    The class takes a dictionary of dictionaries, @observation_group_shapes.
    Each key corresponds to a observation group (e.g. 'obs', 'subgoal', 'goal')
    and each OrderedDict should be a map between modalities and
    expected input shapes (e.g. { 'image' : (3, 120, 160) }).
    """

    def __init__(
        self,
        observation_group_shapes,
        action_input_shape,
        fast_enabled=False,
        bin_enabled=False,
        vq_vae_enabled=False,
        ln_act_enabled=False,
        feature_activation=nn.ReLU,
        encoder_kwargs=None,
    ):
        """
        Args:
            observation_group_shapes (OrderedDict): a dictionary of dictionaries.
                Each key in this dictionary should specify an observation group, and
                the value should be an OrderedDict that maps modalities to
                expected shapes.

            feature_activation: non-linearity to apply after each obs net - defaults to ReLU. Pass
                None to apply no activation.

            encoder_kwargs (dict or None): If None, results in default encoder_kwargs being applied. Otherwise, should
                be nested dictionary containing relevant per-modality information for encoder networks.
                Should be of form:

                obs_modality1: dict
                    feature_dimension: int
                    core_class: str
                    core_kwargs: dict
                        ...
                        ...
                    obs_randomizer_class: str
                    obs_randomizer_kwargs: dict
                        ...
                        ...
                obs_modality2: dict
                    ...
        """
        super(ICLObservationGroupEncoder, self).__init__()

        # type checking
        assert isinstance(observation_group_shapes, OrderedDict)
        assert np.all(
            [
                isinstance(observation_group_shapes[k], OrderedDict)
                for k in observation_group_shapes
            ]
        )

        self.observation_group_shapes = observation_group_shapes

        # create an observation encoder per observation group
        self.nets = nn.ModuleDict()
        for obs_group in self.observation_group_shapes:
            self.nets[obs_group] = obs_encoder_factory(
                obs_shapes=self.observation_group_shapes[obs_group],
                feature_activation=feature_activation,
                encoder_kwargs=encoder_kwargs,
            )

        # Create encoder for action
        action_output_shape = self.output_shape()[0]

        self.fast_enabled = fast_enabled
        self.bin_enabled = bin_enabled
        self.vq_vae_enabled = vq_vae_enabled
        self.ln_act_enabled = ln_act_enabled
        if fast_enabled:
            self.action_tokenizer = AutoProcessor.from_pretrained(
                "physical-intelligence/fast", trust_remote_code=True
            ).from_pretrained("expdata/robocasa/fast_tokenizer")

            self.clip_model, _ = clip.load("ViT-B/32")

            self.action_network = nn.Sequential(
                nn.Linear(512, 64),
                nn.GELU(),
                nn.Linear(64, 128),
                nn.GELU(),
                nn.Linear(128, action_output_shape),
            )

        elif bin_enabled:
            self.action_network = AdaptiveBinActionEmbedding(
                action_dim=action_input_shape, output_dim=action_output_shape
            )

        elif vq_vae_enabled:
            # self.action_network = VQVAE(
            #     feature_dim=action_input_shape, latent_dim=action_output_shape
            # )
            # # self.action_network = LFQVAE(
            #     feature_dim=action_input_shape, latent_dim=action_output_shape
            # )
            # self.action_network = LLFQVAE(
            #     feature_dim=action_input_shape, latent_dim=action_output_shape
            # )
            self.action_network = LLFQVAE_V3(
                feature_dim=action_input_shape, latent_dim=action_output_shape
            )
        elif ln_act_enabled:
            self.action_network = Mamba(
                d_model=action_input_shape,  # Model dimension d_model
                d_state=8,  # SSM state expansion factor
                d_conv=4,  # Local convolution width
                expand=2,  # Block expansion factor
            )

            self.ln_act_layer = nn.Sequential(
                nn.Linear(action_input_shape, 64),
                nn.GELU(),
                nn.Linear(64, 128),
                nn.GELU(),
                nn.Linear(128, action_output_shape),
            )

        else:
            transformer_layer = nn.TransformerEncoderLayer(
                d_model=action_output_shape,
                nhead=8,
                dim_feedforward=256,
                activation="gelu",
            )

            self.action_network = nn.Sequential(
                nn.Linear(action_input_shape, 64),
                nn.GELU(),
                nn.Linear(64, 128),
                nn.GELU(),
                nn.Linear(128, action_output_shape),
                nn.TransformerEncoder(transformer_layer, num_layers=4),
                nn.Linear(action_output_shape, action_output_shape),
            )

        self._vis_counter = 0

    def forward(self, **inputs):
        """
        Process each set of inputs in its own observation group.

        Args:
            inputs (dict): dictionary that maps observation groups to observation
                dictionaries of torch.Tensor batches that agree with
                @self.observation_group_shapes. All observation groups in
                @self.observation_group_shapes must be present, but additional
                observation groups can also be present. Note that these are specified
                as kwargs for ease of use with networks that name each observation
                stream in their forward calls.

        Returns:
            outputs (torch.Tensor): flat outputs of shape [B, D]
        """

        # Process the prompt
        prompt_obs = inputs["prompt"]["obs"]
        prompt_actions = inputs["prompt"]["action"]

        # ensure all observation groups we need are present
        assert set(self.observation_group_shapes.keys()).issubset(
            inputs
        ), "{} does not contain all observation groups {}".format(
            list(inputs.keys()), list(self.observation_group_shapes.keys())
        )

        outputs = []
        # Deterministic order since self.observation_group_shapes is OrderedDict
        for obs_group in self.observation_group_shapes:
            # pass through encoder
            outputs.append(self.nets[obs_group].forward(inputs[obs_group]))

        seq_len = 10  # FIXME
        batch_size = outputs[0].data.shape[0]
        batch_size = int(batch_size / seq_len)

        obs = torch.cat(outputs, dim=-1)
        context_obs = self.nets["obs"].forward(prompt_obs)
        context_obs = torch.cat([context_obs], dim=-1)

        if self.fast_enabled:
            prompt_actions = prompt_actions.view(batch_size, seq_len, -1)
            aggregated_vector_list = []
            for idx in range(batch_size):
                prompt_actions_idx = prompt_actions[idx]
                tokens = self.action_tokenizer(prompt_actions_idx.cpu().numpy())
                clip_tokens = clip.tokenize(list(map(str, tokens[0]))).to(
                    obs.device
                )  # Tokenize using CLIP's tokenizer

                with torch.no_grad():
                    latent_vector = self.clip_model.encode_text(clip_tokens)

                # Normalize the latent vector
                latent_vector = latent_vector / latent_vector.norm(dim=-1, keepdim=True)
                D, dim = latent_vector.shape

                if D >= seq_len:
                    indices = torch.linspace(0, D - 1, steps=seq_len).long()
                    aggregated_vector = latent_vector[indices]
                else:
                    aggregated_vector = torch.zeros(
                        seq_len, dim, device=latent_vector.device
                    )
                    aggregated_vector[:D] = latent_vector
                aggregated_vector_list.append(aggregated_vector)

            context_actions = torch.cat(aggregated_vector_list, dim=0)
            context_actions = self.action_network(context_actions).squeeze(0)
        elif self.vq_vae_enabled:
            context_actions, loss = self.action_network(prompt_actions)
            self._vq_vae_loss = loss
        elif self.ln_act_enabled:
            prompt_actions = prompt_actions.view(batch_size, seq_len, -1)
            context_actions = self.action_network(prompt_actions)
            context_actions = context_actions.view(batch_size * seq_len, -1)
            context_actions = self.ln_act_layer(context_actions)
        else:
            context_actions = self.action_network(prompt_actions)

        # if self._vis_counter == 0:
        #     self._store_vis = [context_actions]
        # else:
        #     self._store_vis.append(context_actions)
        # self._vis_counter += 1
        # if self._vis_counter == 10:
        #     self._store_vis = torch.cat(self._store_vis, dim=0)
        #     print(self._store_vis.data.shape)
        #     torch.save(
        #         self._store_vis,
        #         "/home/anvuong/Desktop/robocasa/expdata/robocasa/action/proposed_action_v4.pt",
        #     )
        #     import sys

        #     sys.exit()
        #     sys.exit()
        return obs, context_obs, context_actions

    def output_shape(self):
        """
        Compute the output shape of this encoder.
        """
        feat_dim = 0
        for obs_group in self.observation_group_shapes:
            # get feature dimension of these keys
            feat_dim += self.nets[obs_group].output_shape()[0]
        return [feat_dim]

    def __repr__(self):
        """Pretty print network."""
        header = "{}".format(str(self.__class__.__name__))
        msg = ""
        for k in self.observation_group_shapes:
            msg += "\n"
            indent = " " * 4
            msg += textwrap.indent("group={}\n{}".format(k, self.nets[k]), indent)
        msg = header + "(" + msg + "\n)"
        return msg


class MIMO_MLP(Module):
    """
    Extension to MLP to accept multiple observation dictionaries as input and
    to output dictionaries of tensors. Inputs are specified as a dictionary of
    observation dictionaries, with each key corresponding to an observation group.

    This module utilizes @ObservationGroupEncoder to process the multiple input dictionaries and
    @ObservationDecoder to generate tensor dictionaries. The default behavior
    for encoding the inputs is to process visual inputs with a learned CNN and concatenating
    the flat encodings with the other flat inputs. The default behavior for generating
    outputs is to use a linear layer branch to produce each modality separately
    (including visual outputs).
    """

    def __init__(
        self,
        input_obs_group_shapes,
        output_shapes,
        layer_dims,
        layer_func=nn.Linear,
        activation=nn.ReLU,
        encoder_kwargs=None,
    ):
        """
        Args:
            input_obs_group_shapes (OrderedDict): a dictionary of dictionaries.
                Each key in this dictionary should specify an observation group, and
                the value should be an OrderedDict that maps modalities to
                expected shapes.

            output_shapes (OrderedDict): a dictionary that maps modality to
                expected shapes for outputs.

            layer_dims ([int]): sequence of integers for the MLP hidden layer sizes

            layer_func: mapping per MLP layer - defaults to Linear

            activation: non-linearity per MLP layer - defaults to ReLU

            encoder_kwargs (dict or None): If None, results in default encoder_kwargs being applied. Otherwise, should
                be nested dictionary containing relevant per-modality information for encoder networks.
                Should be of form:

                obs_modality1: dict
                    feature_dimension: int
                    core_class: str
                    core_kwargs: dict
                        ...
                        ...
                    obs_randomizer_class: str
                    obs_randomizer_kwargs: dict
                        ...
                        ...
                obs_modality2: dict
                    ...
        """
        super(MIMO_MLP, self).__init__()

        assert isinstance(input_obs_group_shapes, OrderedDict)
        assert np.all(
            [
                isinstance(input_obs_group_shapes[k], OrderedDict)
                for k in input_obs_group_shapes
            ]
        )
        assert isinstance(output_shapes, OrderedDict)

        self.input_obs_group_shapes = input_obs_group_shapes
        self.output_shapes = output_shapes

        self.nets = nn.ModuleDict()

        # Encoder for all observation groups.
        self.nets["encoder"] = ObservationGroupEncoder(
            observation_group_shapes=input_obs_group_shapes,
            encoder_kwargs=encoder_kwargs,
        )

        # flat encoder output dimension
        mlp_input_dim = self.nets["encoder"].output_shape()[0]

        # intermediate MLP layers
        self.nets["mlp"] = MLP(
            input_dim=mlp_input_dim,
            output_dim=layer_dims[-1],
            layer_dims=layer_dims[:-1],
            layer_func=layer_func,
            activation=activation,
            output_activation=activation,  # make sure non-linearity is applied before decoder
        )

        # decoder for output modalities
        self.nets["decoder"] = ObservationDecoder(
            decode_shapes=self.output_shapes,
            input_feat_dim=layer_dims[-1],
        )

    def output_shape(self, input_shape=None):
        """
        Returns output shape for this module, which is a dictionary instead
        of a list since outputs are dictionaries.
        """
        return {k: list(self.output_shapes[k]) for k in self.output_shapes}

    def forward(self, **inputs):
        """
        Process each set of inputs in its own observation group.

        Args:
            inputs (dict): a dictionary of dictionaries with one dictionary per
                observation group. Each observation group's dictionary should map
                modality to torch.Tensor batches. Should be consistent with
                @self.input_obs_group_shapes.

        Returns:
            outputs (dict): dictionary of output torch.Tensors, that corresponds
                to @self.output_shapes
        """
        enc_outputs = self.nets["encoder"](**inputs)
        mlp_out = self.nets["mlp"](enc_outputs)
        return self.nets["decoder"](mlp_out)

    def _to_string(self):
        """
        Subclasses should override this method to print out info about network / policy.
        """
        return ""

    def __repr__(self):
        """Pretty print network."""
        header = "{}".format(str(self.__class__.__name__))
        msg = ""
        indent = " " * 4
        if self._to_string() != "":
            msg += textwrap.indent("\n" + self._to_string() + "\n", indent)
        msg += textwrap.indent("\nencoder={}".format(self.nets["encoder"]), indent)
        msg += textwrap.indent("\n\nmlp={}".format(self.nets["mlp"]), indent)
        msg += textwrap.indent("\n\ndecoder={}".format(self.nets["decoder"]), indent)
        msg = header + "(" + msg + "\n)"
        return msg


class RNN_MIMO_MLP(Module):
    """
    A wrapper class for a multi-step RNN and a per-step MLP and a decoder.

    Structure: [encoder -> rnn -> mlp -> decoder]

    All temporal inputs are processed by a shared @ObservationGroupEncoder,
    followed by an RNN, and then a per-step multi-output MLP.
    """

    def __init__(
        self,
        input_obs_group_shapes,
        output_shapes,
        mlp_layer_dims,
        rnn_hidden_dim,
        rnn_num_layers,
        rnn_type="LSTM",  # [LSTM, GRU]
        rnn_kwargs=None,
        mlp_activation=nn.ReLU,
        mlp_layer_func=nn.Linear,
        per_step=True,
        encoder_kwargs=None,
    ):
        """
        Args:
            input_obs_group_shapes (OrderedDict): a dictionary of dictionaries.
                Each key in this dictionary should specify an observation group, and
                the value should be an OrderedDict that maps modalities to
                expected shapes.

            output_shapes (OrderedDict): a dictionary that maps modality to
                expected shapes for outputs.

            rnn_hidden_dim (int): RNN hidden dimension

            rnn_num_layers (int): number of RNN layers

            rnn_type (str): [LSTM, GRU]

            rnn_kwargs (dict): kwargs for the rnn model

            per_step (bool): if True, apply the MLP and observation decoder into @output_shapes
                at every step of the RNN. Otherwise, apply them to the final hidden state of the
                RNN.

            encoder_kwargs (dict or None): If None, results in default encoder_kwargs being applied. Otherwise, should
                be nested dictionary containing relevant per-modality information for encoder networks.
                Should be of form:

                obs_modality1: dict
                    feature_dimension: int
                    core_class: str
                    core_kwargs: dict
                        ...
                        ...
                    obs_randomizer_class: str
                    obs_randomizer_kwargs: dict
                        ...
                        ...
                obs_modality2: dict
                    ...
        """
        super(RNN_MIMO_MLP, self).__init__()
        assert isinstance(input_obs_group_shapes, OrderedDict)
        assert np.all(
            [
                isinstance(input_obs_group_shapes[k], OrderedDict)
                for k in input_obs_group_shapes
            ]
        )
        assert isinstance(output_shapes, OrderedDict)
        self.input_obs_group_shapes = input_obs_group_shapes
        self.output_shapes = output_shapes
        self.per_step = per_step

        self.nets = nn.ModuleDict()

        # Encoder for all observation groups.
        self.nets["encoder"] = ObservationGroupEncoder(
            observation_group_shapes=input_obs_group_shapes,
            encoder_kwargs=encoder_kwargs,
        )

        # flat encoder output dimension
        rnn_input_dim = self.nets["encoder"].output_shape()[0]

        # bidirectional RNNs mean that the output of RNN will be twice the hidden dimension
        rnn_is_bidirectional = rnn_kwargs.get("bidirectional", False)
        num_directions = (
            int(rnn_is_bidirectional) + 1
        )  # 2 if bidirectional, 1 otherwise
        rnn_output_dim = num_directions * rnn_hidden_dim

        per_step_net = None
        self._has_mlp = len(mlp_layer_dims) > 0
        if self._has_mlp:
            self.nets["mlp"] = MLP(
                input_dim=rnn_output_dim,
                output_dim=mlp_layer_dims[-1],
                layer_dims=mlp_layer_dims[:-1],
                output_activation=mlp_activation,
                layer_func=mlp_layer_func,
            )
            self.nets["decoder"] = ObservationDecoder(
                decode_shapes=self.output_shapes,
                input_feat_dim=mlp_layer_dims[-1],
            )
            if self.per_step:
                per_step_net = Sequential(self.nets["mlp"], self.nets["decoder"])
        else:
            self.nets["decoder"] = ObservationDecoder(
                decode_shapes=self.output_shapes,
                input_feat_dim=rnn_output_dim,
            )
            if self.per_step:
                per_step_net = self.nets["decoder"]

        # core network
        self.nets["rnn"] = RNN_Base(
            input_dim=rnn_input_dim,
            rnn_hidden_dim=rnn_hidden_dim,
            rnn_num_layers=rnn_num_layers,
            rnn_type=rnn_type,
            per_step_net=per_step_net,
            rnn_kwargs=rnn_kwargs,
        )

    def get_rnn_init_state(self, batch_size, device):
        """
        Get a default RNN state (zeros)

        Args:
            batch_size (int): batch size dimension

            device: device the hidden state should be sent to.

        Returns:
            hidden_state (torch.Tensor or tuple): returns hidden state tensor or tuple of hidden state tensors
                depending on the RNN type
        """
        return self.nets["rnn"].get_rnn_init_state(batch_size, device=device)

    def output_shape(self, input_shape):
        """
        Returns output shape for this module, which is a dictionary instead
        of a list since outputs are dictionaries.

        Args:
            input_shape (dict): dictionary of dictionaries, where each top-level key
                corresponds to an observation group, and the low-level dictionaries
                specify the shape for each modality in an observation dictionary
        """

        # infers temporal dimension from input shape
        obs_group = list(self.input_obs_group_shapes.keys())[0]
        mod = list(self.input_obs_group_shapes[obs_group].keys())[0]
        T = input_shape[obs_group][mod][0]
        TensorUtils.assert_size_at_dim(
            input_shape,
            size=T,
            dim=0,
            msg="RNN_MIMO_MLP: input_shape inconsistent in temporal dimension",
        )
        # returns a dictionary instead of list since outputs are dictionaries
        return {k: [T] + list(self.output_shapes[k]) for k in self.output_shapes}

    def forward(self, rnn_init_state=None, return_state=False, **inputs):
        """
        Args:
            inputs (dict): a dictionary of dictionaries with one dictionary per
                observation group. Each observation group's dictionary should map
                modality to torch.Tensor batches. Should be consistent with
                @self.input_obs_group_shapes. First two leading dimensions should
                be batch and time [B, T, ...] for each tensor.

            rnn_init_state: rnn hidden state, initialize to zero state if set to None

            return_state (bool): whether to return hidden state

        Returns:
            outputs (dict): dictionary of output torch.Tensors, that corresponds
                to @self.output_shapes. Leading dimensions will be batch and time [B, T, ...]
                for each tensor.

            rnn_state (torch.Tensor or tuple): return the new rnn state (if @return_state)
        """
        for obs_group in self.input_obs_group_shapes:
            for k in self.input_obs_group_shapes[obs_group]:
                # first two dimensions should be [B, T] for inputs
                assert inputs[obs_group][k].ndim - 2 == len(
                    self.input_obs_group_shapes[obs_group][k]
                )

        # use encoder to extract flat rnn inputs
        rnn_inputs = TensorUtils.time_distributed(
            inputs, self.nets["encoder"], inputs_as_kwargs=True
        )
        assert rnn_inputs.ndim == 3  # [B, T, D]
        if self.per_step:
            return self.nets["rnn"].forward(
                inputs=rnn_inputs,
                rnn_init_state=rnn_init_state,
                return_state=return_state,
            )

        # apply MLP + decoder to last RNN output
        outputs = self.nets["rnn"].forward(
            inputs=rnn_inputs, rnn_init_state=rnn_init_state, return_state=return_state
        )
        if return_state:
            outputs, rnn_state = outputs

        assert outputs.ndim == 3  # [B, T, D]
        if self._has_mlp:
            outputs = self.nets["decoder"](self.nets["mlp"](outputs[:, -1]))
        else:
            outputs = self.nets["decoder"](outputs[:, -1])

        if return_state:
            return outputs, rnn_state
        return outputs

    def forward_step(self, rnn_state, **inputs):
        """
        Unroll network over a single timestep.

        Args:
            inputs (dict): expects same modalities as @self.input_shapes, with
                additional batch dimension (but NOT time), since this is a
                single time step.

            rnn_state (torch.Tensor): rnn hidden state

        Returns:
            outputs (dict): dictionary of output torch.Tensors, that corresponds
                to @self.output_shapes. Does not contain time dimension.

            rnn_state: return the new rnn state
        """
        # ensure that the only extra dimension is batch dim, not temporal dim
        assert np.all(
            [inputs[k].ndim - 1 == len(self.input_shapes[k]) for k in self.input_shapes]
        )

        inputs = TensorUtils.to_sequence(inputs)
        outputs, rnn_state = self.forward(
            inputs,
            rnn_init_state=rnn_state,
            return_state=True,
        )
        if self.per_step:
            # if outputs are not per-step, the time dimension is already reduced
            outputs = outputs[:, 0]
        return outputs, rnn_state

    def _to_string(self):
        """
        Subclasses should override this method to print out info about network / policy.
        """
        return ""

    def __repr__(self):
        """Pretty print network."""
        header = "{}".format(str(self.__class__.__name__))
        msg = ""
        indent = " " * 4
        msg += textwrap.indent("\n" + self._to_string(), indent)
        msg += textwrap.indent("\n\nencoder={}".format(self.nets["encoder"]), indent)
        msg += textwrap.indent("\n\nrnn={}".format(self.nets["rnn"]), indent)
        msg = header + "(" + msg + "\n)"
        return msg


class MIMO_Transformer(Module):
    """
    Extension to Transformer (based on GPT architecture) to accept multiple observation
    dictionaries as input and to output dictionaries of tensors. Inputs are specified as
    a dictionary of observation dictionaries, with each key corresponding to an observation group.
    This module utilizes @ObservationGroupEncoder to process the multiple input dictionaries and
    @ObservationDecoder to generate tensor dictionaries. The default behavior
    for encoding the inputs is to process visual inputs with a learned CNN and concatenating
    the flat encodings with the other flat inputs. The default behavior for generating
    outputs is to use a linear layer branch to produce each modality separately
    (including visual outputs).
    """

    def __init__(
        self,
        input_obs_group_shapes,
        output_shapes,
        transformer_embed_dim,
        transformer_num_layers,
        transformer_num_heads,
        transformer_context_length,
        transformer_causal=True,
        transformer_emb_dropout=0.1,
        transformer_attn_dropout=0.1,
        transformer_block_output_dropout=0.1,
        transformer_sinusoidal_embedding=False,
        transformer_activation="gelu",
        transformer_nn_parameter_for_timesteps=False,
        encoder_kwargs=None,
    ):
        """
        Args:
            input_obs_group_shapes (OrderedDict): a dictionary of dictionaries.
                Each key in this dictionary should specify an observation group, and
                the value should be an OrderedDict that maps modalities to
                expected shapes.
            output_shapes (OrderedDict): a dictionary that maps modality to
                expected shapes for outputs.
            transformer_embed_dim (int): dimension for embeddings used by transformer
            transformer_num_layers (int): number of transformer blocks to stack
            transformer_num_heads (int): number of attention heads for each
                transformer block - must divide @transformer_embed_dim evenly. Self-attention is
                computed over this many partitions of the embedding dimension separately.
            transformer_causal (bool): whether to use causal transformer layers
            transformer_context_length (int): expected length of input sequences
            transformer_activation: non-linearity for input and output layers used in transformer
            transformer_emb_dropout (float): dropout probability for embedding inputs in transformer
            transformer_attn_dropout (float): dropout probability for attention outputs for each transformer block
            transformer_block_output_dropout (float): dropout probability for final outputs for each transformer block
            encoder_kwargs (dict): observation encoder config
        """
        super(MIMO_Transformer, self).__init__()

        assert isinstance(input_obs_group_shapes, OrderedDict)
        assert np.all(
            [
                isinstance(input_obs_group_shapes[k], OrderedDict)
                for k in input_obs_group_shapes
            ]
        )
        assert isinstance(output_shapes, OrderedDict)

        self.input_obs_group_shapes = input_obs_group_shapes
        self.output_shapes = output_shapes

        self.nets = nn.ModuleDict()
        self.params = nn.ParameterDict()

        # Encoder for all observation groups.
        self.nets["encoder"] = ObservationGroupEncoder(
            observation_group_shapes=input_obs_group_shapes,
            encoder_kwargs=encoder_kwargs,
            feature_activation=None,
        )

        # flat encoder output dimension
        transformer_input_dim = self.nets["encoder"].output_shape()[0]
        self.nets["embed_encoder"] = nn.Linear(
            transformer_input_dim, transformer_embed_dim
        )

        max_timestep = transformer_context_length

        if transformer_sinusoidal_embedding:
            self.nets["embed_timestep"] = PositionalEncoding(transformer_embed_dim)
        elif transformer_nn_parameter_for_timesteps:
            assert (
                not transformer_sinusoidal_embedding
            ), "nn.Parameter only works with learned embeddings"
            self.params["embed_timestep"] = nn.Parameter(
                torch.zeros(1, max_timestep, transformer_embed_dim)
            )
        else:
            self.nets["embed_timestep"] = nn.Embedding(
                max_timestep, transformer_embed_dim
            )

        # layer norm for embeddings
        self.nets["embed_ln"] = nn.LayerNorm(transformer_embed_dim)

        # dropout for input embeddings
        self.nets["embed_drop"] = nn.Dropout(transformer_emb_dropout)

        # GPT transformer
        self.nets["transformer"] = GPT_Backbone(
            embed_dim=transformer_embed_dim,
            num_layers=transformer_num_layers,
            num_heads=transformer_num_heads,
            context_length=transformer_context_length,
            causal=transformer_causal,
            attn_dropout=transformer_attn_dropout,
            block_output_dropout=transformer_block_output_dropout,
            activation=transformer_activation,
        )

        # decoder for output modalities
        self.nets["decoder"] = ObservationDecoder(
            decode_shapes=self.output_shapes,
            input_feat_dim=transformer_embed_dim,
        )

        self.transformer_context_length = transformer_context_length
        self.transformer_embed_dim = transformer_embed_dim
        self.transformer_sinusoidal_embedding = transformer_sinusoidal_embedding
        self.transformer_nn_parameter_for_timesteps = (
            transformer_nn_parameter_for_timesteps
        )

    def output_shape(self, input_shape=None):
        """
        Returns output shape for this module, which is a dictionary instead
        of a list since outputs are dictionaries.
        """
        return {k: list(self.output_shapes[k]) for k in self.output_shapes}

    def embed_timesteps(self, embeddings):
        """
        Computes timestep-based embeddings (aka positional embeddings) to add to embeddings.
        Args:
            embeddings (torch.Tensor): embeddings prior to positional embeddings are computed
        Returns:
            time_embeddings (torch.Tensor): positional embeddings to add to embeddings
        """
        timesteps = (
            torch.arange(
                0,
                embeddings.shape[1],
                dtype=embeddings.dtype,
                device=embeddings.device,
            )
            .unsqueeze(0)
            .repeat(embeddings.shape[0], 1)
        )
        assert (timesteps >= 0.0).all(), "timesteps must be positive!"
        if self.transformer_sinusoidal_embedding:
            assert torch.is_floating_point(timesteps), timesteps.dtype
        else:
            timesteps = timesteps.long()

        if self.transformer_nn_parameter_for_timesteps:
            time_embeddings = self.params["embed_timestep"]
        else:
            time_embeddings = self.nets["embed_timestep"](
                timesteps
            )  # these are NOT fed into transformer, only added to the inputs.
            # compute how many modalities were combined into embeddings, replicate time embeddings that many times
            num_replicates = embeddings.shape[-1] // self.transformer_embed_dim
            time_embeddings = torch.cat(
                [time_embeddings for _ in range(num_replicates)], -1
            )
            assert (
                embeddings.shape == time_embeddings.shape
            ), f"{embeddings.shape}, {time_embeddings.shape}"
        return time_embeddings

    def input_embedding(
        self,
        inputs,
    ):
        """
        Process encoded observations into embeddings to pass to transformer,
        Adds timestep-based embeddings (aka positional embeddings) to inputs.
        Args:
            inputs (torch.Tensor): outputs from observation encoder
        Returns:
            embeddings (torch.Tensor): input embeddings to pass to transformer backbone.
        """
        embeddings = self.nets["embed_encoder"](inputs)
        time_embeddings = self.embed_timesteps(embeddings)
        embeddings = embeddings + time_embeddings
        embeddings = self.nets["embed_ln"](embeddings)
        embeddings = self.nets["embed_drop"](embeddings)

        return embeddings

    def forward(self, **inputs):
        """
        Process each set of inputs in its own observation group.
        Args:
            inputs (dict): a dictionary of dictionaries with one dictionary per
                observation group. Each observation group's dictionary should map
                modality to torch.Tensor batches. Should be consistent with
                @self.input_obs_group_shapes. First two leading dimensions should
                be batch and time [B, T, ...] for each tensor.
        Returns:
            outputs (dict): dictionary of output torch.Tensors, that corresponds
                to @self.output_shapes. Leading dimensions will be batch and time [B, T, ...]
                for each tensor.
        """
        for obs_group in self.input_obs_group_shapes:
            for k in self.input_obs_group_shapes[obs_group]:
                # first two dimensions should be [B, T] for inputs
                if inputs[obs_group][k] is None:
                    continue
                assert inputs[obs_group][k].ndim - 2 == len(
                    self.input_obs_group_shapes[obs_group][k]
                )

        inputs = inputs.copy()

        transformer_encoder_outputs = None
        transformer_inputs = TensorUtils.time_distributed(
            inputs, self.nets["encoder"], inputs_as_kwargs=True
        )
        assert transformer_inputs.ndim == 3  # [B, T, D]

        if transformer_encoder_outputs is None:
            transformer_embeddings = self.input_embedding(transformer_inputs)
            # pass encoded sequences through transformer
            transformer_encoder_outputs = self.nets["transformer"].forward(
                transformer_embeddings
            )

        transformer_outputs = transformer_encoder_outputs
        # apply decoder to each timestep of sequence to get a dictionary of outputs
        transformer_outputs = TensorUtils.time_distributed(
            transformer_outputs, self.nets["decoder"]
        )
        transformer_outputs["transformer_encoder_outputs"] = transformer_encoder_outputs
        return transformer_outputs

    def _to_string(self):
        """
        Subclasses should override this method to print out info about network / policy.
        """
        return ""

    def __repr__(self):
        """Pretty print network."""
        header = "{}".format(str(self.__class__.__name__))
        msg = ""
        indent = " " * 4
        if self._to_string() != "":
            msg += textwrap.indent("\n" + self._to_string() + "\n", indent)
        msg += textwrap.indent("\nencoder={}".format(self.nets["encoder"]), indent)
        msg += textwrap.indent(
            "\n\ntransformer={}".format(self.nets["transformer"]), indent
        )
        msg += textwrap.indent("\n\ndecoder={}".format(self.nets["decoder"]), indent)
        msg = header + "(" + msg + "\n)"
        return msg


class MIMO_MCR_Transformer(Module):
    """
    Extension to Transformer (based on GPT architecture) to accept multiple observation
    dictionaries as input and to output dictionaries of tensors. Inputs are specified as
    a dictionary of observation dictionaries, with each key corresponding to an observation group.
    This module utilizes @ObservationGroupEncoder to process the multiple input dictionaries and
    @ObservationDecoder to generate tensor dictionaries. The default behavior
    for encoding the inputs is to process visual inputs with a learned CNN and concatenating
    the flat encodings with the other flat inputs. The default behavior for generating
    outputs is to use a linear layer branch to produce each modality separately
    (including visual outputs).
    """

    def __init__(
        self,
        input_obs_group_shapes,
        output_shapes,
        transformer_embed_dim,
        transformer_num_layers,
        transformer_num_heads,
        transformer_context_length,
        transformer_causal=True,
        transformer_emb_dropout=0.1,
        transformer_attn_dropout=0.1,
        transformer_block_output_dropout=0.1,
        transformer_sinusoidal_embedding=False,
        transformer_activation="gelu",
        transformer_nn_parameter_for_timesteps=False,
        encoder_kwargs=None,
    ):
        """
        Args:
            input_obs_group_shapes (OrderedDict): a dictionary of dictionaries.
                Each key in this dictionary should specify an observation group, and
                the value should be an OrderedDict that maps modalities to
                expected shapes.
            output_shapes (OrderedDict): a dictionary that maps modality to
                expected shapes for outputs.
            transformer_embed_dim (int): dimension for embeddings used by transformer
            transformer_num_layers (int): number of transformer blocks to stack
            transformer_num_heads (int): number of attention heads for each
                transformer block - must divide @transformer_embed_dim evenly. Self-attention is
                computed over this many partitions of the embedding dimension separately.
            transformer_causal (bool): whether to use causal transformer layers
            transformer_context_length (int): expected length of input sequences
            transformer_activation: non-linearity for input and output layers used in transformer
            transformer_emb_dropout (float): dropout probability for embedding inputs in transformer
            transformer_attn_dropout (float): dropout probability for attention outputs for each transformer block
            transformer_block_output_dropout (float): dropout probability for final outputs for each transformer block
            encoder_kwargs (dict): observation encoder config
        """
        super(MIMO_MCR_Transformer, self).__init__()

        assert isinstance(input_obs_group_shapes, OrderedDict)
        assert np.all(
            [
                isinstance(input_obs_group_shapes[k], OrderedDict)
                for k in input_obs_group_shapes
            ]
        )
        assert isinstance(output_shapes, OrderedDict)

        self.input_obs_group_shapes = input_obs_group_shapes
        self.output_shapes = output_shapes

        self.nets = nn.ModuleDict()
        self.params = nn.ParameterDict()

        # Encoder for all observation groups.
        self.nets["encoder"] = MCRObservationGroupEncoder(
            observation_group_shapes=input_obs_group_shapes,
            encoder_kwargs=encoder_kwargs,
            feature_activation=None,
            mcr_model=MIMO_MCR_Transformer.mcr_model,
        )

        # flat encoder output dimension
        transformer_input_dim = self.nets["encoder"].output_shape()[0]
        self.nets["embed_encoder"] = nn.Linear(
            transformer_input_dim, transformer_embed_dim
        )

        max_timestep = transformer_context_length

        if transformer_sinusoidal_embedding:
            self.nets["embed_timestep"] = PositionalEncoding(transformer_embed_dim)
        elif transformer_nn_parameter_for_timesteps:
            assert (
                not transformer_sinusoidal_embedding
            ), "nn.Parameter only works with learned embeddings"
            self.params["embed_timestep"] = nn.Parameter(
                torch.zeros(1, max_timestep, transformer_embed_dim)
            )
        else:
            self.nets["embed_timestep"] = nn.Embedding(
                max_timestep, transformer_embed_dim
            )

        # layer norm for embeddings
        self.nets["embed_ln"] = nn.LayerNorm(transformer_embed_dim)

        # dropout for input embeddings
        self.nets["embed_drop"] = nn.Dropout(transformer_emb_dropout)

        # GPT transformer
        self.nets["transformer"] = GPT_Backbone(
            embed_dim=transformer_embed_dim,
            num_layers=transformer_num_layers,
            num_heads=transformer_num_heads,
            context_length=transformer_context_length,
            causal=transformer_causal,
            attn_dropout=transformer_attn_dropout,
            block_output_dropout=transformer_block_output_dropout,
            activation=transformer_activation,
        )

        # decoder for output modalities
        self.nets["decoder"] = ObservationDecoder(
            decode_shapes=self.output_shapes,
            input_feat_dim=transformer_embed_dim,
        )

        self.transformer_context_length = transformer_context_length
        self.transformer_embed_dim = transformer_embed_dim
        self.transformer_sinusoidal_embedding = transformer_sinusoidal_embedding
        self.transformer_nn_parameter_for_timesteps = (
            transformer_nn_parameter_for_timesteps
        )

    @classmethod
    def set_mcr_model(cls, mcr_model):
        cls.mcr_model = mcr_model

    def output_shape(self, input_shape=None):
        """
        Returns output shape for this module, which is a dictionary instead
        of a list since outputs are dictionaries.
        """
        return {k: list(self.output_shapes[k]) for k in self.output_shapes}

    def embed_timesteps(self, embeddings):
        """
        Computes timestep-based embeddings (aka positional embeddings) to add to embeddings.
        Args:
            embeddings (torch.Tensor): embeddings prior to positional embeddings are computed
        Returns:
            time_embeddings (torch.Tensor): positional embeddings to add to embeddings
        """
        timesteps = (
            torch.arange(
                0,
                embeddings.shape[1],
                dtype=embeddings.dtype,
                device=embeddings.device,
            )
            .unsqueeze(0)
            .repeat(embeddings.shape[0], 1)
        )
        assert (timesteps >= 0.0).all(), "timesteps must be positive!"
        if self.transformer_sinusoidal_embedding:
            assert torch.is_floating_point(timesteps), timesteps.dtype
        else:
            timesteps = timesteps.long()

        if self.transformer_nn_parameter_for_timesteps:
            time_embeddings = self.params["embed_timestep"]
        else:
            time_embeddings = self.nets["embed_timestep"](
                timesteps
            )  # these are NOT fed into transformer, only added to the inputs.
            # compute how many modalities were combined into embeddings, replicate time embeddings that many times
            num_replicates = embeddings.shape[-1] // self.transformer_embed_dim
            time_embeddings = torch.cat(
                [time_embeddings for _ in range(num_replicates)], -1
            )
            assert (
                embeddings.shape == time_embeddings.shape
            ), f"{embeddings.shape}, {time_embeddings.shape}"
        return time_embeddings

    def input_embedding(
        self,
        inputs,
    ):
        """
        Process encoded observations into embeddings to pass to transformer,
        Adds timestep-based embeddings (aka positional embeddings) to inputs.
        Args:
            inputs (torch.Tensor): outputs from observation encoder
        Returns:
            embeddings (torch.Tensor): input embeddings to pass to transformer backbone.
        """
        embeddings = self.nets["embed_encoder"](inputs)
        time_embeddings = self.embed_timesteps(embeddings)
        embeddings = embeddings + time_embeddings
        embeddings = self.nets["embed_ln"](embeddings)
        embeddings = self.nets["embed_drop"](embeddings)

        return embeddings

    def forward(self, **inputs):
        """
        Process each set of inputs in its own observation group.
        Args:
            inputs (dict): a dictionary of dictionaries with one dictionary per
                observation group. Each observation group's dictionary should map
                modality to torch.Tensor batches. Should be consistent with
                @self.input_obs_group_shapes. First two leading dimensions should
                be batch and time [B, T, ...] for each tensor.
        Returns:
            outputs (dict): dictionary of output torch.Tensors, that corresponds
                to @self.output_shapes. Leading dimensions will be batch and time [B, T, ...]
                for each tensor.
        """
        for obs_group in self.input_obs_group_shapes:
            for k in self.input_obs_group_shapes[obs_group]:
                # first two dimensions should be [B, T] for inputs
                if inputs[obs_group][k] is None:
                    continue
                assert inputs[obs_group][k].ndim - 2 == len(
                    self.input_obs_group_shapes[obs_group][k]
                )

        inputs = inputs.copy()

        transformer_encoder_outputs = None
        transformer_inputs = TensorUtils.mcr_time_distributed(
            inputs,
            self.nets["encoder"],
            MIMO_MCR_Transformer.mcr_model,
            inputs_as_kwargs=True,
        )
        assert transformer_inputs.ndim == 3  # [B, T, D]

        if transformer_encoder_outputs is None:
            transformer_embeddings = self.input_embedding(transformer_inputs)
            # pass encoded sequences through transformer
            transformer_encoder_outputs = self.nets["transformer"].forward(
                transformer_embeddings
            )

        transformer_outputs = transformer_encoder_outputs
        # apply decoder to each timestep of sequence to get a dictionary of outputs
        transformer_outputs = TensorUtils.time_distributed(
            transformer_outputs, self.nets["decoder"]
        )
        transformer_outputs["transformer_encoder_outputs"] = transformer_encoder_outputs
        return transformer_outputs

    def _to_string(self):
        """
        Subclasses should override this method to print out info about network / policy.
        """
        return ""

    def __repr__(self):
        """Pretty print network."""
        header = "{}".format(str(self.__class__.__name__))
        msg = ""
        indent = " " * 4
        if self._to_string() != "":
            msg += textwrap.indent("\n" + self._to_string() + "\n", indent)
        msg += textwrap.indent("\nencoder={}".format(self.nets["encoder"]), indent)
        msg += textwrap.indent(
            "\n\ntransformer={}".format(self.nets["transformer"]), indent
        )
        msg += textwrap.indent("\n\ndecoder={}".format(self.nets["decoder"]), indent)
        msg = header + "(" + msg + "\n)"
        return msg


class ICL_MIMO_Transformer(Module):
    """
    Extension to Transformer (based on GPT architecture) to accept multiple observation
    dictionaries as input and to output dictionaries of tensors. Inputs are specified as
    a dictionary of observation dictionaries, with each key corresponding to an observation group.
    This module utilizes @ObservationGroupEncoder to process the multiple input dictionaries and
    @ObservationDecoder to generate tensor dictionaries. The default behavior
    for encoding the inputs is to process visual inputs with a learned CNN and concatenating
    the flat encodings with the other flat inputs. The default behavior for generating
    outputs is to use a linear layer branch to produce each modality separately
    (including visual outputs).
    """

    def __init__(
        self,
        input_obs_group_shapes,
        output_shapes,
        transformer_embed_dim,
        transformer_num_layers,
        transformer_num_heads,
        transformer_context_length,
        transformer_causal=True,
        transformer_emb_dropout=0.1,
        transformer_attn_dropout=0.1,
        transformer_block_output_dropout=0.1,
        transformer_sinusoidal_embedding=False,
        transformer_activation="gelu",
        transformer_fast_enabled=False,
        transformer_bin_enabled=False,
        transformer_vq_vae_enabled=False,
        transformer_ln_act_enabled=False,
        transformer_nn_parameter_for_timesteps=False,
        encoder_kwargs=None,
    ):
        """
        Args:
            input_obs_group_shapes (OrderedDict): a dictionary of dictionaries.
                Each key in this dictionary should specify an observation group, and
                the value should be an OrderedDict that maps modalities to
                expected shapes.
            output_shapes (OrderedDict): a dictionary that maps modality to
                expected shapes for outputs.
            transformer_embed_dim (int): dimension for embeddings used by transformer
            transformer_num_layers (int): number of transformer blocks to stack
            transformer_num_heads (int): number of attention heads for each
                transformer block - must divide @transformer_embed_dim evenly. Self-attention is
                computed over this many partitions of the embedding dimension separately.
            transformer_causal (bool): whether to use causal transformer layers
            transformer_context_length (int): expected length of input sequences
            transformer_activation: non-linearity for input and output layers used in transformer
            transformer_emb_dropout (float): dropout probability for embedding inputs in transformer
            transformer_attn_dropout (float): dropout probability for attention outputs for each transformer block
            transformer_block_output_dropout (float): dropout probability for final outputs for each transformer block
            encoder_kwargs (dict): observation encoder config
        """
        super(ICL_MIMO_Transformer, self).__init__()

        assert isinstance(input_obs_group_shapes, OrderedDict)
        assert np.all(
            [
                isinstance(input_obs_group_shapes[k], OrderedDict)
                for k in input_obs_group_shapes
            ]
        )
        assert isinstance(output_shapes, OrderedDict)

        self.input_obs_group_shapes = input_obs_group_shapes
        self.output_shapes = output_shapes

        self.nets = nn.ModuleDict()
        self.params = nn.ParameterDict()

        # Encoder for all observation groups.
        self.nets["encoder"] = ICLObservationGroupEncoder(
            observation_group_shapes=input_obs_group_shapes,
            action_input_shape=12,  # FIXME
            fast_enabled=transformer_fast_enabled,
            bin_enabled=transformer_bin_enabled,
            vq_vae_enabled=transformer_vq_vae_enabled,
            ln_act_enabled=transformer_ln_act_enabled,
            encoder_kwargs=encoder_kwargs,
            feature_activation=None,
        )

        self.vq_vae_enabled = transformer_vq_vae_enabled
        if transformer_vq_vae_enabled:
            self.vq_vae_model = self.nets["encoder"].action_network

        # flat encoder output dimension
        transformer_input_dim = self.nets["encoder"].output_shape()[0]
        self.nets["embed_encoder"] = nn.Linear(
            transformer_input_dim, transformer_embed_dim
        )

        max_timestep = transformer_context_length

        if transformer_sinusoidal_embedding:
            self.nets["embed_timestep"] = PositionalEncoding(transformer_embed_dim)
        elif transformer_nn_parameter_for_timesteps:
            assert (
                not transformer_sinusoidal_embedding
            ), "nn.Parameter only works with learned embeddings"
            self.params["embed_timestep"] = nn.Parameter(
                torch.zeros(1, max_timestep, transformer_embed_dim)
            )
        else:
            self.nets["embed_timestep"] = nn.Embedding(
                max_timestep, transformer_embed_dim
            )

        # layer norm for embeddings
        self.nets["embed_ln"] = nn.LayerNorm(transformer_embed_dim)

        # dropout for input embeddings
        self.nets["embed_drop"] = nn.Dropout(transformer_emb_dropout)

        # GPT transformer
        self.nets["transformer"] = GPT_Backbone(
            embed_dim=transformer_embed_dim,
            num_layers=transformer_num_layers,
            num_heads=transformer_num_heads,
            context_length=transformer_context_length
            * 3,  # multiplying by 3 because of extra tokens for context embeddings
            causal=transformer_causal,
            attn_dropout=transformer_attn_dropout,
            block_output_dropout=transformer_block_output_dropout,
            activation=transformer_activation,
        )

        # decoder for output modalities
        self.nets["decoder"] = ObservationDecoder(
            decode_shapes=self.output_shapes,
            input_feat_dim=transformer_embed_dim,
        )

        self.transformer_context_length = transformer_context_length
        self.transformer_embed_dim = transformer_embed_dim
        self.transformer_sinusoidal_embedding = transformer_sinusoidal_embedding
        self.transformer_nn_parameter_for_timesteps = (
            transformer_nn_parameter_for_timesteps
        )

    def output_shape(self, input_shape=None):
        """
        Returns output shape for this module, which is a dictionary instead
        of a list since outputs are dictionaries.
        """
        return {k: list(self.output_shapes[k]) for k in self.output_shapes}

    def embed_timesteps(self, embeddings):
        """
        Computes timestep-based embeddings (aka positional embeddings) to add to embeddings.
        Args:
            embeddings (torch.Tensor): embeddings prior to positional embeddings are computed
        Returns:
            time_embeddings (torch.Tensor): positional embeddings to add to embeddings
        """
        timesteps = (
            torch.arange(
                0,
                embeddings.shape[1],
                dtype=embeddings.dtype,
                device=embeddings.device,
            )
            .unsqueeze(0)
            .repeat(embeddings.shape[0], 1)
        )
        assert (timesteps >= 0.0).all(), "timesteps must be positive!"
        if self.transformer_sinusoidal_embedding:
            assert torch.is_floating_point(timesteps), timesteps.dtype
        else:
            timesteps = timesteps.long()

        if self.transformer_nn_parameter_for_timesteps:
            time_embeddings = self.params["embed_timestep"]
        else:
            time_embeddings = self.nets["embed_timestep"](
                timesteps
            )  # these are NOT fed into transformer, only added to the inputs.
            # compute how many modalities were combined into embeddings, replicate time embeddings that many times
            num_replicates = embeddings.shape[-1] // self.transformer_embed_dim
            time_embeddings = torch.cat(
                [time_embeddings for _ in range(num_replicates)], -1
            )
            assert (
                embeddings.shape == time_embeddings.shape
            ), f"{embeddings.shape}, {time_embeddings.shape}"
        return time_embeddings

    def input_embedding(
        self,
        inputs,
    ):
        """
        Process encoded observations into embeddings to pass to transformer,
        Adds timestep-based embeddings (aka positional embeddings) to inputs.
        Args:
            inputs (torch.Tensor): outputs from observation encoder
        Returns:
            embeddings (torch.Tensor): input embeddings to pass to transformer backbone.
        """
        embeddings = self.nets["embed_encoder"](inputs)
        time_embeddings = self.embed_timesteps(embeddings)
        embeddings = embeddings + time_embeddings
        embeddings = self.nets["embed_ln"](embeddings)
        embeddings = self.nets["embed_drop"](embeddings)

        return embeddings

    def forward(self, **inputs):
        """
        Process each set of inputs in its own observation group.
        Args:
            inputs (dict): a dictionary of dictionaries with one dictionary per
                observation group. Each observation group's dictionary should map
                modality to torch.Tensor batches. Should be consistent with
                @self.input_obs_group_shapes. First two leading dimensions should
                be batch and time [B, T, ...] for each tensor.
        Returns:
            outputs (dict): dictionary of output torch.Tensors, that corresponds
                to @self.output_shapes. Leading dimensions will be batch and time [B, T, ...]
                for each tensor.
        """
        for obs_group in self.input_obs_group_shapes:
            for k in self.input_obs_group_shapes[obs_group]:
                # first two dimensions should be [B, T] for inputs
                if inputs[obs_group][k] is None:
                    continue
                assert inputs[obs_group][k].ndim - 2 == len(
                    self.input_obs_group_shapes[obs_group][k]
                )

        inputs = inputs.copy()

        transformer_encoder_outputs = None
        obs, context_obs, context_actions = TensorUtils.icl_time_distributed(
            inputs, self.nets["encoder"], inputs_as_kwargs=True
        )
        assert obs.ndim == 3  # [B, T, D]

        if self.vq_vae_enabled:
            self._vq_vae_loss = self.nets["encoder"]._vq_vae_loss

        if transformer_encoder_outputs is None:
            obs_embeddings = self.input_embedding(obs)
            context_obs_embeddings = self.input_embedding(context_obs)
            context_actions_embeddings = self.input_embedding(context_actions)

            bs, timestep, D = obs_embeddings.shape
            # Step 1: Interleave context_obs and context_actions
            interleaved_context = torch.stack(
                [context_obs_embeddings, context_actions_embeddings], dim=2
            )  # [bs, timestep, 2, D]
            interleaved_context = interleaved_context.view(
                bs, -1, D
            )  # [bs, 2 * timestep, D]

            # Step 2: Concatenate interleaved context with obs
            transformer_embeddings = torch.cat(
                [interleaved_context, obs_embeddings], dim=1
            )  # [bs, 3 * timestep, D]
            # pass encoded sequences through transformer
            transformer_encoder_outputs = self.nets["transformer"].forward(
                transformer_embeddings
            )

        transformer_outputs = transformer_encoder_outputs[
            :,
            -self.transformer_context_length :,
        ]
        # apply decoder to each timestep of sequence to get a dictionary of outputs
        transformer_outputs = TensorUtils.time_distributed(
            transformer_outputs, self.nets["decoder"]
        )
        transformer_outputs["transformer_encoder_outputs"] = transformer_encoder_outputs
        return transformer_outputs

    def _to_string(self):
        """
        Subclasses should override this method to print out info about network / policy.
        """
        return ""

    def __repr__(self):
        """Pretty print network."""
        header = "{}".format(str(self.__class__.__name__))
        msg = ""
        indent = " " * 4
        if self._to_string() != "":
            msg += textwrap.indent("\n" + self._to_string() + "\n", indent)
        msg += textwrap.indent("\nencoder={}".format(self.nets["encoder"]), indent)
        msg += textwrap.indent(
            "\n\ntransformer={}".format(self.nets["transformer"]), indent
        )
        msg += textwrap.indent("\n\ndecoder={}".format(self.nets["decoder"]), indent)
        msg = header + "(" + msg + "\n)"
        return msg


class ICL_MIMO_Mamba(Module):
    """
    Extension to Mamba (based on GPT architecture) to accept multiple observation
    dictionaries as input and to output dictionaries of tensors. Inputs are specified as
    a dictionary of observation dictionaries, with each key corresponding to an observation group.
    This module utilizes @ObservationGroupEncoder to process the multiple input dictionaries and
    @ObservationDecoder to generate tensor dictionaries. The default behavior
    for encoding the inputs is to process visual inputs with a learned CNN and concatenating
    the flat encodings with the other flat inputs. The default behavior for generating
    outputs is to use a linear layer branch to produce each modality separately
    (including visual outputs).
    """

    def __init__(
        self,
        input_obs_group_shapes,
        output_shapes,
        mamba_embed_dim,
        mamba_num_layers,
        mamba_num_heads,
        mamba_context_length,
        mamba_causal=True,
        mamba_emb_dropout=0.1,
        mamba_attn_dropout=0.1,
        mamba_block_output_dropout=0.1,
        mamba_sinusoidal_embedding=False,
        mamba_activation="gelu",
        mamba_nn_parameter_for_timesteps=False,
        mamba_fast_enabled=False,
        mamba_bin_enabled=False,
        mamba_vq_vae_enabled=False,
        mamba_ln_act_enabled=False,
        encoder_kwargs=None,
    ):
        """
        Args:
            input_obs_group_shapes (OrderedDict): a dictionary of dictionaries.
                Each key in this dictionary should specify an observation group, and
                the value should be an OrderedDict that maps modalities to
                expected shapes.
            output_shapes (OrderedDict): a dictionary that maps modality to
                expected shapes for outputs.
            mamba_embed_dim (int): dimension for embeddings used by mamba
            mamba_num_layers (int): number of mamba blocks to stack
            mamba_num_heads (int): number of attention heads for each
                mamba block - must divide @transformer_embed_dim evenly. Self-attention is
                computed over this many partitions of the embedding dimension separately.
            mamba_causal (bool): whether to use causal mamba layers
            mamba_context_length (int): expected length of input sequences
            mambaactivation: non-linearity for input and output layers used in mamba
            mamba_emb_dropout (float): dropout probability for embedding inputs in mamba
            mamba_attn_dropout (float): dropout probability for attention outputs for each mamba block
            mamba_block_output_dropout (float): dropout probability for final outputs for each mamba block
            encoder_kwargs (dict): observation encoder config
        """
        super(ICL_MIMO_Mamba, self).__init__()

        assert isinstance(input_obs_group_shapes, OrderedDict)
        assert np.all(
            [
                isinstance(input_obs_group_shapes[k], OrderedDict)
                for k in input_obs_group_shapes
            ]
        )
        assert isinstance(output_shapes, OrderedDict)

        self.input_obs_group_shapes = input_obs_group_shapes
        self.output_shapes = output_shapes

        self.nets = nn.ModuleDict()
        self.params = nn.ParameterDict()

        # Encoder for all observation groups.
        self.nets["encoder"] = ICLObservationGroupEncoder(
            observation_group_shapes=input_obs_group_shapes,
            action_input_shape=12,  # FIXME
            fast_enabled=mamba_fast_enabled,
            bin_enabled=mamba_bin_enabled,
            vq_vae_enabled=mamba_vq_vae_enabled,
            ln_act_enabled=mamba_ln_act_enabled,
            encoder_kwargs=encoder_kwargs,
            feature_activation=None,
        )

        self.vq_vae_enabled = mamba_vq_vae_enabled
        if self.vq_vae_enabled:
            self.vq_vae_model = self.nets["encoder"].action_network

        # flat encoder output dimension
        mamba_input_dim = self.nets["encoder"].output_shape()[0]
        self.nets["embed_encoder"] = nn.Linear(mamba_input_dim, mamba_embed_dim)

        max_timestep = mamba_context_length

        if mamba_sinusoidal_embedding:
            self.nets["embed_timestep"] = PositionalEncoding(mamba_embed_dim)
        elif mamba_nn_parameter_for_timesteps:
            assert (
                not mamba_sinusoidal_embedding
            ), "nn.Parameter only works with learned embeddings"
            self.params["embed_timestep"] = nn.Parameter(
                torch.zeros(1, max_timestep, mamba_embed_dim)
            )
        else:
            self.nets["embed_timestep"] = nn.Embedding(max_timestep, mamba_embed_dim)

        # layer norm for embeddings
        self.nets["embed_ln"] = nn.LayerNorm(mamba_embed_dim)

        # dropout for input embeddings
        self.nets["embed_drop"] = nn.Dropout(mamba_emb_dropout)

        # Mamba backbone
        self.nets["mamba"] = Mamba(
            d_model=mamba_embed_dim,  # Model dimension d_model
            d_state=mamba_num_heads,  # SSM state expansion factor
            d_conv=4,  # Local convolution width
            expand=mamba_num_layers,  # Block expansion factor
        )

        # decoder for output modalities
        self.nets["decoder"] = ObservationDecoder(
            decode_shapes=self.output_shapes,
            input_feat_dim=mamba_embed_dim,
        )

        self.mamba_context_length = mamba_context_length
        self.mamba_embed_dim = mamba_embed_dim
        self.mamba_sinusoidal_embedding = mamba_sinusoidal_embedding
        self.mamba_nn_parameter_for_timesteps = mamba_nn_parameter_for_timesteps

    def output_shape(self, input_shape=None):
        """
        Returns output shape for this module, which is a dictionary instead
        of a list since outputs are dictionaries.
        """
        return {k: list(self.output_shapes[k]) for k in self.output_shapes}

    def embed_timesteps(self, embeddings):
        """
        Computes timestep-based embeddings (aka positional embeddings) to add to embeddings.
        Args:
            embeddings (torch.Tensor): embeddings prior to positional embeddings are computed
        Returns:
            time_embeddings (torch.Tensor): positional embeddings to add to embeddings
        """
        timesteps = (
            torch.arange(
                0,
                embeddings.shape[1],
                dtype=embeddings.dtype,
                device=embeddings.device,
            )
            .unsqueeze(0)
            .repeat(embeddings.shape[0], 1)
        )
        assert (timesteps >= 0.0).all(), "timesteps must be positive!"
        if self.mamba_sinusoidal_embedding:
            assert torch.is_floating_point(timesteps), timesteps.dtype
        else:
            timesteps = timesteps.long()

        if self.mamba_nn_parameter_for_timesteps:
            time_embeddings = self.params["embed_timestep"]
        else:
            time_embeddings = self.nets["embed_timestep"](
                timesteps
            )  # these are NOT fed into mamba, only added to the inputs.
            # compute how many modalities were combined into embeddings, replicate time embeddings that many times
            num_replicates = embeddings.shape[-1] // self.mamba_embed_dim
            time_embeddings = torch.cat(
                [time_embeddings for _ in range(num_replicates)], -1
            )
            assert (
                embeddings.shape == time_embeddings.shape
            ), f"{embeddings.shape}, {time_embeddings.shape}"
        return time_embeddings

    def input_embedding(
        self,
        inputs,
    ):
        """
        Process encoded observations into embeddings to pass to mamba,
        Adds timestep-based embeddings (aka positional embeddings) to inputs.
        Args:
            inputs (torch.Tensor): outputs from observation encoder
        Returns:
            embeddings (torch.Tensor): input embeddings to pass to mamba backbone.
        """
        embeddings = self.nets["embed_encoder"](inputs)
        time_embeddings = self.embed_timesteps(embeddings)
        embeddings = embeddings + time_embeddings
        embeddings = self.nets["embed_ln"](embeddings)
        embeddings = self.nets["embed_drop"](embeddings)

        return embeddings

    def forward(self, **inputs):
        """
        Process each set of inputs in its own observation group.
        Args:
            inputs (dict): a dictionary of dictionaries with one dictionary per
                observation group. Each observation group's dictionary should map
                modality to torch.Tensor batches. Should be consistent with
                @self.input_obs_group_shapes. First two leading dimensions should
                be batch and time [B, T, ...] for each tensor.
        Returns:
            outputs (dict): dictionary of output torch.Tensors, that corresponds
                to @self.output_shapes. Leading dimensions will be batch and time [B, T, ...]
                for each tensor.
        """
        for obs_group in self.input_obs_group_shapes:
            for k in self.input_obs_group_shapes[obs_group]:
                # first two dimensions should be [B, T] for inputs
                if inputs[obs_group][k] is None:
                    continue
                assert inputs[obs_group][k].ndim - 2 == len(
                    self.input_obs_group_shapes[obs_group][k]
                )

        inputs = inputs.copy()

        mamba_encoder_outputs = None
        obs, context_obs, context_actions = TensorUtils.icl_time_distributed(
            inputs, self.nets["encoder"], inputs_as_kwargs=True
        )
        assert obs.ndim == 3  # [B, T, D]

        if self.vq_vae_enabled:
            self._vq_vae_loss = self.nets["encoder"]._vq_vae_loss

        if mamba_encoder_outputs is None:
            obs_embeddings = self.input_embedding(obs)
            context_obs_embeddings = self.input_embedding(context_obs)
            context_actions_embeddings = self.input_embedding(context_actions)

            bs, timestep, D = obs_embeddings.shape
            # Step 1: Interleave context_obs and context_actions
            interleaved_context = torch.stack(
                [context_obs_embeddings, context_actions_embeddings], dim=2
            )  # [bs, timestep, 2, D]
            interleaved_context = interleaved_context.view(
                bs, -1, D
            )  # [bs, 2 * timestep, D]

            # Step 2: Concatenate interleaved context with obs
            mamba_embeddings = torch.cat(
                [interleaved_context, obs_embeddings], dim=1
            )  # [bs, 3 * timestep, D]
            # pass encoded sequences through mamba
            mamba_encoder_outputs = self.nets["mamba"].forward(mamba_embeddings)

        mamba_outputs = mamba_encoder_outputs[
            :,
            -self.mamba_context_length :,
        ]
        # apply decoder to each timestep of sequence to get a dictionary of outputs
        mamba_outputs = TensorUtils.time_distributed(
            mamba_outputs, self.nets["decoder"]
        )
        mamba_outputs["mamba_encoder_outputs"] = mamba_encoder_outputs
        return mamba_outputs

    def _to_string(self):
        """
        Subclasses should override this method to print out info about network / policy.
        """
        return ""

    def __repr__(self):
        """Pretty print network."""
        header = "{}".format(str(self.__class__.__name__))
        msg = ""
        indent = " " * 4
        if self._to_string() != "":
            msg += textwrap.indent("\n" + self._to_string() + "\n", indent)
        msg += textwrap.indent("\nencoder={}".format(self.nets["encoder"]), indent)
        msg += textwrap.indent("\n\mamba={}".format(self.nets["mamba"]), indent)
        msg += textwrap.indent("\n\ndecoder={}".format(self.nets["decoder"]), indent)
        msg = header + "(" + msg + "\n)"
        return msg
