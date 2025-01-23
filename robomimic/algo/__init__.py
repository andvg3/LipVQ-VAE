from robomimic.algo.algo import (
    register_algo_factory_func,
    algo_name_to_factory_func,
    algo_factory,
    Algo,
    PolicyAlgo,
    ValueAlgo,
    PlannerAlgo,
    HierarchicalAlgo,
    RolloutPolicy,
    ICLRolloutPolicy,
)

# note: these imports are needed to register these classes in the global algo registry
from robomimic.algo.bc import BC, BC_Gaussian, BC_GMM, BC_VAE, BC_RNN, BC_RNN_GMM
from robomimic.algo.bcq import BCQ, BCQ_GMM, BCQ_Distributional
from robomimic.algo.cql import CQL
from robomimic.algo.iql import IQL
from robomimic.algo.gl import GL, GL_VAE, ValuePlanner
from robomimic.algo.hbc import HBC
from robomimic.algo.iris import IRIS
from robomimic.algo.td3_bc import TD3_BC
from robomimic.algo.diffusion_policy import DiffusionPolicyUNet
from robomimic.algo.act import ACT
from robomimic.algo.mcr_main import MCR_Transformer_GMM
from robomimic.algo.icl import ICL
from robomimic.algo.icl_mamba import ICLMamba
