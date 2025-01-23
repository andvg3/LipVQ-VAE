# CUDA_VISIBLE_DEVICES=2,3 python train_representation.py hydra/launcher=local \
#         hydra/output=local agent.decode_state_weight=1.0 agent.size=50 experiment=<exp_name> \
#         doaug=rctraj batch_size=32 datapath=<path_to_dataset> \
#         wandbuser=<your_username> wandbproject=<your_proj> use_wandb=false

CUDA_VISIBLE_DEVICES=6 python train_representation.py hydra/launcher=local \
        hydra/output=local agent.bc_weight=1.0 agent.align_state_weight=1.0 agent.size=50 agent.use_action=true experiment=test \
        doaug=rctraj batch_size=32 datapath=/localdata_ssd/yifei/droid_processed \
        wandbuser=luccachiang wandbproject=robo_feature use_wandb=false