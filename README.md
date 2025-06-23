# Action Tokenizer Matters in In-Context Imitation Learning

This is the official codebase of the paper "Action Tokenizer Matters in In-Context Imitation Learning."

-------
## Installation
Below is a brief explanation for setting up RoboCasa. For further instructions, please refer to [RoboCasa](https://robocasa.ai/).
1. Set up conda environment:

   ```sh
   conda create -c conda-forge -n lipvq python=3.10
   ```
2. Activate conda environment:
   ```sh
   conda activate lipvq
   ```
3. Clone and setup robosuite dependency (**important: use the master branch!**):

   ```sh
   git clone https://github.com/ARISE-Initiative/robosuite
   cd robosuite
   pip install -e .
   ```
4. Clone and setup this repo:

   ```sh
   cd ..
   git clone https://github.com/andvg3/LipVQ-VAE.git
   cd robocasa
   pip install -e .
   pip install pre-commit; pre-commit install           # Optional: set up code formatter.

   (optional: if running into issues with numba/numpy, run: conda install -c numba numba=0.56.4 -y)
   ```
5. Install the package and download assets:
   ```sh
   python robocasa/scripts/download_kitchen_assets.py   # Caution: Assets to be downloaded are around 5GB.
   python robocasa/scripts/setup_macros.py              # Set up system variables.
   ```

-------
## Download Datasets
Please refer to the [official documentation page](https://robocasa.ai/docs/introduction/overview.html) for information about tasks and assets, downloading datasets.

## Policy Learning

### Training
Each algorithm has its own config generator script. For example for ICRT+LipVQ-VAE policy run:
```
robomimic/scripts/config_gen/icl_xfmr_gen.py --name <experiment-name>
```
After running this script you just need to run the command(s) outputted.
**Note:** You can modify different types of action tokenizer in the outputted config in:

```json
"observation": "modalities": { "fast_enabled": false, "bin_enabled": false, "vq_vae_enabled": true, "ln_act_enabled": false } 
```

Change the config to your desired tokenizers to test.

### Weights
Weight are available at [this link](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/an_vuong_mbzuai_ac_ae/EYkH6i1UUhxHvEJ-wtG_wg0Bmer1_uFScjOXqoI2TjVtag?e=Ze0maL).

### Evaluation
Similar to training, run:
```
python robomimic/scripts/config_gen/eval_ckpt.py --ckpt <ckpt-path> --name <experiment-name>
```
then execute the scripts on the screeen.

-------
## Citation
This repository is largely based on [RoboCasa](https://github.com/robocasa/robocasa). If you find our code useful, please consider citing it:
```bibtex
@inproceedings{vuong2025action,
  title={Action Tokenizer Matters in In-Context Imitation Learning},
  author={Vuong, An Dinh and Vu, Minh Nhat and An, Dong and Reid, Ian},
  journal={IROS},
  year={2025}
}
```
