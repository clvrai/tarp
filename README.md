# Task-Induced Representation Learning

#### [[Project Website]](https://clvrai.github.io/tarp/) [[Paper]](https://openreview.net/forum?id=OzyXtIZAzFv)
[Jun Yamada](https://junjungoal.github.io/)<sup>1</sup>, [Karl Pertsch](https://kpertsch.github.io/)<sup>2</sup>,
[Anisha Gunjal](https://anisha2102.github.io/)<sup>2</sup>, [Joseph Lim](https://www.clvrai.com/)<sup>2</sup>

<sup>1</sup>A2I Lab, University of Oxford, 
<sup>2</sup>CLVR Lab, University of Southern California 

<!-- 
<a href="https://clvrai.github.io/tarp/">
<p align="center">
<img src="docs/resources/spirl_teaser.png" width="800">
</p>
</img></a> -->

This is the official PyTorch implementation of the paper "**Task-Induced Representation Learning**" (ICLR 2022).

## Requirements

- python 3.7+
- mujoco 2.0 (for RL experiments)
- MPI (for RL experiments)

## Installation Instructions

Create a virtual environment and install all required packages.
```
cd tarp
pip3 install virtualenv
virtualenv -p $(which python3) ./venv
source ./venv/bin/activate

# Install dependencies
sudo apt install cmake libboost-all-dev libsdl2-dev libfreetype6-dev libgl1-mesa-dev libglu1-mesa-dev libp
pip3 install -r requirements.txt
```

Set the environment variable that specifies the root experiment directory. For example:
```
mkdir ./experiments
export EXP_DIR=./experiments
export DATA_DIR=./datasets 
```

### CARLA Installation
```
wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.11.tar.gz

# add the following to your python path
export PYTHONPATH=$PYTHONPATH:/home/{username}/CARLA_0.9.11/PythonAPI
export PYTHONPATH=$PYTHONPATH:/home/{username}/CARLA_0.9.11/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:/home/{username}/CARLA_0.9.11/PythonAPI/carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg


# run CARLA server
cd ./CARLA_0.9.11
./CarlaUE4.sh -carla-rpc-port=2002 -fps=20 -carla-server
```


## Example Commands
All results will be stored in [WandB](https://wandb.ai/). Before running scripts, you need to set the wandb entity and project in `tarp/train.py` and `tarp/rl/train.py` for logging.

### Representation Pretraining
To train TARP-BC model in distracting DMControl, run:
```
python3 -m tarp/train.py --path tarp/configs/representation_pretraining/tarp_bc_mdl/distracting_control/walker --prefix TARP-BC --val_data_size 160
```
To train other models, you need to change the path for the argument of `--path`.

For training TARP-CQL in distracting DMControl, run:
```
python3 -m tarp/rl/multi_train.py --path tarp/config/representation_pretraining/tarp_cql/distracting_control/walker --prefix TARP-CQL --gpu 0
```

### Representation Transfer

For training a SAC agent on the distracting DMControl environment using the pre-trained encoder, run:
```
python3 tarp/rl/train.py --path=tarp/configs/rl/sac/distracting_control/walker/representation_transfer --prefix TARP-BC.seed123  --seed=123
```
Note that you need to replace a path of `encoder_checkpoint` argument with the experiment directory of the model training above in `conf.py`.

For training a multiprocessing (6 processes) PPO agent on ViZDoom, run:

```
mpirun -n 6 python tarp/rl/train.py --path tarp/configs/rl/ppo/vizdoom/representation_transfer --prefix=TARP-BC.seed123 --seed=123
```

## Code Structure Overview
```
tarp
  |- components            # reusable infrastructure for model training
  |    |- base_model.py    # basic model class that all models inherit from
  |    |- checkpointer.py  # handles storing + loading of model checkpoints
  |    |- data_loader.py   # basic dataset classes, new datasets need to inherit from here
  |    |- evaluator.py     # defines basic evaluation routines, eg top-of-N evaluation, + eval logging
  |    |- logger.py        # implements tarp logging functionality using tensorboardX
  |    |- params.py        # definition of command line params for model training
  |    |- trainer_base.py  # basic training utils used in main trainer file
  |
  |- configs               # all experiment configs should be placed here
  |    |- default_data_configs   # defines one default data config per dataset, e.g. state/action dim etc
  |
  |- data                  # any dataset-specific code should go here (like data generation scripts, custom loaders etc)
  |- models                # holds all model classes that implement forward, loss, visualization
  |- modules               # reusable architecture components (like MLPs, CNNs, LSTMs, Flows etc)
  |- rl                    # all code related to RL
  |    |- agents           # implements tarp algorithms in agent classes, like SAC etc
  |    |- components       # reusable infrastructure for RL experiments
  |        |- agent.py     # basic agent and hierarchial agent classes - do not implement any specific RL algo
  |        |- critic.py    # basic critic implementations (eg MLP-based critic)
  |        |- environment.py    # defines environment interface, basic gym env
  |        |- normalization.py  # observation normalization classes, only optional
  |        |- params.py    # definition of command line params for RL training
  |        |- policy.py    # basic policy interface definition
  |        |- replay_buffer.py  # simple numpy-array replay buffer, uniform sampling and versions
  |        |- sampler.py   # rollout sampler for collecting experience, for flat and hierarchical agents
  |    |- envs             # all custom RL environments should be defined here
  |    |- policies         # policy implementations go here, MLP-policy and RandomAction are implemented
  |    |- utils            # utilities for RL code like MPI, WandB related code
  |    |- train.py         # main RL training script, builds all components + runs training
  |
  |- utils                 # general utilities, pytorch / visualization utilities etc
  |- train.py              # main model training script, builds all components + runs training loop and logging
```

The general philosophy is that each new experiment gets a new config file that captures all hyperparameters etc. so that experiments
themselves are version controllable. **Command-line parameters should be reduced to a minimum.**


## Starting to Modify the Code
### Adding a new model architecture
Start by defining a model class in the `tarp/models` directory that inherits from the `BaseModel` class.
The new model needs to define the architecture in the constructor, implement the forward pass and loss functions,
as well as model-specific logging functionality if desired. For an example see `tarp/models/vae_mdl.py`.

Note, that most basic architecture components (MLPs, CNNs, LSTMs, Flow models etc) are defined in `tarp/modules` and can be
conveniently reused for easy architecture definitions. Below are some links to the most important classes.

|Component        | File         | Description |
|:------------- |:-------------|:-------------|
| MLP | [```Predictor```](tarp/modules/subnetworks.py#L33) | Basic N-layer fully-connected network. Defines number of inputs, outputs, layers and hidden units. |
| CNN-Encoder | [```ConvEncoder```](tarp/modules/subnetworks.py#L66) | Convolutional encoder, number of layers determined by input dimensionality (resolution halved per layer). Number of channels doubles per layer. Returns encoded vector + skip activations. |
| CNN-Decoder | [```ConvDecoder```](tarp/modules/subnetworks.py#L145) | Mirrors architecture of conv. encoder. Can take skip connections as input, also versions that copy pixels etc. |
| Processing-LSTM | [```BaseProcessingLSTM```](tarp/modules/recurrent_modules.py#L70) | Basic N-layer LSTM for processing an input sequence. Produces one output per timestep, number of layers / hidden size configurable.|
| Prediction-LSTM | [```RecurrentPredictor```](tarp/modules/recurrent_modules.py#L241) | Same as processing LSTM, but for autoregressive prediction. |
| Mixture-Density Network | [```MDN```](tarp/modules/mdn.py#L10) | MLP that outputs GMM distribution. |
| Normalizing Flow Model | [```NormalizingFlowModel```](tarp/modules/flow_models.py#L9) | Implements normalizing flow model that stacks multiple flow blocks. Implementation for RealNVP block provided. |

### Adding a new dataset for model training
All code that is dataset-specific should be placed in a corresponding subfolder in `tarp/data`.
To add a data loader for a new dataset, the `Dataset` classes from [```data_loader.py```](tarp/components/data_loader.py) need to be subclassed
and the `__getitem__` function needs to be overwritten to load a single data sample.

All datasets used with the codebase so far have been based on `HDF5` files. The `GlobalSplitDataset` provides functionality to read all
HDF5-files in a directory and split them in `train/val/test` based on percentages. The `VideoDataset` class provides
many functionalities for manipulating sequeces, like randomly cropping subsequences, padding etc.

### Adding a new RL algorithm
The core RL algorithms are implemented within the `Agent` class. For adding a new algorithm, a new file needs to be created in
`tarp/rl/agents` and [```BaseAgent```](tarp/rl/components/agent.py#L19) needs to be subclassed. In particular, any required
networks (actor, critic etc) need to be constructed and the `update(...)` function needs to be overwritten.

### Adding a new RL environment
To add a new RL environment, simply define a new environent class in `tarp/rl/envs` that inherits from the environment interface
in [```tarp/rl/components/environment.py```](tarp/rl/components/environment.py).


## References
- Base implementation: https://github.com/clvrai/spirl


## Citation
If you find this work useful in your research, please consider citing:
```
@inproceedings{yamada2022tarp,
    title={Task-Induced Representation Learning},
    author={Jun Yamada and Karl Pertsch and Anisha Gunjal and Joseph J Lim},
    booktitle={International Conference on Learning Representations},
    year={2022},
}
```
