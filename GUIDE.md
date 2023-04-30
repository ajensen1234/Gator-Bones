# GatorBones
GatorBones makes use of transformer architecture for segmentation, but is otherwise similar to past lab projects in its use of Python and Pytorch Lightning.

### Branch-specific Setup on HiPerGator (Banks Only)

For HiPerGator, create a conda environment from hpg-environment.yml using `conda env create --file=hpg_environment.yml`. This should create an environment called `hpg`. It may take a while (and it uses `pip`) but it should eventually work. This is the environment that should be used in the training script (currently `training.sh`).

### Using the Architecture Builder

Model components are chosen via an architecture builder, which is located in the /models directory. The architecture builder consumes the backbone and head via ModelManager.py. You can specify these values in the config file. Additionally, heads and backbones make use of modules contained in /models/modules. This allows for modules to be reused between any added networks that are built from the same components.
