# Lightning Segmentation

### This repo is for practicing using our workflow.



## Setup:

### Conda environment

1. [Create](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) the conda environment `jtml` from the `environment.yml` using the command `conda create env create -f environment.yml`.
2. Activate the conda env with `conda activate jtml`.
3. There may be other dependencies that you can install using conda or pip3.

### [WandB](https://wandb.ai/) - our logging system.

1. Create an account from the website and send the email you used to Sasank (to get invited to the Wandb team).

## Use:

1. Be in the LitJTML directory (use the `cd` command to change the directory to the `blah/blah/LitJTML/` directory).
2. To fit (train) a model, call `python scripts/fit.py my_config` where `my_config` is the name of the config.py file in the `config/` directory.
    - The config file should specify the model, data, and other parameters.
