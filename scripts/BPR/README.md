# Boundary Patch Refinement (BPR)
BPR is a post-processing framework that takes in the results of a segmentation model and refines their boundary qualities. The main repository for BPR can be viewed [here](https://github.com/chenhang98/BPR). Additional details about BPR's original implementation can be read in its [paper](https://arxiv.org/abs/2104.05239).

## Structure
We have kept a subset of BPR's code in this repository under scripts/BPR. Particular attention should be paid to the demo, mmseg, and tools subdirectories.

The demo folder contains the inference_img.ipynb notebook, which can be used to test BPR on a single image. Input and output examples from the BoneMeal group is also included.

The mmseg folder contains BPR's architecture, which comes from the [mmsegmentation codebase](https://github.com/open-mmlab/mmsegmentation). 

The tools folder contains the Python scripts used to run BPR, including its train and test files.

## Existing Work
For our project, we have focused on modifying the encoder_decoder files under mmseg/models/segmentors to better fit with our code. In the BPR/tools directory, modifications have also been made to the train.py file. Our goal is to make this file more compatible with our established training process and with WandB.

## Running
1. Go to scripts/BPR/tools/train.py and change line 185 to your config directory
2. Run 'python BPR/tools/train.py' from the scripts directory
3. Run 'BPR/tools/train config' from the scripts directory

Current error: forward_train() got multiple values for argument 'img'